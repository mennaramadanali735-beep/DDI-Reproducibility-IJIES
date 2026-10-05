
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from sklearn.metrics import accuracy_score, f1_score

features = Path(sys.argv[1])
pretrain_dir = Path(sys.argv[2])
output = pretrain_dir / "interaction_classifier"
output.mkdir(exist_ok=True)

SEED, EPOCHS, BATCH_SIZE = 44, 100, 256
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")
assert torch.cuda.is_available()

with np.load(
    pretrain_dir / "molecular_embeddings_1d2d.npz",
    allow_pickle=False,
) as saved:
    assert np.array_equal(saved["drug_ids"], np.arange(1660))
    x = torch.tensor(saved["x"], device=device)

splits = {
    name: pd.read_csv(features / f"{name}_pairs.csv")
    for name in ("train", "calibration", "test")
}
expected = {"train": 78736, "calibration": 13894, "test": 17896}
for name, df in splits.items():
    assert len(df) == expected[name]
    assert df["source_row"].tolist() == list(range(len(df)))
    assert df["label_id"].between(0, 83).all()

data = {
    name: torch.tensor(
        df[["drug1_id", "drug2_id", "label_id"]].to_numpy(),
        dtype=torch.long, device=device,
    )
    for name, df in splits.items()
}
train = data["train"]

dev_drugs = set(
    pd.concat([splits["train"], splits["calibration"]])[
        ["drug1_id", "drug2_id"]
    ].to_numpy().ravel()
)
test_drugs = set(
    splits["test"][["drug1_id", "drug2_id"]].to_numpy().ravel()
)
assert not dev_drugs & test_drugs

# Interleave forward and reverse training edges, as in the source.
edge_index = torch.stack(
    (train[:, :2], train[:, :2].flip(1)), dim=1
).reshape(-1, 2).T.contiguous()
edge_type = train[:, 2].repeat_interleave(2)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_o1 = RGCNConv(256, 64, num_relations=84)
        self.encoder_o2 = RGCNConv(64, 32, num_relations=84)
        self.attt = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.mlp = nn.Sequential(
            nn.Linear(704, 256), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(128, 84),
        )

    def encode(self):
        h1 = torch.relu(self.encoder_o1(x, edge_index, edge_type))
        h1 = nn.functional.dropout(h1, p=0.5, training=self.training)
        h2 = self.encoder_o2(h1, edge_index, edge_type)
        return torch.cat(
            (self.attt[0] * h1, self.attt[1] * h2, x), dim=1
        )

    def classify(self, z, pairs):
        return self.mlp(torch.cat(
            (z[pairs[:, 0]], z[pairs[:, 1]]), dim=1
        ))

model = Classifier().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=5e-4
)
criterion = nn.CrossEntropyLoss()

def save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

@torch.no_grad()
def evaluate(pairs):
    model.eval()
    z = model.encode()
    predictions = []
    for batch in pairs.split(BATCH_SIZE):
        logits = model.classify(z, batch)
        assert torch.isfinite(logits).all()
        predictions.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(predictions)

best_score, start_epoch, history = -1.0, 1, []
last_path = output / "last_checkpoint.pt"
best_path = output / "best_checkpoint.pt"

# Bind checkpoints to the exact data and training script.
import hashlib
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

identity = {
    "script": sha(Path(__file__)),
    "embeddings": sha(pretrain_dir / "molecular_embeddings_1d2d.npz"),
    "splits": {
        name: sha(features / f"{name}_pairs.csv") for name in splits
    },
    "torch": torch.__version__,
}

if last_path.exists():
    saved = torch.load(last_path, map_location="cpu", weights_only=False)
    assert saved["identity"] == identity
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    best_score = saved["best_score"]
    history = saved["history"]
    start_epoch = saved["epoch"] + 1
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    print("Resuming epoch:", start_epoch, flush=True)

cal_y = splits["calibration"]["label_id"].to_numpy()

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # Fixed saved row order; stochasticity comes from initialization/dropout.
    for step, batch in enumerate(train.split(BATCH_SIZE), 1):
        optimizer.zero_grad(set_to_none=True)
        z = model.encode()
        logits = model.classify(z, batch)
        loss = criterion(logits, batch[:, 2])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        )
        optimizer.step()
        total_loss += loss.item() * len(batch)

        if step == 1 or step % 100 == 0:
            print(
                f"Epoch {epoch:03d} | batch {step} | "
                f"loss {loss.item():.5f}", flush=True
            )

    cal_predictions = evaluate(data["calibration"])
    score = f1_score(
        cal_y, cal_predictions, labels=list(range(84)),
        average="macro", zero_division=0,
    )
    improved = score > best_score
    best_score = max(best_score, score)

    history.append({
        "epoch": epoch,
        "train_loss": total_loss / len(train),
        "cal_macro84": score,
    })

    checkpoint = {
        "identity": identity,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if improved:
        save_atomic(checkpoint, best_path)
    save_atomic(checkpoint, last_path)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

    print(
        f" Epoch {epoch} saved | CAL Macro84={score:.6f} | "
        f"Best={best_score:.6f}", flush=True
    )

best = torch.load(best_path, map_location="cpu", weights_only=False)
model.load_state_dict(best["model"])

reloaded_cal = f1_score(
    cal_y, evaluate(data["calibration"]), labels=list(range(84)),
    average="macro", zero_division=0,
)
gap = abs(reloaded_cal - best["best_score"])
assert gap < 1e-8, f"Checkpoint gap: {gap}"

test_predictions = evaluate(data["test"])
test_y = splits["test"]["label_id"].to_numpy()
observed = np.unique(test_y)
assert len(observed) == 64

result = {
    "model": "MHAFR_DDI_adapted",
    "seed": SEED,
    "best_epoch": best["epoch"],
    "cal_macro84": reloaded_cal,
    "checkpoint_gap": gap,
    "test_accuracy": accuracy_score(test_y, test_predictions),
    "test_macro64": f1_score(
        test_y, test_predictions, labels=observed,
        average="macro", zero_division=0,
    ),
    "test_macro84": f1_score(
        test_y, test_predictions, labels=list(range(84)),
        average="macro", zero_division=0,
    ),
    "test_weighted_f1": f1_score(
        test_y, test_predictions, average="weighted", zero_division=0
    ),
}
pd.DataFrame([result]).to_csv(output / "final_result.csv", index=False)

predictions = splits["test"].copy()
predictions["prediction"] = test_predictions
predictions.to_csv(output / "test_predictions.csv", index=False)

print("\n CLASSIFICATION COMPLETED:", result, flush=True)
print("Saved in:", output, flush=True)

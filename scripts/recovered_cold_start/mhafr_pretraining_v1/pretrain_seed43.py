
import os
import sys
import random
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

print("Loading libraries...", flush=True)
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

source = Path(sys.argv[1])
features = Path(sys.argv[2])
sys.path.insert(0, str(source))

assert torch.cuda.is_available()
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load definitions without entering the original __main__ block.
script_path = source / "pretrain_MHAFR-DDI.py"
code = script_path.read_text(encoding="utf-8")

old_line = 'os.environ["CUDA_LAUNCH_BLOCKING"] = "0,1"'
assert code.count(old_line) == 1
code = code.replace(
    old_line,
    'os.environ["CUDA_LAUNCH_BLOCKING"] = "0"',
)

namespace = {
    "__name__": "mhafr_smoke_definitions",
    "__file__": str(script_path),
}
exec(compile(code, str(script_path), "exec"), namespace)

namespace["preprocessor"] = namespace["PreprocessBatch"]()
device = namespace["device"]


import json
import hashlib
import time

run_root = Path(sys.argv[3])
run_dir = run_root / "seed_43"
run_dir.mkdir(exist_ok=True)

SEED = 43
EPOCHS = 30
BATCH_SIZE = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

split_path = run_root / "molecular_pretrain_split.csv"
split = pd.read_csv(split_path)
train_ids = split.loc[
    split.pretrain_partition.eq("train"), "drug_id"
].tolist()
val_ids = split.loc[
    split.pretrain_partition.eq("validation"), "drug_id"
].tolist()

assert len(train_ids) == 864 and len(val_ids) == 216
assert not set(train_ids) & set(val_ids)

identity = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "torch": torch.__version__,
    "split_hash": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "feature_metadata_hash": hashlib.sha256(
        (features / "input_metadata.json").read_bytes()
    ).hexdigest(),
    "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}

print("Loading saved molecular inputs...", flush=True)
graph_cache = {}

for drug_id in train_ids + val_ids:
    with np.load(
        features / "molecules" / f"{int(drug_id):05d}.npz",
        allow_pickle=False,
    ) as saved:
        assert saved["has_3d"].item()
        graph = Data(
            x=torch.from_numpy(saved["x"].copy()).long(),
            edge_index=torch.from_numpy(saved["edge_index"].copy()).long(),
            edge_attr=torch.from_numpy(saved["edge_attr"].copy()).long(),
            pos=torch.from_numpy(saved["positions"].copy()).float(),
        )
        graph.smiles = saved["token_ids"].copy()
        graph.mask = saved["token_mask"].copy()
        graph_cache[int(drug_id)] = graph

model = namespace["CL_model"]().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-5
)
low_amc = namespace["AMC_loss_low"]
high_amc = namespace["AMC_loss_high"]

def get_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

last_path = run_dir / "last_checkpoint.pt"
best_path = run_dir / "best_checkpoint.pt"

start_epoch = 1
best_loss = float("inf")
history = []

if last_path.exists():
    # Only load our own locally generated checkpoint.
    saved = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    assert saved["identity"] == identity, "Run configuration changed."
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    low_amc.last_loss = saved["low_history"].clone()
    high_amc.last_loss = saved["high_history"].clone()
    best_loss = saved["best_loss"]
    history = saved["history"]
    start_epoch = saved["epoch"] + 1
    set_rng(saved["rng"])
    print("Resuming at epoch:", start_epoch, flush=True)

def calculate_loss(outputs, epoch, training):
    assert len(outputs) == 12
    assert all(torch.isfinite(x).all() for x in outputs)

    low = torch.stack([
        outputs[0], outputs[4], outputs[1],
        outputs[5], outputs[8], outputs[10],
    ], dim=1)
    high = torch.stack([
        outputs[2], outputs[6], outputs[3],
        outputs[7], outputs[9], outputs[11],
    ], dim=1)

    ids = torch.arange(low.shape[0], device=device).float()
    loss = (
        low_amc(low, epoch, ids, train=training)
        + high_amc(high, epoch, ids, train=training)
    )
    assert torch.isfinite(loss), f"Nonfinite loss at epoch {epoch}"
    return loss

def run_pass(ids, epoch, training):
    model.train(training)
    total, count = 0.0, 0
    batches = [
        ids[i:i + BATCH_SIZE]
        for i in range(0, len(ids), BATCH_SIZE)
    ]

    # Match the original training drop_last policy.
    if training and len(batches[-1]) < BATCH_SIZE:
        batches = batches[:-1]

    for step, drug_ids in enumerate(batches, 1):
        batch = Batch.from_data_list([
            graph_cache[int(i)].clone() for i in drug_ids
        ])

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(batch)
            loss = calculate_loss(outputs, epoch, training)

            if training:
                loss.backward()
                assert all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ), "Nonfinite gradients"
                optimizer.step()

        total += loss.item() * len(drug_ids)
        count += len(drug_ids)

        if step == 1 or step % 10 == 0 or step == len(batches):
            phase = "Train" if training else "Validation"
            print(
                f"Epoch {epoch:02d} | {phase} "
                f"{step}/{len(batches)} | loss {loss.item():.5f}",
                flush=True,
            )

        del batch, outputs, loss

    return total / count

for epoch in range(start_epoch, EPOCHS + 1):
    started = time.perf_counter()

    # Original adaptive weights use preceding epoch loss histories.
    if epoch > 2:
        for amc in (low_amc, high_amc):
            ratio = amc.last_loss[epoch - 1] / amc.last_loss[epoch - 2]
            assert torch.isfinite(ratio).all()
            assert torch.isfinite(torch.exp(ratio)).all(), (
                "Adaptive weights overflow; stop for review."
            )

    order = np.random.permutation(train_ids).tolist()
    train_loss = run_pass(order, epoch, training=True)

    # Fixed validation randomness; do not alter training RNG state.
    training_rng = get_rng()
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    val_loss = run_pass(val_ids, epoch, training=False)
    set_rng(training_rng)

    improved = val_loss < best_loss
    if improved:
        best_loss = val_loss

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "seconds": time.perf_counter() - started,
        "best": improved,
    })

    checkpoint = {
        "identity": identity,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "low_history": low_amc.last_loss.clone(),
        "high_history": high_amc.last_loss.clone(),
        "best_loss": best_loss,
        "history": history,
        "rng": get_rng(),
    }

    if improved:
        atomic_save(checkpoint, best_path)
    atomic_save(checkpoint, last_path)

    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv", index=False
    )

    print(
        f" Epoch {epoch} saved | "
        f"Train={train_loss:.5f} | Val={val_loss:.5f} | "
        f"Best={best_loss:.5f}",
        flush=True,
    )

print("\n SEED 42 PRETRAINING COMPLETED", flush=True)
print("Best validation loss:", best_loss, flush=True)
print("Saved in:", run_dir, flush=True)
print("No interaction classification or test evaluation performed.",
      flush=True)

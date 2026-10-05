from pathlib import Path
import os
import json
import hashlib
import time
import math
import gc
import importlib.metadata as md
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
BASE = Path("/content/drive/MyDrive/DDI_V2")
FEATURES = BASE / "PRIMARY_TOPOLOGY_TRAIN_ONLY_V2"
REFERENCE = FEATURES / "training_current_node_bn_v2/original/seed_42"
RUN = FEATURES / "training_current_node_bn_v2/self_loops_only/seed_42"
GRAPH = FEATURES / "topology_graphs_v1/self_loops_only.npz"
SEED = 42
EPOCHS = 120
BATCH_SIZE = 512
SAVE_EVERY = 50
assert torch.cuda.is_available(), "A CUDA GPU is required."
assert BASE.exists(), "Mount Google Drive first."
DEVICE = torch.device("cuda")
RUN.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)
def save_pt(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
def save_history(history):
    tmp = RUN / "training_history.csv.tmp"
    pd.DataFrame(history).to_csv(tmp, index=False)
    os.replace(tmp, RUN / "training_history.csv")
class CurrentNodeBNGCN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2250, 384, 384, 192]
        self.convs = nn.ModuleList([
            GCNConv(
                dims[i],
                dims[i + 1],
                improved=False,
                cached=False,
                add_self_loops=False,
                normalize=True,
                bias=True
            )
            for i in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(d, track_running_stats=False)
            for d in [384, 384, 192]
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 84)
        )
    def encode(self, x, edges):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, edges))
            if i < 2:
                x = F.dropout(
                    F.relu(x),
                    p=0.35,
                    training=self.training
                )
        return x
    def classify(self, z, pairs):
        a = z[pairs[:, 0]]
        b = z[pairs[:, 1]]
        pair_features = torch.cat(
            [a, b, (a - b).abs(), a * b],
            dim=1
        )
        return self.head(pair_features)
    def forward(self, x, edges, pairs):
        return self.classify(self.encode(x, edges), pairs)
def main():
    reference_path = REFERENCE / "config.json"
    cache = FEATURES / "train_only_feature_cache.npz"
    for path in [reference_path, cache, GRAPH]:
        assert path.exists(), f"Missing file: {path}"
    reference = json.loads(reference_path.read_text())
    assert reference["condition"] == "original"
    assert reference["seed"] == SEED
    assert digest(cache) == reference["feature_sha256"]
    versions = {
        package: md.version(package)
        for package in [
            "torch",
            "torch-geometric",
            "numpy",
            "pandas",
            "scikit-learn"
        ]
    }
    original_v2_path = (
        FEATURES / "training_current_node_bn_v2/original/seed_42/config.json"
    )
    assert original_v2_path.exists(), "Corrected Original 42 config missing."
    original_v2 = json.loads(original_v2_path.read_text())
    assert original_v2["protocol"] == "current_node_bn_v2"
    assert original_v2["condition"] == "original"
    assert original_v2["seed"] == SEED
    assert original_v2["feature_sha256"] == reference["feature_sha256"]
    assert original_v2["split_sha256"] == reference["split_sha256"]
    differences = {
        package: [original_v2["versions"].get(package), version]
        for package, version in versions.items()
        if original_v2["versions"].get(package) != version
    }
    assert not differences, (
        f"Environment differs from corrected Original 42: {differences}"
    )
    print("Environment matches corrected Original 42 ")
    with np.load(cache, allow_pickle=False) as loaded:
        x = torch.tensor(
            loaded["x"],
            dtype=torch.float32,
            device=DEVICE
        )
        drugs = loaded["all_drugs"].tolist()
    assert x.shape == (1705, 2250), f"Unexpected shape: {x.shape}"
    assert torch.isfinite(x).all()
    mapping = {drug: i for i, drug in enumerate(drugs)}
    assert len(mapping) == len(drugs), "Duplicate drug IDs in cache."
    data = {}
    # Only Train and Calibration are read. Test is not read.
    for split in ["train", "calibration"]:
        path = BASE / f"clean_{split}_split.csv"
        assert path.exists(), f"Missing file: {path}"
        assert digest(path) == reference["split_sha256"][split]
        frame = pd.read_csv(path)
        a = frame["Drug1"].map(mapping)
        b = frame["Drug2"].map(mapping)
        assert a.notna().all() and b.notna().all(), (
            f"Unmapped drug IDs in {split}."
        )
        labels = frame["label_id"].to_numpy(dtype=np.int64)
        assert set(labels) == set(range(84)), (
            f"Unexpected class coverage in {split}."
        )
        data[split] = {
            "pairs": torch.tensor(
                np.column_stack([a, b]).astype(np.int64),
                dtype=torch.long,
                device=DEVICE
            ),
            "y": torch.tensor(
                labels,
                dtype=torch.long,
                device=DEVICE
            ),
            "truth": labels
        }
    with np.load(GRAPH, allow_pickle=False) as loaded:
        edges = torch.tensor(
            loaded["edge_index"],
            dtype=torch.long,
            device=DEVICE
        )
    assert edges.ndim == 2 and edges.shape[0] == 2
    assert edges.numel() > 0
    assert int(edges.min()) >= 0
    assert int(edges.max()) < len(drugs)
    # Exactly one self-loop per node, with no inter-node edges.
    assert edges.shape[1] == len(drugs), "Expected one edge per node."
    assert torch.equal(edges[0], edges[1]), "Found non-self-loop edges."
    expected_nodes = torch.arange(len(drugs), device=DEVICE)
    assert torch.equal(torch.sort(edges[0]).values, expected_nodes), (
        "Self-loops must cover every node exactly once."
    )
    print(f"Graph verified: {len(drugs)} self-loops, no neighbor edges ")
    counts = np.bincount(
        data["train"]["truth"],
        minlength=84
    ).astype(float)
    weights = np.clip(
        np.sqrt(counts.max() / (counts + 1)),
        0.5,
        3.0
    )
    weights /= weights.mean()
    np.testing.assert_allclose(
        weights,
        reference["class_weights"],
        rtol=0,
        atol=1e-12
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Fresh initialization; no Original or Rewired weights loaded.
    model = CurrentNodeBNGCN().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0007,
        weight_decay=0.0001
    )
    assert sum(p.numel() for p in model.parameters()) == 1636244
    assert all(
        not module.track_running_stats
        for module in model.modules()
        if isinstance(module, nn.BatchNorm1d)
    )
    config = {
        "protocol": "current_node_bn_v2",
        "condition": "self_loops_only",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": 0.0007,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "early_stopping": None,
        "scheduler": None,
        "architecture": str(model),
        "class_weights": weights.tolist(),
        "primary_selection": "calibration_macro_f1_all84",
        "tie_rule": "earliest epoch",
        "BN": "current full-node statistics in train and eval",
        "BN_track_running_stats": False,
        "BN_node_scope": "all 1705 fixed nodes; transductive",
        "descriptor_preprocessing_fit": "Train nodes only",
        "graph_source": "Train edges only",
        "shuffle_seed": "seed + 100000 * epoch",
        "feature_sha256": digest(cache),
        "graph_sha256": digest(GRAPH),
        "split_sha256": reference["split_sha256"],
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0),
        "test_evaluated": False,
        "bitwise_GPU_resume_guaranteed": False
    }
    # All model and training settings must match corrected Original 42.
    for key in [
        "protocol", "seed", "epochs", "batch_size", "optimizer",
        "learning_rate", "weight_decay", "gradient_clip_norm",
        "early_stopping", "scheduler", "architecture", "class_weights",
        "primary_selection", "tie_rule", "BN", "BN_track_running_stats",
        "BN_node_scope", "descriptor_preprocessing_fit", "shuffle_seed"
    ]:
        assert config[key] == original_v2[key], f"Original mismatch: {key}"
    config_path = RUN / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, (
            "Saved configuration differs; do not overwrite."
        )
    else:
        save_json(config_path, config)
    config_hash = digest(config_path)
    done = RUN / "TRAINING_COMPLETE.json"
    if done.exists():
        completed = json.loads(done.read_text())
        assert completed["config_sha256"] == config_hash
        print("Run already completed.")
        print(json.dumps(completed, indent=2))
        print("\nSaved:", RUN)
        return
    last = RUN / "last_training_state.pt"
    epoch = 1
    next_batch = 0
    loss_sum = 0.0
    loss_batches = 0
    history = []
    best = {
        metric: -1.0
        for metric in ["macro_f1", "accuracy", "weighted_f1"]
    }
    best_epochs = {metric: None for metric in best}
    elapsed_before = 0.0
    if last.exists():
        saved = torch.load(
            last,
            map_location="cpu",
            weights_only=True
        )
        assert saved["config_sha256"] == config_hash
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch = saved["epoch"]
        next_batch = saved["next_batch"]
        loss_sum = saved["loss_sum"]
        loss_batches = saved["loss_batches"]
        history = saved["history"]
        best = saved["best"]
        best_epochs = saved["best_epochs"]
        elapsed_before = saved["elapsed_seconds"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved
        print(
            f"RESUME: Self-loops only 42 — epoch {epoch}, "
            f"completed batches {next_batch}"
        )
    else:
        print("START: Self-loops only 42 — corrected BN — FROM SCRATCH")
    started = time.perf_counter()
    def checkpoint():
        save_pt(last, {
            "config_sha256": config_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "loss_sum": loss_sum,
            "loss_batches": loss_batches,
            "history": history,
            "best": best,
            "best_epochs": best_epochs,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "elapsed_seconds": (
                elapsed_before + time.perf_counter() - started
            )
        })
    @torch.no_grad()
    def calibration_metrics():
        # Dropout off; BN uses current full-node statistics.
        model.eval()
        z = model.encode(x, edges)
        assert torch.isfinite(z).all()
        predictions = []
        pairs = data["calibration"]["pairs"]
        for start in range(0, len(pairs), 2048):
            logits = model.classify(z, pairs[start:start + 2048])
            assert torch.isfinite(logits).all()
            predictions.append(logits.argmax(1).cpu().numpy())
        pred = np.concatenate(predictions)
        truth = data["calibration"]["truth"]
        return {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(
                truth,
                pred,
                labels=np.arange(84),
                average="macro",
                zero_division=0
            )),
            "weighted_f1": float(f1_score(
                truth,
                pred,
                labels=np.arange(84),
                average="weighted",
                zero_division=0
            ))
        }
    n = len(data["train"]["pairs"])
    n_batches = math.ceil(n / BATCH_SIZE)
    progress = tqdm(
        total=EPOCHS,
        initial=min(epoch - 1, EPOCHS),
        desc="Self-loops only 42 — current-node BN"
    )
    while epoch <= EPOCHS:
        epoch_start = time.perf_counter()
        resumed = next_batch > 0
        generator = torch.Generator().manual_seed(
            SEED + 100000 * epoch
        )
        order = torch.randperm(
            n,
            generator=generator
        ).to(DEVICE)
        model.train()
        batches = tqdm(
            total=n_batches,
            initial=next_batch,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False
        )
        for batch in range(next_batch, n_batches):
            idx = order[
                batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE
            ]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                x,
                edges,
                data["train"]["pairs"][idx]
            )
            loss = loss_fn(logits, data["train"]["y"][idx])
            assert torch.isfinite(loss), "Nonfinite loss."
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )
            assert torch.isfinite(grad_norm), "Nonfinite gradients."
            optimizer.step()
            loss_sum += float(loss.detach())
            loss_batches += 1
            next_batch = batch + 1
            batches.update(1)
            if next_batch % 10 == 0:
                batches.set_postfix(
                    loss=f"{loss_sum / loss_batches:.4f}"
                )
            if (
                next_batch % SAVE_EVERY == 0
                or next_batch == n_batches
            ):
                checkpoint()
        batches.close()
        metrics = calibration_metrics()
        for metric, value in metrics.items():
            if value > best[metric]:
                best[metric] = value
                best_epochs[metric] = epoch
                save_pt(RUN / f"best_by_{metric}.pt", {
                    "config_sha256": config_hash,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "criterion": metric,
                    "calibration_metrics": metrics,
                    "test_evaluated": False
                })
        row = {
            "epoch": epoch,
            "mean_batch_train_loss": loss_sum / loss_batches,
            "cal_accuracy": metrics["accuracy"],
            "cal_macro_f1": metrics["macro_f1"],
            "cal_weighted_f1": metrics["weighted_f1"],
            "epoch_segment_seconds": (
                time.perf_counter() - epoch_start
            ),
            "resumed_partway": resumed
        }
        history.append(row)
        epoch += 1
        next_batch = 0
        loss_sum = 0.0
        loss_batches = 0
        checkpoint()
        save_history(history)
        progress.update(1)
        progress.set_postfix(
            best_macro=f"{100 * best['macro_f1']:.2f}%"
        )
        tqdm.write(
            f"Epoch {row['epoch']:03d}/{EPOCHS} | "
            f"loss={row['mean_batch_train_loss']:.4f} | "
            f"Cal Acc={100 * metrics['accuracy']:.2f}% | "
            f"Macro-F1={100 * metrics['macro_f1']:.2f}% | "
            f"Weighted-F1={100 * metrics['weighted_f1']:.2f}% | saved"
        )
    progress.close()
    save_history(history)
    result = {
        "protocol": "current_node_bn_v2",
        "condition": "self_loops_only",
        "seed": SEED,
        "completed_epochs": EPOCHS,
        "best_calibration_scores": best,
        "best_epochs": best_epochs,
        "primary_checkpoint": "best_by_macro_f1.pt",
        "elapsed_hours": (
            elapsed_before + time.perf_counter() - started
        ) / 3600,
        "test_evaluated": False,
        "config_sha256": config_hash
    }
    save_json(done, result)
    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(result, indent=2))
    print("\nSaved:", RUN)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
main()

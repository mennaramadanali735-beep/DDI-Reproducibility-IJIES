
import os
import sys
from pathlib import Path

print("Loading libraries...", flush=True)
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch

source = Path(sys.argv[1])
features = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
sys.path.insert(0, str(source))

# Load definitions only; do not run the classification training script.
script_path = source / "ddi_dnn-z.py"
code = script_path.read_text(encoding="utf-8")
code = code.replace(
    'os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"', ''
)
code = code.replace(
    "os.environ['CUDA_LAUNCH_BLOCKING'] = '1'", ''
)

namespace = {
    "__name__": "mhafr_embedding_extraction",
    "__file__": str(script_path),
}
exec(compile(code, str(script_path), "exec"), namespace)

def fast_shortest_paths(adjacency):
    distances = shortest_path(
        csr_matrix(adjacency),
        directed=True,
        unweighted=True,
    )
    # Match the original unreachable-distance sentinel.
    return np.minimum(distances, 510).astype(np.float64)

example = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
], dtype=bool)

np.testing.assert_array_equal(
    namespace["floyd_warshall"](example),
    fast_shortest_paths(example),
)
namespace["floyd_warshall"] = fast_shortest_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
namespace["device"] = device

checkpoint = torch.load(
    run_dir / "best_checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)

encoder = namespace["CL_model_2d"]().to(device)

# Require an exact match; no silently skipped weights.
encoder.load_state_dict(checkpoint["model"], strict=True)
encoder.eval()

print(
    f" Encoder loaded strictly from epoch {checkpoint['epoch']}",
    flush=True,
)

manifest = pd.read_csv(features / "drug_input_manifest.csv")
manifest = manifest.sort_values("drug_id").reset_index(drop=True)
assert manifest["drug_id"].tolist() == list(range(1660))

chunks = []
BATCH_SIZE = 4

with torch.no_grad():
    for start in range(0, len(manifest), BATCH_SIZE):
        rows = manifest.iloc[start:start + BATCH_SIZE]
        graphs = []

        for row in rows.itertuples(index=False):
            path = features / "molecules" / f"{row.drug_id:05d}.npz"

            with np.load(path, allow_pickle=False) as saved:
                assert saved["drug_id"].item() == row.drug_id
                assert saved["smiles"].item() == row.SMILES

                graph = Data(
                    x=torch.from_numpy(saved["x"].copy()).long(),
                    edge_index=torch.from_numpy(
                        saved["edge_index"].copy()
                    ).long(),
                    edge_attr=torch.from_numpy(
                        saved["edge_attr"].copy()
                    ).long(),
                )
                graph.smiles = saved["token_ids"].copy()
                graph.mask = saved["token_mask"].copy()
                graphs.append(graph)

        batch = Batch.from_data_list(graphs).to(device)
        low_2d, high_2d, low_1d, high_1d = encoder(batch)

        # Same averaging and concatenation as the official classifier.
        embeddings = torch.cat(
            (
                (low_1d + high_1d) / 2,
                (low_2d + high_2d) / 2,
            ),
            dim=1,
        )

        assert embeddings.shape == (len(rows), 256)
        assert torch.isfinite(embeddings).all()
        chunks.append(embeddings.cpu().numpy())

        completed = start + len(rows)
        if start == 0 or completed % 100 == 0 or completed == len(manifest):
            print(f"Encoded {completed}/{len(manifest)} drugs", flush=True)

        del batch, embeddings

all_embeddings = np.concatenate(chunks, axis=0)
assert all_embeddings.shape == (1660, 256)
assert np.isfinite(all_embeddings).all()

output_path = run_dir / "molecular_embeddings_1d2d.npz"
temporary = output_path.with_suffix(".tmp")

with temporary.open("wb") as handle:
    np.savez_compressed(
        handle,
        x=all_embeddings,
        drug_ids=manifest["drug_id"].to_numpy(dtype=np.int64),
        smiles=manifest["SMILES"].to_numpy(dtype=str),
        checkpoint_epoch=np.asarray(checkpoint["epoch"]),
    )

with np.load(temporary, allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved["x"], all_embeddings)
    np.testing.assert_array_equal(
        saved["drug_ids"], manifest["drug_id"].to_numpy()
    )

temporary.replace(output_path)

print("\nEMBEDDING EXTRACTION COMPLETED", flush=True)
print("Shape:", all_embeddings.shape, flush=True)
print("Maximum absolute value:", np.abs(all_embeddings).max(), flush=True)
print("Saved:", output_path, flush=True)
print(" All drugs retained; saved embeddings verified.", flush=True)
print(" No training or test-label evaluation performed.", flush=True)


import sys
import json
import hashlib
from pathlib import Path

print("Loading libraries...", flush=True)
import numpy as np
import pandas as pd
from rdkit import Chem
from subword_nmt.apply_bpe import BPE

source, protocol, coordinates_root, output = map(Path, sys.argv[1:5])
sys.path.insert(0, str(source))

from process_dataset.MPP.utils.features import (
    atom_to_feature_vector,
    bond_to_feature_vector,
)

print("Reading saved splits and 3D records...", flush=True)

filenames = {
    "train": "cold_train.csv",
    "calibration": "cold_calibration.csv",
    "test": "cold_test_both_unseen.csv",
}
expected_rows = {"train": 78736, "calibration": 13894, "test": 17896}

splits = {
    name: pd.read_csv(protocol / filename, keep_default_na=False)
    for name, filename in filenames.items()
}
for name, df in splits.items():
    assert len(df) == expected_rows[name]
    assert df["label_id"].between(0, 83).all()

drugs_by_split = {
    name: set(df["Drug1"]) | set(df["Drug2"])
    for name, df in splits.items()
}
assert not (
    (drugs_by_split["train"] | drugs_by_split["calibration"])
    & drugs_by_split["test"]
)

all_drugs = sorted(set.union(*drugs_by_split.values()))
drug_to_id = {s: i for i, s in enumerate(all_drugs)}
assert len(all_drugs) == 1660

status = pd.read_csv(coordinates_root / "final_3d_status.csv")
accepted = set(status.loc[status["status"].eq("accepted"), "SMILES"])
assert len(accepted) == 1080
assert accepted <= drugs_by_split["train"]

codes_path = source / "ESPF/drug_codes_chembl_freq_1500.txt"
mapping_path = source / "ESPF/subword_units_map_chembl_freq_1500.csv"

tokens = pd.read_csv(mapping_path)["index"].tolist()
assert len(tokens) == len(set(tokens)) == 2586
token_to_id = {token: i for i, token in enumerate(tokens)}

with codes_path.open(encoding="utf-8") as handle:
    bpe = BPE(handle, merges=-1, separator="")

molecule_dir = output / "molecules"
molecule_dir.mkdir(exist_ok=True)
records = []

print("Building molecular inputs...", flush=True)

for drug_id, smiles in enumerate(all_drugs):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None and mol.GetNumAtoms() > 0

    x = np.asarray(
        [atom_to_feature_vector(a) for a in mol.GetAtoms()],
        dtype=np.int64,
    )

    edges, attributes = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_to_feature_vector(bond)
        edges.extend([(a, b), (b, a)])
        attributes.extend([feature, feature])

    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    edge_attr = np.asarray(attributes, dtype=np.int64).reshape(-1, 3)

    words = bpe.process_line(smiles).split()
    unknown = [word for word in words if word not in token_to_id]
    fallback = bool(unknown or not words)

    # Preserve the official whole-sequence fallback and record it.
    ids = [0] if fallback else [token_to_id[word] for word in words]
    length = min(len(ids), 50)
    token_ids = np.zeros(50, dtype=np.int64)
    token_mask = np.zeros(50, dtype=np.int64)
    token_ids[:length] = ids[:length]
    token_mask[:length] = 1

    arrays = {
        "drug_id": np.asarray(drug_id, dtype=np.int64),
        "smiles": np.asarray(smiles),
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "token_ids": token_ids,
        "token_mask": token_mask,
        "has_3d": np.asarray(smiles in accepted),
    }

    if smiles in accepted:
        key = hashlib.sha256(smiles.encode()).hexdigest()
        path = coordinates_root / "coordinates" / f"{key}.npz"

        with np.load(path, allow_pickle=False) as saved:
            assert saved["smiles"].item() == smiles
            np.testing.assert_array_equal(
                saved["atomic_numbers"],
                [a.GetAtomicNum() for a in mol.GetAtoms()],
            )
            positions = saved["positions"].copy()

        assert positions.shape == (len(x), 3)
        assert np.isfinite(positions).all()
        arrays["positions"] = positions

    path = molecule_dir / f"{drug_id:05d}.npz"
    np.savez_compressed(path, **arrays)

    with np.load(path, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)

    records.append({
        "drug_id": drug_id,
        "SMILES": smiles,
        "is_train": smiles in drugs_by_split["train"],
        "atoms": len(x),
        "bonds": mol.GetNumBonds(),
        "raw_token_count": len(words),
        "tokenizer_fallback": fallback,
        "unknown_token_count": len(unknown),
        "tokens_truncated": not fallback and len(words) > 50,
        "has_3d": smiles in accepted,
    })

    if (drug_id + 1) % 100 == 0 or drug_id + 1 == len(all_drugs):
        print(f"Saved {drug_id + 1}/{len(all_drugs)} drugs", flush=True)

audit = pd.DataFrame(records)
audit.to_csv(output / "drug_input_manifest.csv", index=False)

for name, df in splits.items():
    pairs = pd.DataFrame({
        "source_row": np.arange(len(df)),
        "drug1_id": df["Drug1"].map(drug_to_id).to_numpy(),
        "drug2_id": df["Drug2"].map(drug_to_id).to_numpy(),
        "label_id": df["label_id"].to_numpy(),
    })
    assert not pairs.isna().any().any()
    pairs.to_csv(output / f"{name}_pairs.csv", index=False)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    "official_commit": "92964ff7696ca6fdba5c8496fbc9a3212cbb5eb1",
    "drugs": len(all_drugs),
    "train_3d_drugs": len(accepted),
    "classes": 84,
    "max_1d_tokens": 50,
    "graph_atoms_truncated": False,
    "atom_order": "original_SMILES_parse_order",
    "vocabulary_policy": "fixed_official_external_vocabulary",
    "vocabulary_sha256": sha256(codes_path),
    "token_mapping_sha256": sha256(mapping_path),
    "source_split_sha256": {
        name: sha256(protocol / filename)
        for name, filename in filenames.items()
    },
    "training_performed": False,
}
(output / "input_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nMOLECULAR INPUTS COMPLETED", flush=True)
print("Drugs:", len(audit), flush=True)
print("Train drugs with 3D:", int(audit["has_3d"].sum()), flush=True)
print("Tokenizer fallbacks:", int(audit["tokenizer_fallback"].sum()), flush=True)
print("Truncated sequences:", int(audit["tokens_truncated"].sum()), flush=True)
print(" Saved arrays verified; all classification rows retained.", flush=True)

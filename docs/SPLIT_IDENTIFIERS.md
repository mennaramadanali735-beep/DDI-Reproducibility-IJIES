# Ordered identifiers and matching

Primary interaction splits contain 134,249 Train, 28,768 Calibration and 28,768 Test records. `scripts/build_split_manifests.py` checks the entire source CSV SHA-256 before writing identifiers. A record identifier is SHA-256 of UTF-8 compact JSON `[Drug1,Drug2,Label,integer label_id]`, preserving exact strings and direction. `row_index` is zero-based saved order. `occurrence_index` is the zero-based occurrence of that exact content hash within a split.

These identifiers do not shuffle, sort or resplit the data. They let licensed users verify exact files and rows without distributing SMILES. Hashes are identifiers, not a guarantee of anonymity. Distinct source rows with identical content require occurrence handling. Primary exact directed keys are unique in this dataset; the general matching implementation still checks uniqueness explicitly.

Side-effect Calibration/Test manifests use the same serialization and class mapping but preserve their independent task-specific order. Candidate counts come from the original Morgan-identity audit and are bound by package hashes. Only candidate_count == 1 enters the common held-out analysis. Matching both Test memberships produces 4,390 records. Same-sized task splits are not interchangeable.

Side-effect Train identifiers use sorted CSR feature-column indices and values, preserving all 134,249 saved input rows. Their source sparse-matrix hash and the archived molecular row-file hash are recorded separately. Feature collisions mean these are not proven unique molecular-record identifiers; the separate `train_ids.csv` now supplies the recovered molecular-record identifiers, verified against the original row-file hash.

The 86-to-84 audit preserves original labels and raw/final counts. It establishes 191,808 to 191,795 records, with 7 and 6 records absent for two labels. It does not establish why those records were excluded, nor a merged-class transformation. Ten further molecular-invalid rows lead to the 191,785-record primary experiment. Do not label the 13-row exclusion as an invalid-SMILES decision without separate evidence.

## Recovered molecular Train and scaffold identities

`side_effects/train_ids.csv` now binds the original 134,249 rows to exact molecular-record hashes and occurrence indices. The older sparse-feature identifiers remain available as a distinct feature-level audit.

`scaffold_nonchiral/` contains ordered identifiers, source-row membership, original protocol metadata, class support, and feature metadata for the corrected September 21 protocol. Raw molecule strings are excluded. The recovered preprocessing metadata belongs to scaffold experiments, not the side-effect LightGBM environment.

Regenerate these identifiers from original private inputs with `python scripts/recover_split_manifests.py --side-train PATH_TO_TRAIN_ROWS --scaffold PATH_TO_EXTRACTED_SCAFFOLD`. This verifies archived bytes and saved scaffold columns; it does not rerun RDKit scaffold generation, training, or checkpoint inference.

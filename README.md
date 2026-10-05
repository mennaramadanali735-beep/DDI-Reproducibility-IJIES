# DDI reproducibility package

This candidate accompanies **A Modular Pipeline for Drug–Drug Interaction Classification and Multi-Label Side-Effect Prediction with Same-Record Evaluation**. It targets the revised manuscript dated 3 October 2026, including the 4,390-record integrated evaluation. It does not reproduce the superseded 28,747-record integrated analysis.

## Reproduce saved-prediction results

From this directory, in a dedicated Python environment:

```bash
python -m pip install -r environments/evaluation-requirements.txt
python scripts/reproduce_tables_8_to_13.py
python scripts/select_side_effect_thresholds.py
python scripts/validate_package.py
python scripts/verify_release.py
```

The first command regenerates Tables 8, 10, 11 and 12 from actual prediction outputs and derives the evaluation-scope Table 13. Decimal metrics in CSV files are fractions; multiply by 100 for percentages. Sample SD uses `ddof=1`. Table 12 contains counts and percentages. Table 9 is complete in `results/table9.csv`: all ten settings use archived per-seed metrics (seeds 42/43/44), including feature and checkpoint-selection studies. These aggregates do not represent fresh inference. Other release limitations remain in `docs/REMAINING_ITEMS.md`.

Threshold reselection reads **Calibration only**, reproduces all 263 saved thresholds exactly, selects per-label thresholds, and recovers global threshold 0.54. Frozen probabilities, numeric targets, class order and threshold artifacts are bound through `SHA256SUMS.txt` and the threshold protocol. These operations evaluate saved predictions; they do not rerun neural-network inference or training.

## Contents

- `splits/`: ordered primary interaction identifiers and side-effect input identifiers.
- `mappings/`: exact 84-class mapping, final 263-column order, class-count audit and training-derived numeric lookup.
- `configs/`: original run configurations, source hashes and frozen threshold selection.
- `artifacts/`: numeric ground truth, predictions and frozen probabilities. No molecular strings or raw DrugBank rows.
- `scripts/`: CPU evaluation, integrated matching, threshold selection, identity checks and recovered training launchers.
- `notebooks/`: selected experimental workflows with cleared outputs.
- `third_party/MHAFR-DDI/`: verified upstream source and vocabulary, with original MIT license.
- `results/`: outputs produced during this preparation, including explicit completeness status.

## Training and inference

Follow `docs/WORKFLOW_ORDER.md`. Training families use distinct environments and input files. Original Colab/Kaggle paths remain in recovered workflows; restore that layout or deliberately adjust the root variables. Never bypass hash, class-order or Calibration checks. The scientific protocol and data requirements are documented in `docs/SPLIT_IDENTIFIERS.md` and `docs/UPSTREAM_AND_ENVIRONMENTS.md`.

## Publication status

This is a prepared upload folder, not an already published repository. The package contains no invented repository URL or DOI. After creating a repository and immutable release, add its actual URL and revision to the manuscript and response letter. See `docs/PUBLIC_RELEASE_SELECTION.md` and `docs/REMAINING_ITEMS.md` before stating that Comment 8 is fully resolved.

Recovered split evidence: `splits/side_effects/train_ids.csv` contains verified ordered Train molecular-record hashes; `splits/scaffold_nonchiral/` contains corrected split manifests and protocol/feature metadata. Archived six-run scaffold summaries are under `artifacts/scaffold/`; validation scope is recorded in `docs/recovered_split_validation.json`.

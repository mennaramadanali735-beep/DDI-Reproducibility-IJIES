# Outstanding items and verified completion

## Completed in this candidate

- Exact primary interaction split SHA-256 values and ordered identifiers for 134,249 Train, 28,768 Calibration and 28,768 Test rows.
- Standalone 84-class mapping checked against the saved primary splits.
- Standalone final 263-label order; the older similarly named mapping differs at 49 positions and is excluded from the public folder.
- All 263 thresholds, global threshold 0.54, selected method and label-order hash.
- Fresh Calibration-only threshold reselection: every threshold and both objective values match the archived artifacts exactly.
- Saved side-effect Test predictions reconstructed exactly from probabilities and thresholds; final Y_test matched the independently recovered target strings before raw strings were excluded.
- Tables 8, 10, 11 and 12 recomputed from prediction outputs; Table 13 generated from these results and protocol scope.
- Exact-identity common-test matching reproduces 4,390 records; ambiguous Morgan identities are excluded. Four outcome counts reproduce 3,216 / 56 / 905 / 213.
- Original training source for all nine topology runs, historical weight/pair ablations, scaffold workflows, feature/checkpoint studies, AutoDDI final training/evaluation for seeds 42–44, and eight MHAFR adaptation scripts.
- MHAFR official commit, both vocabulary hashes and source files verified. AutoDDI official commit and all 15 source-file hashes verified against all three final configs.
- 86-to-84 class counts included. Cause of the 13 excluded rows remains unknown, not invented.
- Table 9: all ten settings aggregated from original per-seed metric files, with sample SD (ddof=1). Feature inputs and checkpoint-selection outputs are packaged; no rounded manuscript values were substituted and no fresh inference was performed.

- Side-effect Train molecular-record manifest recovered: all 134,249 ordered rows verified against the historical raw-file SHA-256.
- Corrected scaffold split identities recovered for Train 91,477 / Calibration 16,151 / Test 11,142; all saved CSV and feature hashes verified, feature pair order matched, and saved nonchiral scaffold columns have zero Test/development overlap. RDKit scaffolds were not recomputed here.
- Scaffold six-run summary means and sample SD reproduced from archived metrics. The summary retains prediction/checkpoint hashes; prediction bytes and fresh neural inference were not verified in this recovery.

## Not complete; do not claim closure

1. **Public URL/release:** no public repository has been created by this preparation. Upload and publish an immutable release, then cite the real URL/revision.
2. **Side-effect historical environment:** Python and LightGBM are recorded in training configs. The complete direct-dependency file is supplied, but unrecorded historical versions remain unpinned. This is not a verified historical lockfile. Freeze and validate a compatible training environment before making that claim.
3. **Fresh clean-machine training/checkpoint inference:** not performed. This runtime lacks the neural training stack/GPU. Completed validation consists of real CPU prediction evaluation, exact matching, threshold reselection, source/identity checks and syntax checks. It is not a new end-to-end training reproduction.
4. **Licensing:** computational artifacts are separated from V8 clinical/MedDRA material. No project-wide redistribution license has been chosen for the authors. Upstream licenses are preserved; absence of raw rows is not a claim that all derived artifacts have been legally cleared.

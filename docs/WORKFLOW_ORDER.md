# Execution order

| Family | Ordered workflow | Inputs / constraints |
|---|---|---|
| Saved Tables 8–13 | `scripts/reproduce_tables_8_to_13.py` | Packaged numeric artifacts; Table 9 has an explicit completeness gate |
| Threshold selection | `scripts/select_side_effect_thresholds.py` | Packaged Calibration probabilities and targets; never Test selection |
| Integrated inference and alignment | `integrated_alignment_audit.ipynb`, then CPU table script | Original historical concat seed-44 checkpoint, split CSVs, side X/Y and feature cache; recovery cells include alternative forensic attempts and require their stated inputs; use final 2F/2G/2H protocol |
| Primary topology | `topology_train_only_preprocessing.ipynb`; condition-specific training; `topology_final_evaluation.ipynb` | V2 corrected 2,250-dimensional cache and three graph conditions; all original split hashes |
| Historical weighted GCN | `historical_weighted_three_seeds.ipynb` | Original 2,249-dimensional historical cache |
| Historical weight/pair study | `historical_weight_pair_ablation.ipynb` | Run setup and declarations first, then the three unweighted and three concat runs; weighted baseline is separate |
| Feature study | `feature_ablation_preparation.ipynb`; `feature_ablation_training.ipynb` | Run all training cells in one IPython session: later cells recover the seed-42 source from session history |
| Checkpoint study | `model_selection_training.ipynb`; where needed `model_selection_cache_recovery.ipynb`; `model_selection_evaluation.ipynb` | Correct historical 2,249-dimensional train-only cache; three checkpoints from each trajectory |
| Corrected scaffold | `scaffold_nonchiral_preparation.ipynb`; `scaffold_training_three_seeds.ipynb`; `scaffold_results.ipynb` | Nonchiral Murcko scaffolds, training-only preprocessing; update downstream PROTOCOL root to the preparation output; preserve split-generation parameters; run trainer cells in one session |
| Side effects | `side_effect_unique263_targets.ipynb`; `side_effect_unique263_training.ipynb` per seed; `side_effect_primary_analysis.ipynb` | Exact final vocabulary and saved Morgan X matrices. Hybrid/weighted-search notebooks are alternative studies, not the primary equal ensemble |
| AutoDDI | Verified upstream, original prepared molecular inputs, `scripts/run_autoddi.py` per seed; recovered `evaluate_seedXX.py` | Fixed candidate 8 final training from scratch. Search candidates are separate. Evaluation scripts take INPUTS SOURCE RUN_ROOT and require completed checkpoints |
| MHAFR | `build_molecular_inputs.py`; `run_mhafr.py pretrain`; `run_mhafr.py embeddings`; `run_mhafr.py classify` per seed | Original molecular inputs, coordinates, split file and verified upstream; place archived molecular_pretrain_split.csv under RUN_ROOT |

Notebook filenames above are under `notebooks/`. Preparations that create timestamped directories must pass their actual output directory to downstream root variables. Recovered research workflows are not portable one-command GPU jobs; the package does not assert otherwise.

The integrated CPU path matches exact directed record hashes plus occurrence indices, checks one-to-one correspondence and excludes every candidate_count != 1. It independently recomputes side-vector correctness. The training-derived class lookup is indexed by predicted class, not true Test class; true classes are used only for a separate annotation-consistency assertion.

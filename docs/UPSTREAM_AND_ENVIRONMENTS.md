# Upstream source and environment provenance

## MHAFR

Repository: https://github.com/lml-mengli/MHAFR-DDI

Exact recorded commit: `92964ff7696ca6fdba5c8496fbc9a3212cbb5eb1`. It agrees with both `input_metadata.json` and `pretraining_protocol.json`. The distributed subset contains all root Python modules, `method/`, `process_dataset/`, the two ESPF vocabulary files, upstream environment record and original MIT license. Dataset directories, weights and bytecode were excluded. Source bytes were not cosmetically rewritten.

```bash
python scripts/verify_upstream.py mhafr third_party/MHAFR-DDI
```

Recovered adaptation pretraining uses 30 epochs, batch size 10 and Adam with learning rate 1e-4 and weight decay 1e-5. Inspect the complete recovered script for the molecular objectives and masking. Do not confuse the upstream Python-3.7 environment export with the modern adapted-run environment. `mhafr-upstream-environment.yaml` is a provenance record, not a tested lockfile for these wrappers. The upstream export contains old CUDA wheels and multiple RDKit distributions; no successful fresh installation is claimed.

## AutoDDI

Repository: https://github.com/Zhen-Peng-Wu/AutoDDI

Exact recorded commit: `fe8a39e87259251935197a0b39885cad91038a5c`. All 15 source hashes match the final seed-42/43/44 configs. Obtain the source separately and verify it:

```bash
git clone https://github.com/Zhen-Peng-Wu/AutoDDI.git external/AutoDDI
git -C external/AutoDDI checkout fe8a39e87259251935197a0b39885cad91038a5c
python scripts/verify_upstream.py autoddi external/AutoDDI
python scripts/run_autoddi.py --seed 42 --inputs INPUTS --source external/AutoDDI --run-root RUN_ROOT
```

No upstream license file was found in this checkout, so its source is not redistributed here. The recovered adaptation launchers and hash manifests are included. `autoddi-recorded-requirements.txt` retains the recorded core versions; additional required packages are explicitly unpinned where not recorded.

## Evaluation and side effects

`evaluation-requirements.txt` records the packages actually used to run the CPU evaluation in this preparation. It is not the historical training environment. `side-effects-requirements.txt` lists complete direct packages for recovered training and analysis workflows, with only known historical versions pinned. `side-effects-requirements.partial.txt` is retained as the original limited version record. Missing historical pins cannot be reconstructed from the Morgan environment.

To document a new successful environment, save `python --version`, `python -m pip freeze`, platform/GPU details and input/output hashes alongside its validation output. Record it as a reconstruction environment unless equivalence to the historical runtime is established.

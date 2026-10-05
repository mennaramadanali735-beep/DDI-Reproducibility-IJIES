# Recorded environments

The requirements below were transcribed from actual experiment metadata, not chosen as newer replacements. They are partial environment records, not fresh installation or compatibility results.

## Topology

`evidence/original__seed_42__config.json` records PyTorch 2.11.0+cu128, PyG 2.8.0.post1, NumPy 2.1.3, pandas 2.2.3 and scikit-learn 1.6.1. Install the recorded PyTorch build separately:

```bash
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/topology-requirements.txt
```

Training used a Tesla T4 in the Original seed-42 config. Match the recorded execution environment for comparison; bitwise GPU resume is not guaranteed by the original protocol. Feature regeneration also needs the RDKit version and descriptor names from its `build_identity.json`; the training config alone does not specify them.

## Morgan interaction baseline

`evidence/morgan__config.json` records Python 3.13.15 and the versions in `morgan-requirements.txt`. Use a separate environment from topology. The workflow also uses RDKit and SciPy; these recorded versions are included.

```bash
python -m pip install -r environments/morgan-requirements.txt
```

## Side effects

The selected training metadata records Python 3.13.15 and LightGBM 4.6.0, but does not record every imported dependency. `side-effects-requirements.partial.txt` lists that distinction. A resolved environment lock and fresh execution still need to be produced; the unpinned dependencies are not evidence of historical versions.

## Cold-start wrappers and Colab

The MHAFR wrappers need the original upstream source tree and its compatible environment. Do not substitute a current upstream revision without recording and validating the change. Colab-specific imports (`google.colab`) run in Colab; they are not ordinary dependencies for the package validator. `tqdm` is listed for convenience without a recovered historical pin.

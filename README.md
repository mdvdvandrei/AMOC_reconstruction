# AMOC reconstruction from surface fields (CMIP6 / CMIP5 + deep learning)

Deep learning pipeline for reconstructing Atlantic Meridional Overturning Circulation (AMOC) at 26.5°N from gridded sea-surface properties (SST, SSH), trained on ensembles of climate models. The code supports heteroscedastic (mean + uncertainty) networks, leave-one-model-out cross-validation. During training, anomalies of AMOC are computed relative to the piControl mean; during inference on observations, SST anomalies are computed relative to the 1880–1940 mean and SSH anomalies relative to the 1993–2023 mean

## Purpose of the code

This repository reproduces the end-to-end workflow used in our analyses: **obtain model output → preprocess to a common Zarr layout → climatology / normalization steps → train Bayesian-style or deterministic CNNs → evaluate on held-out models and on observations**.

## Basic instructions

Work in the **main directory of the clone** (the folder that contains `train/`, `conf/`, and `requirements.txt`). All example commands below assume your shell’s current working directory is that folder.

```
amoc/          models, datasets, model lists
preprocess/    CMIP and DUACS preprocessing
train/         Hydra training entry points
inference/     held-out and observational inference
figures/       paper figures and tables
notebooks/     figure notebooks
conf/          Hydra config
data/          processed observation files (NetCDF)
artifacts/     saved runs and result tables
```

### Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   
pip install --upgrade pip
pip install -r requirements.txt
```


### Data: CMIP6 / CMIP5 from ESGF

1. **Download** the variables and experiments you need from the [ESGF](https://esgf-node.llnl.gov/) (or your node mirror): typically monthly `tos`, sea surface height / `zos`, and AMOC-related target fields already in your Zarr pipeline, plus the scenarios used in config (e.g. `historical`, `piControl`, SSPs for CMIP6 or `historical` / `rcp85` for CMIP5).  
2. **Place** raw or staging data in a layout you can point preprocessing scripts at (see hard-coded `base` / paths inside `preprocess/preprocess_cmip6_stream_zarr.py`, `preprocess/preprocess_cmip5_stream_zarr.py`, and related scripts).  
3. Adjust any **absolute paths** left from the original workstation: search the repo for path fragments you no longer use and replace with your directories.

### Preprocessing to Zarr

- **`preprocess/preprocess_cmip6_stream_zarr.py`** and **`preprocess/preprocess_cmip5_stream_zarr.py`** stream monthly fields onto a regular grid, apply masks / regridding as configured, and write **`output_<model>.zarr`** (or the template set in each script) under your chosen output root. This produces the **multi-model Zarr stores** the dataset classes expect. **`preprocess/preprocess_cmip5_climatology.py`** covers the CMIP5 climatology step.
- **`preprocess/preprocess_climatology_zarr.py`** — climatologies, and additional grid / scenario processing in the same chain; run after raw CMIP is on disk and paths in the script match your machine.
- **`amoc/datasets_cmip_zarr.py`** — PyTorch `Dataset` classes and sequence helpers; see the Zarr group layout (`scenario` / `x` / `y` / `piControl_stats`, etc.).


### Training

Run the two-stage heteroscedastic trainer (Hydra + optional Weights & Biases):

```bash
python train/train_weights_BNN.py
```

Checkpoints are written under **`weights/<save_name>/`** by default (`weights_root` in **`conf/config_bnn.yaml`**). 

### Inference

- **Across climate models (held-out folds, evaluation):**  
  `python inference/inference.py`  

- **Real-world / observation-based inference and manuscript-style figures:**  
  `python inference/inference_obs_rw.py`  

- **CMIP5-specific inference**:  
  `python inference/inference_cmip5.py`

### Additional graphs and tables

These are optional but useful for papers and supplements:

| Script | Role |
|--------|------|
| **`figures/plot_trend_violins.py`** | Trend-error violins; **`plot_per_model_reconstructions(..., tag="LPF120")`** writes **`per_model_reconstructions_LPF120.png`** from `series.parquet` / `metrics.parquet` under an artifact run directory. |
| **`figures/plot_ssh_rapid_ecco.py`** | SSH / RAPID / ECCO-style panels. |
| **`figures/scatter_figure.py`** | MSE vs number of training models |
| **`figures/trend_tables.py`** | Observational trend tables. |

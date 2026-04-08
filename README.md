# AMOC reconstruction from surface fields (CMIP6 / CMIP5 + deep learning)

Deep learning pipeline for reconstructing Atlantic Meridional Overturning Circulation (AMOC) at 26.5°N from gridded sea-surface properties (SST, SSH), trained on ensembles of climate models. The code supports heteroscedastic (mean + uncertainty) networks, leave-one-model-out cross-validation, and a full suite of evaluation and publicationfigures.

## Purpose of the code

This repository reproduces the end-to-end workflow used in our analyses: **obtain model output → preprocess to a common Zarr layout → climatology / normalization steps → train Bayesian-style or deterministic CNNs → evaluate on held-out models and on observations**.

## Basic instructions

Work in the **main directory of the clone** (the folder that contains `train_weights_BNN.py`, `conf/`, and `requirements.txt`). All example commands below assume your shell’s current working directory is that folder.

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
2. **Place** raw or staging data in a layout you can point preprocessing scripts at (see hard-coded `base` / paths inside `preprocess_cmip6_stream_zarr.py`, `preprocess_cmip5_stream_zarr.py`, and related scripts).  
3. Adjust any **absolute paths** left from the original workstation: search the repo for path fragments you no longer use and replace with your directories.

### Preprocessing to Zarr

- **`preprocess_cmip6_stream_zarr.py`** (and **`preprocess_cmip5_stream_zarr.py`**) stream monthly fields onto a regular grid, apply masks / regridding as configured, and write **`output_<model>.zarr`** (or the template set in each script) under your chosen output root. This produces the **multi-model Zarr stores** the dataset classes expect. **`preprocess_cmip5_zarr.py`** covers CMIP5-style paths where applicable.  
- **`preprocess_climatology_zarr.py`** — climatologies, and additional grid / scenario processing in the same chain; run after raw CMIP is on disk and paths in the script match your machine.  
- **`datasets_cmip_zarr.py`** — PyTorch `Dataset` classes and sequence helpers; see the Zarr group layout (`scenario` / `x` / `y` / `piControl_stats`, etc.).


### Training

Run the two-stage heteroscedastic trainer (Hydra + optional Weights & Biases):

```bash
python train_weights_BNN.py
```

Checkpoints are written under **`weights/<save_name>/`** by default (`weights_root` in **`conf/config_bnn.yaml`**). 

### Inference

- **Across climate models (held-out folds, evaluation):**  
  `python inference.py`  

- **Real-world / observation-based inference and manuscript-style figures:**  
  `python inference_obs_new.py`  

- **CMIP5-specific inference**:  
  `python inference_cmip5.py`

### Additional graphs and tables

These are optional but useful for papers and supplements:

| Script | Role |
|--------|------|
| **`plot_trend_violins.py`** | Trend-error violins; **`plot_per_model_reconstructions(..., tag="LPF120")`** writes **`per_model_reconstructions_LPF120.png`** from `series.parquet` / `metrics.parquet` under an artifact run directory. |
| **`plot_ssh_rapid_ecco.py`** | SSH / RAPID / ECCO-style panels. |
| **`scatter_figure.py`** | MSE vs number of training models |
| **`compute_trends_dl_vs_sim.py`** | Compare DL vs simulated trends. |

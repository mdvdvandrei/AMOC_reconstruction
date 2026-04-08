#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"]        = "16"
os.environ["MKL_NUM_THREADS"]        = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import TwoBranchModel, CustomSqueezeNet, ResidualCNN, SimpleCNN, SimpleViT
from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE

# ----------------------- utils -----------------------
def _reinit_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias,   0.0)

np.random.seed(1336)
torch.manual_seed(1336)

def build_model(in_channels, out_dim, model_name, device):
    cls = {
        "TwoBranchModel":   TwoBranchModel,
        "SimpleCNN":        SimpleCNN,
        "CustomSqueezeNet": CustomSqueezeNet,
        "ResidualCNN":      ResidualCNN,
        "SimpleViT":        SimpleViT
    }[model_name]
    return cls(in_channels, out_dim).to(device)

@torch.no_grad()
def _gather_series(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        out = model(x)
        d = min(out.shape[1], y.shape[1])
        preds.append(out[:, :d].cpu().numpy())
        trues.append(y[:, :d].cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    ds = loader.dataset
    base_ds = getattr(ds, 'dataset', ds)
    times = None
    for cand in ('times','time','time_index','time_idx'):
        if hasattr(base_ds, cand):
            t = getattr(base_ds, cand)
            if isinstance(t, (np.ndarray, list, tuple)) and len(t) == len(trues):
                times = np.array(t)
            break
    return preds, trues, times

def metrics_loop(model, loader, crit, device):
    model.eval()
    mse_sum, n_samples = 0.0, 0
    preds, trgs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            out = model(x)
            d = min(out.shape[1], y.shape[1])
            out, y = out[:, :d], y[:, :d]
            mse_sum   += crit(out, y).item() * x.size(0)
            n_samples += x.size(0)
            preds.append(out.cpu().numpy())
            trgs.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trgs  = np.concatenate(trgs,  axis=0)
    rss = ((trgs - preds) ** 2).sum(0)
    tss = ((trgs - trgs.mean(0)) ** 2).sum(0)
    r2  = 1 - rss / tss
    corr = np.corrcoef(preds[:, 0], trgs[:, 0])[0, 1]
    return mse_sum / n_samples, float(corr), float(np.mean(r2))

def train_on_subset(model_subset_name, train_ds, val_ds, test_ds, cfg, device):
    os.makedirs("vesa_dlya_subsetov", exist_ok=True)
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=16)
    va_loader = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    te_loader = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    in_ch   = train_ds[0][0].shape[0]
    out_dim = train_ds[0][1].shape[0]
    model   = build_model(in_ch, out_dim, cfg.model, device)
    model.apply(_reinit_weights)

    crit  = nn.MSELoss()
    opt   = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.t0, T_mult=2)

    best_val = np.inf
    best_path = f"vesa_dlya_subsetov_tos/best_{model_subset_name}.pth"

    for ep in range(cfg.epochs):
        model.train()
        running = 0.
        for x, y in tr_loader:
            x, y = x.to(device).float(), y.to(device).float()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        val_loss, _, _ = metrics_loop(model, va_loader, crit, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        print(f"[ep {ep:03d}] train_loss={running/len(tr_loader.dataset):.6f}  val_loss={val_loss:.6f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mse, test_cor, test_r2 = metrics_loop(model, te_loader, crit, device)
    pred_series, true_series, times = _gather_series(model, te_loader, device)
    return test_mse, test_r2, test_cor, pred_series, true_series, times

# ------------------------- main --------------------------
if __name__ == "__main__":
    class Cfg: pass
    cfg = Cfg()
    cfg.zarr_dir       = "monthly_stream_zarr"
    cfg.x_vars         = ["tos"]
    cfg.model          = "ResidualCNN"
    cfg.batch_size     = 128
    cfg.lr             = 1e-4
    cfg.weight_decay   = 1e-2
    cfg.t0             = 256
    cfg.epochs         = 64
    cfg.noise          = None
    cfg.lpf            = "LPF120"
    cfg.target_var     = "y"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    all_models = [
        "ACCESS-ESM1-5","ACCESS-CM2","GFDL-ESM4","FGOALS-g3",
        "MRI-ESM2-0","MIROC6","CanESM5","GISS-E2-1-G",
        "NorESM2-LM","NorESM2-MM","HadGEM3-GC31-LL","UKESM1-1-LL",
        "CMCC-ESM2","HadGEM3-GC31-MM","MPI-ESM1-2-HR","CanESM5-CanOE",
        "GFDL-CM4","IPSL-CM6A-LR", "CAS-ESM2-0"
    ]

    test_members = [f"r{i}i1p1f1" for i in range(1, 12)]  # CESM2 r1..r11
    base_dir = os.path.join(os.getcwd(), cfg.zarr_dir)

    subset_sizes = [1] #[1, 4, 8, 15, 18]
    records = []

    # N=1 plotting buffers
    n1_recon_draws = []
    n1_truth_cached = None
    n1_times_cached = None

    # SSP scenarios included in training
    ssp_list = ["ssp126", "ssp245", "ssp585"]

    for size in subset_sizes:
        n_subsets = 18 if size in (1, 4) else 5

        for draw in range(n_subsets):
            subset = np.random.choice(all_models, size, replace=False).tolist()
            name   = f"{size}models_draw{draw}"

            # ------------- load all samples for train/val split -------------
            # historical + piControl + all listed SSPs
            ds_all = PreprocessedCMIP6Dataset_LE(
                zarr_dir       = base_dir,
                models         = subset,
                x_vars         = cfg.x_vars,
                scenarios      = ["historical", "piControl"] + ssp_list,
                target_group   = cfg.target_var,
                output_type    = "max",
                member_selection = "all",
                selected_lats  = [26.5],
                lpf            = cfg.lpf,
                noise          = cfg.noise,
            )

            # Per model: first historical member -> VAL
            # Remaining historical members -> TRAIN, rule depends on subset size:
            #   - size == 1: all other historical members -> TRAIN
            #   - size > 1: only the second historical -> TRAIN (legacy behaviour)
            first_hist = {}
            extra_hists_for_train = {mdl: [] for mdl in subset}

            for mdl in subset:
                hist_mems = sorted(ds_all.get_members_for_model_scenario(mdl, "historical"))
                if not hist_mems:
                    raise RuntimeError(f"No historical members for {mdl}")
                first_hist[mdl] = hist_mems[0]

                if size == 1:
                    # all other historical members -> train
                    extra_hists_for_train[mdl] = hist_mems[1:]
                else:
                    # second historical only (if present)
                    extra_hists_for_train[mdl] = hist_mems[1:2]

            val_idx, train_idx = [], []
            for i in range(len(ds_all)):
                info = ds_all.get_sample_info(i)  # {model, scenario, member}
                mdl, scn, mem = info["model"], info["scenario"], info["member"]

                if scn == "historical":
                    if mem == first_hist[mdl]:
                        val_idx.append(i)  # first historical -> VAL
                    elif mem in set(extra_hists_for_train[mdl]):
                        train_idx.append(i)  # extras (N=1: all; N>1: second only)
                    # other historical members (if any, N>1) are skipped
                elif scn == "piControl":
                    train_idx.append(i)
                elif scn in ssp_list:
                    train_idx.append(i)  # all SSP samples -> train
                else:
                    # safety: do not send unknown scenarios to val
                    train_idx.append(i)

            train_ds = Subset(ds_all, train_idx)
            val_ds   = Subset(ds_all, val_idx)

            # ---------------------- TEST ---------------------------
            test_ds = PreprocessedCMIP6Dataset_LE(
                zarr_dir        = base_dir,
                models          = ["CESM2"],
                x_vars          = cfg.x_vars,
                scenarios       = ["historical"],
                target_group    = cfg.target_var,
                output_type     = "max",
                member_selection = test_members,
                selected_lats   = [26.5],
                lpf             = cfg.lpf,
                noise           = cfg.noise
            )

            print(f"\n=== subset ({size} models): {subset} ===")
            print('Train / Val sizes:', len(train_ds), len(val_ds))

            test_mse, test_r2, test_cor, pred_series, true_series, times = \
                train_on_subset(name, train_ds, val_ds, test_ds, cfg, device)

            records.append({
                "n_models": size,
                "subset":   ",".join(subset),
                "test_mse": test_mse,
                "test_r2":  test_r2,
                "test_cor": test_cor
            })
            pd.DataFrame(records).to_csv(
                "mse_vs_nmodels_ssp_in_train_hist_rule_tos.csv", index=False
            )

            # N=1: collect reconstructions for the overlay figure
            if size == 1:
                n1_recon_draws.append(pred_series[:, 0].astype(float))
                if n1_truth_cached is None:
                    n1_truth_cached = true_series[:, 0].astype(float)
                    n1_times_cached = times

            if size == 18:
                break

    # ---------- scalar MSE vs n_models ----------
    df = pd.DataFrame(records)
    df.to_csv("mse_vs_nmodels_ssp_in_train_hist_rule_tos.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.scatter(df["n_models"], df["test_mse"], s=50)
    plt.xlabel("Number of models in training subset")
    plt.ylabel("CESM2 test MSE")
    plt.title("MSE vs # of training models")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mse_vs_nmodels_ssp_in_train_hist_rule_tos.png", dpi=300)
    plt.close()

    # ---------- N=1 figure (gray draws + mean vs truth) ----------
    if n1_recon_draws:
        xx = n1_times_cached if n1_times_cached is not None else np.arange(len(n1_recon_draws[0]))
        stack = np.stack(n1_recon_draws, axis=0)
        mean_recon = stack.mean(axis=0)

        plt.figure(figsize=(10, 4.5))
        for k in range(stack.shape[0]):
            plt.plot(xx, stack[k], linewidth=1.0, alpha=0.35, color="gray")
        plt.plot(xx, mean_recon, linewidth=2.2, color="black", label="Mean reconstruction (N=1)")
        if n1_truth_cached is not None:
            plt.plot(xx, n1_truth_cached, linewidth=1.6, color="#1f77b4", label="CESM2 simulated AMOC")

        plt.title("N=1: Reconstructions across draws vs CESM2 simulated AMOC")
        plt.xlabel("Time")
        plt.ylabel("AMOC (Sv or target units)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.savefig("N1_reconstructions_vs_CESM2_with_SSPs_tos.png", dpi=300)
        plt.close()

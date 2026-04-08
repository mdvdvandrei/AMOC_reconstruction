#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================
# ConvNeXt/ResidualCNNHetBaseline, two-stage training (μ-only -> μ+σ)
# Uses ALL variables from cfg.x_vars at once.
# Cross-val: Leave-One-Model-Out over cfg.val_model(s)
# Data: PreprocessedCMIP6Dataset_LE
# Coords: real lat (+ sin/cos(lon)) as extra channels
# Logging: W&B
# =============================================================

import os, math, json, copy
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from models import ResidualCNNHetBaseline


torch.set_default_dtype(torch.float32)

# ───────────────────────────── Misc utils ─────────────────────────────

def seed_everything(seed: int | None):
    if seed is None: return
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, str): return [s.strip() for s in x.split(",") if s.strip()]
    return [x]

# Gaussian CRPS (репортинг)
def _erf_np(x):
    s = np.sign(x); x = np.abs(x)
    t = 1.0/(1.0+0.3275911*x)
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    y = 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*np.exp(-x*x)
    return s*y
def gaussian_crps(mu, sigma, y, eps=1e-8):
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    Phi = 0.5*(1.0 + _erf_np(z/np.sqrt(2.0)))
    phi = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z*z)
    return float(np.mean(sigma*( z*(2.0*Phi-1.0) + 2.0*phi - 1.0/np.sqrt(np.pi) )))

# ─────────────────────── Coords wrapper (real lat/lon) ───────────────────────

def make_coord_tensor_from_arrays(lat_vals, W, lon_vals=None, use_sin_cos=True):
    lat_vals = np.asarray(lat_vals, dtype=np.float32)
    H = len(lat_vals)
    lat = torch.from_numpy(lat_vals).view(1, H, 1).expand(1, H, W)
    lat_min = float(lat_vals.min()); lat_max = float(lat_vals.max())
    lat_norm = 2.0 * (lat - lat_min) / max(1e-6, (lat_max - lat_min)) - 1.0

    if lon_vals is None:
        delta = 360.0 / W
        lon_vals = (np.arange(W, dtype=np.float32) + 0.5) * delta - 180.0
    else:
        lon_vals = np.asarray(lon_vals, dtype=np.float32)
        assert len(lon_vals) == W, f"lon_vals length={len(lon_vals)} must equal W={W}"
    lon = torch.from_numpy(lon_vals).view(1, 1, W).expand(1, H, W)

    if use_sin_cos:
        lon_rad = lon * (math.pi / 180.0)
        coord = torch.cat([lat_norm, torch.sin(lon_rad), torch.cos(lon_rad)], dim=0)
    else:
        lon_min = float(lon_vals.min()); lon_max = float(lon_vals.max())
        lon_norm = 2.0 * (lon - lon_min) / max(1e-6, (lon_max - lon_min)) - 1.0
        coord = torch.cat([lat_norm, lon_norm], dim=0)
    return coord  # [C_coord, H, W]

class WithCoordsFromArrays(torch.utils.data.Dataset):
    """Добавляет координатные каналы на основе реальных lat (и опц. lon)."""
    def __init__(self, base_ds, lat_vals, lon_vals=None, use_sin_cos=True):
        super().__init__()
        self.base = base_ds
        x0, *_ = base_ds[0]
        _, H, W = x0.shape
        lat_vals = np.asarray(lat_vals, dtype=np.float32)
        assert len(lat_vals) == H, f"len(lat_vals)={len(lat_vals)} must match input H={H}"
        self.coord = make_coord_tensor_from_arrays(lat_vals, W, lon_vals, use_sin_cos=use_sin_cos)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x, y = item[:2]
        x = torch.cat([x, self.coord], dim=0)
        return (x, y, *item[2:])

# ───────────────────── Eval helpers ─────────────────────

def get_mu_loss(name="huber", beta=0.5):
    name = str(name).lower()
    if name == "huber": return nn.SmoothL1Loss(beta=float(beta))
    if name == "l1":    return nn.L1Loss()
    if name == "mse":   return nn.MSELoss()
    raise ValueError(f"unknown mu loss: {name}")

@torch.no_grad()
def eval_epoch_mu(model, loader, device, mu_loss_fn):
    model.eval()
    run=0.0; mus=[]; ys=[]
    for batch in loader:
        x,y = batch[:2]
        x=x.to(device); y=y.to(device)
        mu,_ = model(x)
        loss = mu_loss_fn(mu,y)
        run += loss.item()*x.size(0)
        mus.append(mu.cpu()); ys.append(y.cpu())
    mu = torch.cat(mus).numpy(); y = torch.cat(ys).numpy()
    rmse = float(np.sqrt(np.mean((mu-y)**2)))
    mae  = float(np.mean(np.abs(mu-y)))
    r2   = float(1. - (( (y-mu)**2 ).sum() / ((y - y.mean(axis=0))**2).sum() + 1e-12))
    corr = float(np.corrcoef(mu.ravel(), y.ravel())[0,1]) if (mu.std()>0 and y.std()>0) else 0.0
    return {"val_loss": run/len(y), "rmse": rmse, "mae": mae, "r2": r2, "corr": corr}

@torch.no_grad()
def eval_epoch_joint(model, loader, device, eps=1e-6):
    model.eval()
    nll = nn.GaussianNLLLoss(eps=eps)
    run=0.0; mus=[]; ys=[]; sigmas=[]
    for batch in loader:
        x,y = batch[:2]
        x=x.to(device); y=y.to(device)
        mu, logv = model(x)
        var = torch.exp(logv)
        loss = nll(mu, y, var)
        run += loss.item()*x.size(0)
        mus.append(mu.cpu()); ys.append(y.cpu()); sigmas.append(torch.sqrt(var).cpu())
    mu = torch.cat(mus).numpy(); y = torch.cat(ys).numpy()
    sigma = torch.cat(sigmas).numpy()
    rmse = float(np.sqrt(np.mean((mu-y)**2)))
    mae  = float(np.mean(np.abs(mu-y)))
    r2   = float(1. - (( (y-mu)**2 ).sum() / ((y - y.mean(axis=0))**2).sum() + 1e-12))
    corr = float(np.corrcoef(mu.ravel(), y.ravel())[0,1]) if (mu.std()>0 and y.std()>0) else 0.0
    crps = gaussian_crps(mu, sigma, y)
    return {"val_nll": run/len(y), "rmse": rmse, "mae": mae, "r2": r2, "corr": corr, "crps": crps}

# ───────────────────── Optim helpers ─────────────────────

def set_requires_grad(module, flag: bool):
    if module is None: return
    for p in module.parameters(): p.requires_grad = flag

def build_param_groups(model, wd=1e-4):
    """WD=0 для норм и bias; WD=wd для остального."""
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_bias = name.endswith("bias")
        is_ln   = ("ln" in name.lower()) or ("layernorm" in name.lower()) or ("batchnorm" in name.lower()) or ("bn" in name.lower())
        (nodecay if (is_bias or is_ln) else decay).append(p)
    groups=[]
    if decay:   groups.append({"params": decay,   "weight_decay": wd})
    if nodecay: groups.append({"params": nodecay, "weight_decay": 0.0})
    return groups

# ───────────────────── Heads discovery (μ/σ) ─────────────────────

def get_mu_head(model):
    # распространённые варианты названий
    for name in ["mu_head", "mean_head", "head_mu"]:
        if hasattr(model, name): return getattr(model, name)
    raise AttributeError("Cannot find μ-head on model (expected one of: mu_head/mean_head/head_mu).")

def get_var_head(model):
    # поддерживаем и var_head, и log_head (у некоторых реализаций лог-варианса)
    for name in ["var_head", "log_head", "sigma_head"]:
        if hasattr(model, name): return getattr(model, name)
    raise AttributeError("Cannot find σ-head on model (expected one of: var_head/log_head/sigma_head).")

def init_var_head_bias_with_logvar(model, logvar_vec: np.ndarray):
    """Инициализируем bias последнего Linear в σ-голове лог-дисперсией train-таргетов."""
    var_head = get_var_head(model)
    last_linear = None
    for m in reversed(list(var_head.modules())):
        if isinstance(m, nn.Linear):
            last_linear = m
            break
    if last_linear is not None and last_linear.bias is not None:
        with torch.no_grad():
            nn.init.zeros_(last_linear.weight)
            last_linear.bias.copy_(torch.from_numpy(logvar_vec))

# ───────────────────── Train loops ─────────────────────

def train_epoch_mu_only(model, loader, opt, device, mu_loss_fn):
    model.train(); run=0.0
    for batch in loader:
        x,y = batch[:2]
        x=x.to(device); y=y.to(device)
        mu,_ = model(x)
        loss = mu_loss_fn(mu,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item()*x.size(0)
    return run/len(loader.dataset)

def train_epoch_joint_nll(model, loader, opt, device, var_lambda=0.0, eps=1e-6):
    model.train(); nll = nn.GaussianNLLLoss(eps=eps); run=0.0
    for batch in loader:
        x,y = batch[:2]
        x=x.to(device); y=y.to(device)
        mu, logv = model(x)
        var = torch.exp(logv)
        loss = nll(mu, y, var)
        if var_lambda and var_lambda > 0:
            loss = loss + var_lambda*(logv).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item()*x.size(0)
    return run/len(loader.dataset)

# ───────────────────── Build model ─────────────────────

def build_model(in_ch, out_dim, flat_hw, cfg, device):
    margs = getattr(cfg, "model_args", {})
    # Пример с ResidualCNNHetBaseline (как в вашем текущем коде)
    model = ResidualCNNHetBaseline(
        in_channels=in_ch,
        out_dim=out_dim,
        flat_hw=list(flat_hw),
        p_drop=float(margs.get("dropout", 0.5)),
        tanh_scale=float(margs.get("sigma_tanh_scale", 8.0))
    ).to(device)

    # Если хотите ConvNeXtHet — раскомментируйте и закомментируйте блок выше
    # dims   = tuple(margs.get("dims",   [48,96,192,384]))
    # depths = tuple(margs.get("depths", [3,3,9,3]))
    # model = ConvNeXtHet(
    #     in_ch=in_ch, out_dim=out_dim, flat_hw=flat_hw,
    #     dims=dims, depths=depths, p_drop=float(margs.get("dropout", 0.3)),
    #     tanh_scale=float(margs.get("sigma_tanh_scale", 8.0)),
    #     use_flat_skip=bool(margs.get("use_flat_skip", False)),
    #     anti_alias=bool(margs.get("anti_alias", True)),
    #     blur_sigma=float(margs.get("blur_sigma", 1.0))
    # ).to(device)

    return model

# ───────────────────── Main (Hydra) ─────────────────────

@hydra.main(config_path="conf", config_name="config_bnn")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(int(getattr(cfg, "seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.wandb_project,
               name=f"CMIP6_ALLVARS_TwoStage_{cfg.save_name}",
               config=OmegaConf.to_container(cfg, resolve=True))

    from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE
    base_dir = os.path.join(get_original_cwd(), cfg.base_dir)
    weights_root = str(getattr(cfg, "weights_root", "weights"))

    models_all = _as_list(cfg.models)
    val_models = _as_list(getattr(cfg, "val_model", None)) or _as_list(getattr(cfg, "val_models", None))
    scenarios  = _as_list(cfg.scenarios)
    x_vars     = _as_list(cfg.x_vars)  # <<< используем весь список разом!

    # coords options
    add_coords  = bool(getattr(cfg.data, "add_coords", True))
    use_sincos  = bool(getattr(cfg.data, "use_sin_cos", True))
    lat_values  = getattr(cfg.data, "lat_values", None)
    lon_values  = getattr(cfg.data, "lon_values", None)

    results = []

    for v_model in val_models:
        train_models = [m for m in models_all if m != v_model]

        # ─── Datasets (RAW) — ПЕРЕДАЁМ ВСЕ x_vars СПИСКОМ ───
        ds_train_raw = PreprocessedCMIP6Dataset_LE(
            zarr_dir=base_dir, models=train_models, x_vars=x_vars,
            scenarios=scenarios, target_group=cfg.target_var, output_type=cfg.output_type,
            member_selection="first",
            selected_lats=[float(l) for l in _as_list(cfg.selected_lats)],
            lpf=cfg.lpf, noise=cfg.noise
        )
        ds_val_raw = PreprocessedCMIP6Dataset_LE(
            zarr_dir=base_dir, models=[v_model], x_vars=x_vars,
            scenarios=_as_list(cfg.val_scenario), target_group=cfg.target_var, output_type=cfg.output_type,
            member_selection="all",
            selected_lats=[float(l) for l in _as_list(cfg.selected_lats)],
            lpf=cfg.lpf, noise=None
        )
        if len(ds_val_raw) == 0:
            print(f"[skip] empty val for {v_model} (ALLVARS)")
            continue

        # ─── Shapes ───
        x0, y0 = ds_val_raw[0][:2]
        in_ch_base, H, W = x0.shape
        out_dim = y0.shape[0]
        flat_hw = (H, W)

        # ─── Wrap with coords ───
        if add_coords:
            if (lat_values is None) or (len(lat_values) == 0):
                if hasattr(ds_train_raw, "lat") and isinstance(ds_train_raw.lat, (list, np.ndarray)):
                    lat_vals = np.asarray(ds_train_raw.lat, dtype=np.float32)
                elif hasattr(ds_train_raw, "lats") and isinstance(ds_train_raw.lats, (list, np.ndarray)):
                    lat_vals = np.asarray(ds_train_raw.lats, dtype=np.float32)
                else:
                    raise ValueError("Provide data.lat_values in config_bnn.yaml (list of latitudes).")
            else:
                lat_vals = np.asarray(lat_values, dtype=np.float32)

            if (lon_values is None) or (len(_as_list(lon_values)) == 0):
                lon_vals = None  # uniform
            else:
                lon_vals = np.asarray(lon_values, dtype=np.float32)

            ds_train = WithCoordsFromArrays(ds_train_raw, lat_vals=lat_vals, lon_vals=lon_vals, use_sin_cos=use_sincos)
            ds_val   = WithCoordsFromArrays(ds_val_raw,   lat_vals=lat_vals, lon_vals=lon_vals, use_sin_cos=use_sincos)
        else:
            ds_train, ds_val = ds_train_raw, ds_val_raw

        # Обновлённый in_ch с учётом координат
        x0_aug, _ = ds_val[0][:2]
        in_ch = x0_aug.shape[0]

        # ─── Model ───
        model = build_model(in_ch, out_dim, flat_hw, cfg, device)

        # ─── Loaders ───
        train_loader = DataLoader(ds_train, batch_size=int(cfg.training.batch_size), shuffle=True,
                                  num_workers=cfg.training.num_workers, pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=int(cfg.training.batch_size), shuffle=False,
                                  num_workers=max(1, int(cfg.training.num_workers)//2), pin_memory=True)

        lr   = float(cfg.training.learning_rate)
        wd   = float(cfg.training.weight_decay)
        me   = int(cfg.training.mu_epochs)
        we   = int(cfg.training.warmup_epochs)

        # ─── σ warm-start via train y variance ───
        train_targets = []
        for batch in train_loader:
            yb = batch[1].to(torch.float32)
            train_targets.append(yb.cpu().numpy())
        train_targets = np.concatenate(train_targets, axis=0) if len(train_targets) else np.empty((0,out_dim),np.float32)
        if train_targets.size > 0:
            init_logvar = np.log(np.var(train_targets, axis=0) + 1e-6).astype(np.float32)
            try:
                init_var_head_bias_with_logvar(model, init_logvar)
            except Exception as e:
                print(f"[warn] var-head init skipped: {e}")

        # ─── Stage-1: μ-only ───
        # Фризим σ-голову (и опциональные skip-ветки для σ, если есть)
        try:
            set_requires_grad(get_var_head(model), False)
        except Exception as e:
            print(f"[warn] cannot freeze var head: {e}")
        for name in ["skip_var", "skip_log"]:  # поддержка разных реализаций
            if hasattr(model, name): set_requires_grad(getattr(model, name), False)

        mu_loss_fn = get_mu_loss(str(getattr(cfg.mu, "loss", "huber")).lower(),
                                 beta=float(getattr(cfg.mu, "huber_beta", 0.5)))

        opt_mu = AdamW(build_param_groups(model, wd=wd), lr=lr, betas=(0.9,0.999))
        def lr_lambda(e):
            if we > 0 and e < we: return (e+1)/max(1,we)
            t = (e - we)/max(1, (me - we)); t = min(max(t,0.0),1.0)
            return 0.5*(1+math.cos(math.pi*t))
        sched_mu = torch.optim.lr_scheduler.LambdaLR(opt_mu, lr_lambda=lr_lambda)

        weight_dir = os.path.join(get_original_cwd(), weights_root, cfg.save_name, "ALLVARS", v_model)
        os.makedirs(weight_dir, exist_ok=True)
        best_mu_path = os.path.join(weight_dir, "best_stage1_mu.pt")

        best_r2 = -1e9; best_metrics_mu=None

        for epoch in range(me):
            tr = train_epoch_mu_only(model, train_loader, opt_mu, device, mu_loss_fn)
            sched_mu.step()
            metrics_mu = eval_epoch_mu(model, val_loader, device, mu_loss_fn)
            wandb.log({
                "fold": f"{v_model}/ALLVARS", "stage":"mu_only",
                "mu_train_loss": tr, "mu_val_loss": metrics_mu["val_loss"],
                "mu_rmse": metrics_mu["rmse"], "mu_mae": metrics_mu["mae"],
                "mu_r2": metrics_mu["r2"], "mu_corr": metrics_mu["corr"],
                "epoch": epoch, "lr_mu": opt_mu.param_groups[0]["lr"]
            })
            print(f"[μ] ALLVARS/{v_model} e{epoch:03d} | tr {tr:.4f} | val {metrics_mu['val_loss']:.4f} "
                  f"| R2 {metrics_mu['r2']:.3f} | corr {metrics_mu['corr']:.3f} "
                  f"| RMSE {metrics_mu['rmse']:.3f} | MAE {metrics_mu['mae']:.3f}")
            if metrics_mu["r2"] > best_r2:
                best_r2 = metrics_mu["r2"]; best_metrics_mu=metrics_mu
                torch.save(model.state_dict(), best_mu_path)

        if os.path.isfile(best_mu_path):
            model.load_state_dict(torch.load(best_mu_path, map_location=device), strict=False)
            print(f"[Stage-1] loaded best μ → {best_mu_path}")

        # ─── Stage-2: joint μ+σ ───
        set_requires_grad(model, True)  # разморозить всё
        opt_joint = AdamW(build_param_groups(model, wd=wd), lr=lr, betas=(0.9,0.999))
        t0 = int(cfg.training.t0)
        sched_joint = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_joint, T_0=t0, T_mult=2)
        var_lambda0 = float(cfg.training.var_lambda0)
        var_tau     = float(cfg.training.var_lambda_tau)
        patience    = int(cfg.training.early_stop_patience)
        total_ep2   = int(cfg.training.epochs) - me

        best_val_nll = math.inf; best_metrics_joint=None; wait=0
        for step in range(total_ep2):
            global_epoch = me + step
            lam = float(var_lambda0 * math.exp(- step / max(1.0, var_tau)))
            tr_nll = train_epoch_joint_nll(model, train_loader, opt_joint, device, var_lambda=lam)
            sched_joint.step()

            metrics_joint = eval_epoch_joint(model, val_loader, device)
            wandb.log({
                "fold": f"{v_model}/ALLVARS", "stage":"joint_nll",
                "train_nll": tr_nll, "val_nll": metrics_joint["val_nll"],
                "rmse": metrics_joint["rmse"], "mae": metrics_joint["mae"],
                "r2": metrics_joint["r2"], "corr": metrics_joint["corr"],
                "crps": metrics_joint["crps"],
                "var_lambda": lam, "epoch": global_epoch,
                "lr_joint": opt_joint.param_groups[0]["lr"]
            })
            print(f"[joint] ALLVARS/{v_model} e{global_epoch:03d} | trNLL {tr_nll:.4f} | valNLL {metrics_joint['val_nll']:.4f} "
                  f"| R2 {metrics_joint['r2']:.3f} | corr {metrics_joint['corr']:.3f} "
                  f"| RMSE {metrics_joint['rmse']:.3f} | CRPS {metrics_joint['crps']:.3f} | λ {lam:.2e}")

            if metrics_joint["val_nll"] < best_val_nll:
                best_val_nll = metrics_joint["val_nll"]; best_metrics_joint = metrics_joint
                torch.save(model.state_dict(), os.path.join(weight_dir, "best_stage2_joint.pt")); wait=0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[joint] early stop (patience={patience})")
                    break

        # финальная оценка
        final = eval_epoch_joint(model, val_loader, device)
        fold_summary = {
            "val_model": v_model,
            "vars": "+".join(x_vars),
            "stage1_best_r2": float(best_r2),
            "stage2_val_nll": float(final["val_nll"]),
            "rmse": float(final["rmse"]), "mae": float(final["mae"]),
            "r2": float(final["r2"]), "corr": float(final["corr"]),
            "crps": float(final["crps"])
        }
        results.append(fold_summary)
        with open(os.path.join(weight_dir, "summary.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)
        print(f"[summary] {fold_summary}")

    # CSV по всем фолдам
    csv_out = os.path.join(get_original_cwd(), f"results_allvars_twostage_{cfg.save_name}.csv")
    pd.DataFrame(results).to_csv(csv_out, index=False)
    print(f"[done] wrote {csv_out}")
    wandb.finish()

if __name__ == "__main__":
    main()

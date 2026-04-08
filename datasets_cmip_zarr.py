"""PyTorch ``Dataset`` classes for CMIP6 Zarr stores (inputs, targets, sequence windows)."""
from __future__ import annotations

import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import zarr

from torch.utils.data import Dataset
import torch, zarr, os, numpy as np

class PreprocessedCMIP6Dataset(Dataset):
    def __init__(
        self,
        zarr_dir,
        models,
        x_vars,
        scenarios,
        *,
        target_group="y",          # NEW  ←  "y" | "y_no_ekman" | "ekman"
        output_type="max",
        selected_lats=None,
        transform=None,
        lpf="raw",
        local=False,
        noise = None
    ):
        assert target_group in {"y", "y_no_ekman", "ekman"}
        self.target_group = target_group

        # storage -----------------------
        self.zarr_dir, self.models, self.x_vars = zarr_dir, models, x_vars
        self.scenarios, self.output_type = scenarios, output_type
        self.selected_lats, self.transform = selected_lats, transform
        self.lpf, self.local = lpf, local
        self.noise = noise
        # sampling stride -----------------------------------
        self.num_skip = 64
        self.stride = 6 if lpf in ("LPF120") else 12

        # ----------------------------------------------------------------
        # global & per‑model norm‑stats  
        # ----------------------------------------------------------------
        self.global_norm_stats = {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}}
        self.norm_stats = {m: {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}} for m in models}

        def _load_stats(model_name, tgt_dict):
            path = os.path.join(zarr_dir, f"output_data_{model_name}.zarr")
            try:
                root = zarr.open_group(path, mode="r")
            except Exception:
                return
            if "piControl_stats" not in root:
                return
            for grp in ("x", "y", "y_no_ekman", "ekman"):
                if grp not in root["piControl_stats"]:
                    continue
                for key in root["piControl_stats"][grp]:
                    mu = zarr.open_array(os.path.join(path, "piControl_stats", grp, key, "mean"))
                    sd = zarr.open_array(os.path.join(path, "piControl_stats", grp, key, "std"))
                    tgt_dict[grp][key] = (np.asarray(mu), np.asarray(sd))

        _load_stats("MPI-ESM1-2-HR", self.global_norm_stats)
        for m in models:
            _load_stats(m, self.norm_stats[m])

        # ----------------------------------------------------------------
        # build self.samples  (only check presence of chosen target_group)
        # ----------------------------------------------------------------
        self.samples = []
        for model in models:
            path = os.path.join(zarr_dir, f"output_data_{model}.zarr")
            try:
                root = zarr.open_group(path, mode="r")
            except Exception:
                continue


            for scn in scenarios:

                
                if scn not in root:
                    continue
                if "x" not in root[scn] or self.target_group not in root[scn]:
                    continue
                xg = root[scn]["x"]
                yg = root[scn][self.target_group]

                if any(f"{v}_{lpf}" not in xg for v in x_vars):
                    continue
                if lpf not in yg:
                    continue

                T = min([xg[f"{v}_{lpf}"].shape[0] for v in x_vars] + [yg[lpf].shape[0]])
                for t in range(self.num_skip, T, self.stride):
                    self.samples.append({"model": model, "scenario": scn, "time_idx": t})

    # --------------------------------------------------------------------
    #  EVERYTHING BELOW IS YOUR ORIGINAL CODE,
    #  with ONLY **2** identifiers switched to self.target_group.
    # --------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        model    = sample["model"]
        scenario = sample["scenario"]
        time_idx = sample["time_idx"]

        store_path = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        # ------------- X -------------------------------------------------
        x_data_list = []
        for var in self.x_vars:
            key = f"{var}_{self.lpf}"
            x_arr_path = os.path.join(store_path, scenario, "x", key)
            x_arr  = zarr.open_array(x_arr_path, mode="r")
            x_data = np.asarray(x_arr[time_idx]).astype(np.float32)

            if key in self.global_norm_stats["x"]:
                global_norm_mean, global_norm_std = self.global_norm_stats["x"][key]

            #print(np.nanmean(x_data),model)
            if model in self.norm_stats and key in self.norm_stats[model]["x"]:
                norm_mean, norm_std = self.norm_stats[model]["x"][f"{var}_raw"]
                #print(model)
                #print(np.nanmean(norm_std) , np.nanmean(norm_mean))
                #if 'pbo' in var:
                #    x_data = (x_data - norm_mean + np.random.normal(loc=0.0, scale=150.0, size=x_data.shape)  ) / (norm_std + 1e-4)
                #else: 
                #if model == "CMCC-ESM2":
                #    x_data = x_data / 1000
                #    norm_mean = norm_mean / 1000
                if self.noise is not None:
                    x_data = (x_data - norm_mean + np.random.normal(loc=0.0, scale = self.noise, size=x_data.shape)) #/ (np.nanmean(global_norm_std) + 1e-4)
                else:
                    x_data = (x_data - norm_mean) #/ (np.nanmean(global_norm_std) + 1e-4)
                
                #print(np.nanmean(global_norm_std))
                #print(1/0)




                

            if model == "MRI-ESM2-0":
                x_data[50, 94] = 0

            x_data = np.nan_to_num(x_data)
            x_data_list.append(x_data)

        x_data_concat = np.stack(x_data_list, axis=0)

        # ------------- Y‑like target  -----------------------------------
        y_arr_path = os.path.join(           # ← CHANGED: group name
            store_path, scenario, self.target_group, self.lpf
        )
        y_arr  = zarr.open_array(y_arr_path, mode="r")
        y_data = np.asarray(y_arr[time_idx]).astype(np.float32)

        if (
            model in self.norm_stats
            and self.lpf in self.norm_stats[model][self.target_group]   # ← CHANGED
        ):
            norm_mean, norm_std = self.norm_stats[model][self.target_group][self.lpf]
            max_idx = np.argmax(y_data, axis=0)

            ref = norm_mean[np.argmax(norm_mean, axis=0), np.arange(norm_mean.shape[1])]
            #ref = norm_mean[np.argmax(norm_mean[:, 23]), 23]


            y_data = (y_data - ref) / 1161159294
            #y_data = (y_data) #- norm_mean[0, 23]) / 1161159294
            #print(self.target_group, "std:", norm_std[np.argmax(norm_mean[:, 23]), 23] / 1161159294)

        # optional latitude selection (unchanged) ------------------------
        if self.selected_lats is not None:
            lat_arr  = np.arange(-30, 70 + 2.5, 2.5)
            profiles, idx_list = [], []
            for sel in self.selected_lats:
                idx_lat = np.abs(lat_arr - sel).argmin()
                profiles.append(y_data[:, idx_lat])
                idx_list.append(idx_lat)
            y_data = np.stack(profiles, axis=1)

        if self.transform:
            x_data_concat = self.transform(x_data_concat)

        if self.output_type == "max":
            y_data = y_data[max_idx[idx_list], np.arange(y_data.shape[1])]
            #y_data = y_data[1, np.arange(y_data.shape[1])]

        y_data = np.nan_to_num(y_data)
        return torch.from_numpy(x_data_concat), torch.from_numpy(y_data)







import os, zarr, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


class PreprocessedCMIP6DatasetInMem(Dataset):
    """
    • Loads every usable snapshot into RAM in parallel (thread pool).
    • Keeps **identical output** to the original dataset:  
        X  → (C, H, W)  torch.float32  
        y  → (n_lat,)   or (1,) if single-latitude & output_type='max'
    • All heavy work lives in `_process_model`, mirroring your example.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        zarr_dir: str,
        models: List[str],
        x_vars: List[str],
        scenarios: List[str],
        *,
        target_group: str = "y",
        output_type: str = "max",
        selected_lats: List[float] | None = None,
        lpf: str = "raw",
        num_skip: int = 12,
        stride: int = 6,
        transform=None,
        workers: int = 8,
        noise = None
    ):
        assert target_group in {"y", "y_no_ekman", "ekman"}

        # ---- store args ------------------------------------------------ #
        self.zarr_dir, self.models, self.x_vars = zarr_dir, models, x_vars
        self.scenarios, self.output_type = scenarios, output_type
        self.selected_lats, self.lpf = selected_lats, lpf
        self.num_skip, self.stride = num_skip, stride
        self.target_group, self.transform = target_group, transform

        # ---- π-Control statistics (unchanged logic) ------------------- #
        self.global_norm_stats = {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}}
        self.norm_stats = {m: {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}}
                           for m in models}

        def _load_stats(model_name, tgt_dict):
            path = os.path.join(zarr_dir, f"output_data_{model_name}.zarr")
            try:
                root = zarr.open_group(path, mode="r")
            except Exception:
                return
            if "piControl_stats" not in root:
                return
            for grp in ("x", "y", "y_no_ekman", "ekman"):
                if grp not in root["piControl_stats"]:
                    continue
                for key in root["piControl_stats"][grp]:
                    mu = zarr.open_array(os.path.join(path, "piControl_stats",
                                                     grp, key, "mean"))
                    sd = zarr.open_array(os.path.join(path, "piControl_stats",
                                                     grp, key, "std"))
                    tgt_dict[grp][key] = (np.asarray(mu), np.asarray(sd))

        _load_stats("GFDL-ESM4", self.global_norm_stats)
        for m in models:
            _load_stats(m, self.norm_stats[m])

        # ---- parallel preload via _process_model ---------------------- #
        self.X_cache: List[torch.Tensor] = []
        self.y_cache: List[torch.Tensor] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self._process_model, m, scenarios) for m in models]
            for fut in as_completed(futs):
                X_part, y_part = fut.result()
                self.X_cache.extend(X_part)
                self.y_cache.extend(y_part)


    # ------------------------------------------------------------------ #
    # 1️⃣  Stats loader – keep ONLY the mean like in your template
    # ------------------------------------------------------------------ #
    def _load_stats(self, models):
        """
        After this call
            self.norm_stats[model]["x"][f"{var}_raw"]    ->  np.ndarray  (mean)
            self.norm_stats[model][target_group]         ->  np.ndarray  (mean)
        """
        for m in models:
            path = os.path.join(self.zarr_dir, f"output_data_{m}.zarr")
            if not os.path.exists(path):
                continue
            root = zarr.open_group(path, mode="r")
            if "piControl_stats" not in root:
                continue

            # --- X means -------------------------------------------------- #
            if "x" in root["piControl_stats"]:
                for key in root["piControl_stats"]["x"]:
                    mu = zarr.open_array(
                        os.path.join(path, "piControl_stats", "x", key, "mean"))
                    self.norm_stats[m][key] = np.asarray(mu, dtype=np.float32)

            # --- target-group means --------------------------------------- #
            if self.target_group in root["piControl_stats"]:
                mu_t = zarr.open_array(os.path.join(
                    path, "piControl_stats", self.target_group, self.lpf, "mean"))
                self.norm_stats[m][self.target_group] = np.asarray(
                    mu_t, dtype=np.float32
                )
    # ------------------------------------------------------------------ #
    # 2️⃣  Heavy loader – μ is now an array, so the slice works
    # ------------------------------------------------------------------ #
    def _process_model(self, model: str, scenarios: List[str]):
        X_out, y_out = [], []
        store = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        try:
            root = zarr.open_group(store, mode="r")
        except Exception:
            return X_out, y_out

        for scn in scenarios:
            if scn not in root:
                continue
            if "x" not in root[scn] or self.target_group not in root[scn]:
                continue

            # ---------- Y (all timesteps) -------------------------------- #
            y_arr = zarr.open_array(
                os.path.join(store, scn, self.target_group, self.lpf), mode="r")
            y_all = y_arr[:].astype(np.float32)

            # ----- per-model centring with *mean array* ------------------ #
            if self.target_group in self.norm_stats[model]:
                mu, std = self.norm_stats[model][self.target_group]['raw']


                #ref = mu[np.argmax(mu[:, 23]), 23]

                # mu.shape = (21, 41)
                # i wanna ref shape to be (41,)

                ref = mu[np.argmax(mu, axis=0), np.arange(mu.shape[1])]

                #ref = norm_mean[np.argmax(norm_mean, axis=0), np.arange(norm_mean.shape[1])]

                #print(mu.shape, ref.shape)
                y_all = (y_all - ref) / 1_161_159_294.0

                
            y_all = np.nan_to_num(y_all)

            # latitude sub-selection (optional) --------------------------- #
            if self.selected_lats is not None:
                lat_axis = np.arange(-30, 70 + 2.5, 2.5)
                idx_lat = [np.abs(lat_axis - lat).argmin()
                        for lat in self.selected_lats]
                y_all = y_all[:, :, idx_lat]                              # (T, depth, n_sel)

            # ---------- X variables -------------------------------------- #
            X_vars = []
            for var in self.x_vars:
                key = f"{var}_{self.lpf}"
                if key not in root[scn]["x"]:
                    break
                x_arr = zarr.open_array(os.path.join(store, scn, "x", key), mode="r")
                x_all = x_arr[:].astype(np.float32)

                mu_key = f"{var}_raw"
                if mu_key in self.norm_stats[model]:
                    x_all -= self.norm_stats[model][mu_key]               # subtract mean
                if model == "MRI-ESM2-0":
                    x_all[:, 50, 94] = 0                                  # fix bad cell
                X_vars.append(np.nan_to_num(x_all))
            else:
                # stack (T, C, H, W) and iterate over timesteps ---------- #
                x_stack = np.stack(X_vars, axis=1)
                #T = x_stack.shape[0]

                T = min(x_stack.shape[0], y_all.shape[0])


                for t in range(self.num_skip, T, self.stride):
                    # -------- snapshot X -------------------------------- #
                    X_out.append(torch.from_numpy(
                        np.ascontiguousarray(x_stack[t])).float())

                    # -------- snapshot y -------------------------------- #
                    y_snap = y_all[t]                                     # (depth,…)
                    if self.output_type == "max":
                        y_snap = y_snap.max(axis=0)
                        if y_snap.ndim == 0:
                            y_snap = y_snap[None]                         # (1,)
                    y_out.append(torch.from_numpy(
                        np.ascontiguousarray(y_snap)).float())

        return X_out, y_out
    # ------------------- PyTorch interface ---------------------------- #
    def __len__(self) -> int:
        return len(self.X_cache)

    def __getitem__(self, idx: int):
        x = self.X_cache[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y_cache[idx]



## cmip6_dataset_inmem_fast.py  – X → (T, C, H, W);  y → (1,)
import os
import numpy as np
import zarr
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict


class PreprocessedCMIP6SeqDatasetInMem(Dataset):
    """
    In-memory sequence dataset (fast path).

    X window  : (seq_len, C, H, W)
    y scalar  : (1,)  when output_type='max' and a single latitude is selected
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        zarr_dir: str,
        models: List[str],
        x_vars: List[str],
        scenarios: List[str],
        *,
        seq_len: int = 12,
        stride: int | None = None,
        num_skip: int = 12,
        target_group: str = "y",
        output_type: str = "max",
        selected_lats: List[float] | None = None,
        lpf: str = "raw",
        workers: int = 4,
    ):
        assert target_group in {"y", "y_no_ekman", "ekman"}
        self.seq_len, self.stride = seq_len, stride or seq_len // 2
        self.num_skip = num_skip
        self.target_group, self.output_type = target_group, output_type
        self.selected_lats, self.lpf = selected_lats, lpf
        self.x_vars, self.zarr_dir = x_vars, zarr_dir

        # ------- π-Control means --------------------------------------
        self.norm_stats: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
        self._load_stats(models)

        # ------- preload in parallel (thread pool) -------------------
        self.X_cache, self.y_cache = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self._process_model, m, scenarios) for m in models]
            for f in as_completed(futs):
                Xp, yp = f.result()
                self.X_cache.extend(Xp)
                self.y_cache.extend(yp)

        # to tensors; ensure contiguous writable numpy buffers
        self.X_cache = [torch.from_numpy(np.ascontiguousarray(x)).float()
                        for x in self.X_cache]
        self.y_cache = [torch.from_numpy(
                            np.ascontiguousarray(np.array(y, np.float32, ndmin=1))
                        ).float()
                        for y in self.y_cache]

    # ------------------------------------------------------------------ #
    def _load_stats(self, models):
        for m in models:
            path = os.path.join(self.zarr_dir, f"output_data_{m}.zarr")
            if not os.path.exists(path):
                continue
            root = zarr.open_group(path, mode="r")
            if "piControl_stats" not in root:
                continue

            if "x" in root["piControl_stats"]:
                for key in root["piControl_stats"]["x"]:
                    mu = zarr.open_array(
                        os.path.join(path, "piControl_stats", "x", key, "mean"))
                    self.norm_stats[m][key] = np.asarray(mu)

            if self.target_group in root["piControl_stats"]:
                mu_t = zarr.open_array(os.path.join(
                    path, "piControl_stats", self.target_group, self.lpf, "mean"))
                self.norm_stats[m][self.target_group] = np.asarray(mu_t)

    # ------------------------------------------------------------------ #
    def _process_model(self, model: str, scenarios: List[str]):
        X_out, y_out = [], []
        store = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        try:
            root = zarr.open_group(store, mode="r")
        except Exception:
            return X_out, y_out

        for scn in scenarios:
            if scn not in root: continue
            if "x" not in root[scn] or self.target_group not in root[scn]:
                continue

            # ---------- full Y series ---------------------------------
            y_arr = zarr.open_array(os.path.join(
                store, scn, self.target_group, self.lpf), mode="r")
            y_all = y_arr[:].astype(np.float32)

            if self.target_group in self.norm_stats[model]:
                mu = self.norm_stats[model][self.target_group]
                ref = mu[np.argmax(mu[:, 23]), 23]
                y_all = (y_all - ref) / 1161159294
            y_all = np.nan_to_num(y_all)

            if self.selected_lats is not None:
                lat_axis = np.arange(-30, 70 + 2.5, 2.5)
                idx = [np.abs(lat_axis - lat).argmin() for lat in self.selected_lats]
                y_all = y_all[:, :, idx]              # (T, depth, n_sel)

            y_aligned = y_all[self.num_skip + self.seq_len - 1:]
            y_windows = y_aligned[::self.stride]      # (N, depth, n_sel)

            # ---------- X variables -----------------------------------
            X_vars = []
            for var in self.x_vars:
                key = f"{var}_{self.lpf}"
                if key not in root[scn]["x"]:
                    break
                x_arr = zarr.open_array(os.path.join(store, scn, "x", key), mode="r")
                x_all = x_arr[:].astype(np.float32)

                if f"{var}_raw" in self.norm_stats[model]:
                    x_all -= self.norm_stats[model][f"{var}_raw"]
                X_vars.append(np.nan_to_num(x_all))
            else:
                x_stack = np.stack(X_vars, axis=1)    # (T, C, H, W)

                win = sliding_window_view(
                    x_stack, window_shape=(self.seq_len,), axis=0
                )[self.num_skip:]                    # (N′, C, H, W, seq)
                win = win[::self.stride]             # (N, C, H, W, seq)
                # move seq_len to front → (N, seq, C, H, W)
                X_windows = np.moveaxis(win, -1, 1)

                # ---------- y: max over depth ---------------------------
                if self.output_type == "max":
                    y_max = y_windows.max(axis=1)     # (N, n_sel)
                    if y_max.shape[1] == 1:
                        y_max = y_max[:, 0:1]         # (N, 1)
                    y_windows = y_max

                X_out.extend(list(X_windows))
                y_out.extend(y_windows.astype(np.float32))

        return X_out, y_out

    # -------------------- PyTorch API ---------------------------------
    def __len__(self):
        return len(self.X_cache)

    def __getitem__(self, idx):
        return self.X_cache[idx], self.y_cache[idx]



class PreprocessedCMIP6SeqDataset(Dataset):
    """
    Return a *sequence* of length `seq_len` ending at `time_idx`.
    All timesteps in a sample belong to the same model / scenario,
    so nothing is ever mixed across experiments.
    """
    def __init__(
        self,
        zarr_dir: str,
        models: list[str],
        x_vars: list[str],
        scenarios: list[str],
        *,
        seq_len: int = 12,          # ←  how many steps the LSTM sees
        stride: int | None = None,  # ←  spacing between *targets* (defaults to seq_len//2)
        num_skip: int = 12,         # ←  spin-up months to ignore at start of run
        target_group: str = "y",    # "y" | "y_no_ekman" | "ekman"
        output_type: str = "max",
        selected_lats: list[float] | None = None,
        transform=None,
        lpf: str = "raw",
        local: bool = False,
    ):
        assert target_group in {"y", "y_no_ekman", "ekman"}
        self.zarr_dir, self.models, self.x_vars = zarr_dir, models, x_vars
        self.scenarios       = scenarios
        self.seq_len         = seq_len
        self.stride          = stride or seq_len // 2
        self.num_skip        = num_skip
        self.target_group    = target_group
        self.output_type     = output_type
        self.selected_lats   = selected_lats
        self.transform       = transform
        self.lpf, self.local = lpf, local

        # ----------------------------------------------------------------
        #  normalisation statistics  (unchanged – just folded into a fn)
        # ----------------------------------------------------------------
        self.global_norm_stats = {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}}
        self.norm_stats        = {
            m: {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}} for m in models
        }
        self._load_stats()

        # ----------------------------------------------------------------
        #  Build (model, scenario, start_t) triplets for which we have a
        #  *complete* window  [start_t,  start_t+seq_len-1]
        # ----------------------------------------------------------------
        self.samples = []
        for model in models:
            store_path = os.path.join(zarr_dir, f"output_data_{model}.zarr")
            try:
                root = zarr.open_group(store_path, mode="r")
            except Exception:
                continue

            for scn in scenarios:
                if scn not in root:
                    continue
                if "x" not in root[scn] or self.target_group not in root[scn]:
                    continue
                xg = root[scn]["x"]
                yg = root[scn][self.target_group]

                # all requested variables + target must exist for this LPF
                if any(f"{v}_{lpf}" not in xg for v in x_vars):
                    continue
                if lpf not in yg:
                    continue

                # shortest run length among all vars / target
                T = min([xg[f"{v}_{lpf}"].shape[0] for v in x_vars] +
                        [yg[lpf].shape[0]])

                # valid window starts
                t0_min = self.num_skip
                t0_max = T - seq_len          # inclusive
                for t0 in range(t0_min, t0_max + 1, self.stride):
                    self.samples.append(
                        {"model": model, "scenario": scn, "start_idx": t0}
                    )

    # --------------------------------------------------------------------
    #                    ───  private helpers  ───
    # --------------------------------------------------------------------
    def _load_stats(self):
        def read_stats(model_name, target_dict):
            path = os.path.join(self.zarr_dir, f"output_data_{model_name}.zarr")
            if not os.path.exists(path):
                return
            root = zarr.open_group(path, mode="r")
            if "piControl_stats" not in root:
                return
            for grp in ("x", "y", "y_no_ekman", "ekman"):
                if grp not in root["piControl_stats"]:
                    continue
                for key in root["piControl_stats"][grp]:
                    mu = zarr.open_array(os.path.join(path, "piControl_stats",
                                                     grp, key, "mean"))
                    sd = zarr.open_array(os.path.join(path, "piControl_stats",
                                                     grp, key, "std"))
                    target_dict[grp][key] = (np.asarray(mu), np.asarray(sd))

        # global stats → always try GFDL-ESM4 first
        read_stats("GFDL-ESM4", self.global_norm_stats)
        for m in self.models:
            read_stats(m, self.norm_stats[m])

    # --------------------------------------------------------------------
    #                    ───  PyTorch Dataset API  ───
    # --------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------------------
    #  PyTorch-style accessor with ORIGINAL normalisation rules
    # --------------------------------------------------------------------
    def __getitem__(self, idx):
        sample    = self.samples[idx]
        model     = sample["model"]
        scenario  = sample["scenario"]
        start_idx = sample["start_idx"]              # first timestep in window
        end_idx   = start_idx + self.seq_len         # non-inclusive

        store_path = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        # ---------- X  (T, C, lat, lon)  --------------------------------
        x_seq = []
        for t in range(start_idx, end_idx):
            x_t = []
            for var in self.x_vars:
                key     = f"{var}_{self.lpf}"
                x_arr   = zarr.open_array(os.path.join(store_path, scenario,
                                                      "x", key), mode="r")
                x_data  = np.asarray(x_arr[t]).astype(np.float32)

                # *** ORIGINAL per-model mean subtraction (no /std) *******
                if model in self.norm_stats and key in self.norm_stats[model]["x"]:
                    mu, _ = self.norm_stats[model]["x"][f"{var}_raw"]
                    x_data = (x_data - mu)

                # same MRI bad-point fix as before
                #if model == "MRI-ESM2-0":
                #    x_data[50, 94] = 0

                x_t.append(np.nan_to_num(x_data))
            x_seq.append(np.stack(x_t, axis=0))      # (C, lat, lon)

        x_seq = np.stack(x_seq, axis=0)              # (T, C, lat, lon)
        if self.transform:
            x_seq = self.transform(x_seq)

        # ---------- Y / target  -----------------------------------------
        y_arr  = zarr.open_array(
            os.path.join(store_path, scenario, self.target_group, self.lpf), mode="r"
        )
        y_data = np.asarray(y_arr[end_idx - 1]).astype(np.float32)

        # *** ORIGINAL custom scaling for target **************************
        if (
            model in self.norm_stats
            and self.lpf in self.norm_stats[model][self.target_group]
        ):
            norm_mean, norm_std = self.norm_stats[model][self.target_group][self.lpf]

            # original behaviour: pick the (depth-max, lat=23) cell of mean
            ref_val = norm_mean[np.argmax(norm_mean[:, 23]), 23]
            y_data  = (y_data - ref_val) / 1161159294

            # -- If you later want the simple mean-std scheme, just swap: --
            # y_data = (y_data - norm_mean) / (norm_std + 1e-8)

        # latitude slice (unchanged)
        if self.selected_lats is not None:
            lat_arr  = np.arange(-30, 70 + 2.5, 2.5)
            profiles = []
            for sel in self.selected_lats:
                idx_lat = np.abs(lat_arr - sel).argmin()
                profiles.append(y_data[:, idx_lat])
            y_data = np.stack(profiles, axis=1)      # (depth, n_sel)

        if self.output_type == "max":
            max_idx = np.argmax(y_data, axis=0)
            y_data  = y_data[max_idx, np.arange(y_data.shape[1])]

        y_data = np.nan_to_num(y_data)
        return (
            torch.from_numpy(x_seq).float(),         # (T, C, lat, lon)
            torch.from_numpy(y_data).float(),        # target at t = end_idx-1
        )














import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import torch
import zarr
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset


class PreprocessedCMIP6SeqDatasetInMem_past_future(Dataset):
    """
    Sequence dataset that keeps **symmetric** windows fully in memory.

    The input window now contains `(seq_len // 2 - 1)` frames *before* the target
    time step **and** the same number of frames *after* it, together with the
    target frame itself in the centre.  This allows the model to see equal
    context from the past and the future.  All preprocessing / normalisation of
    the individual samples stays exactly the same as before.

    X window  : `(seq_len, C, H, W)` — temporal dimension is the **first**.
    y scalar  : `(1,)`  (if `output_type='max'` and only one latitude)
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        zarr_dir: str,
        models: List[str],
        x_vars: List[str],
        scenarios: List[str],
        *,
        seq_len: int = 11,  # must be odd → symmetric window around y
        stride: int | None = None,
        num_skip: int = 12,
        target_group: str = "y",
        output_type: str = "max",
        selected_lats: List[float] | None = None,
        lpf: str = "raw",
        workers: int = 4,
    ):
        assert (seq_len % 2) == 1, "`seq_len` must be an odd number so that y is centred"
        assert target_group in {"y", "y_no_ekman", "ekman"}
        self.seq_len, self.stride = seq_len, stride or seq_len // 2
        self.num_skip = num_skip
        self.target_group, self.output_type = target_group, output_type
        self.selected_lats, self.lpf = selected_lats, lpf
        self.x_vars, self.zarr_dir = x_vars, zarr_dir

        # ------- π‑Control means -------------------------------------
        self.norm_stats: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
        self._load_stats(models)

        # ------- preload in several threads --------------------------
        self.X_cache, self.y_cache = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self._process_model, m, scenarios) for m in models]
            for f in as_completed(futs):
                Xp, yp = f.result()
                self.X_cache.extend(Xp)
                self.y_cache.extend(yp)

        # to tensors; make arrays contiguous & writable
        self.X_cache = [torch.from_numpy(np.ascontiguousarray(x)).float() for x in self.X_cache]
        self.y_cache = [
            torch.from_numpy(np.ascontiguousarray(np.array(y, np.float32, ndmin=1))).float()
            for y in self.y_cache
        ]

    # ------------------------------------------------------------------ #
    def _load_stats(self, models):
        for m in models:
            path = os.path.join(self.zarr_dir, f"output_data_{m}.zarr")
            if not os.path.exists(path):
                continue
            root = zarr.open_group(path, mode="r")
            if "piControl_stats" not in root:
                continue

            if "x" in root["piControl_stats"]:
                for key in root["piControl_stats"]["x"]:
                    mu = zarr.open_array(os.path.join(path, "piControl_stats", "x", key, "mean"))
                    self.norm_stats[m][key] = np.asarray(mu)

            if self.target_group in root["piControl_stats"]:
                mu_t = zarr.open_array(
                    os.path.join(path, "piControl_stats", self.target_group, self.lpf, "mean")
                )
                self.norm_stats[m][self.target_group] = np.asarray(mu_t)

    # ------------------------------------------------------------------ #
    def _process_model(self, model: str, scenarios: List[str]):
        X_out, y_out = [], []
        store = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        try:
            root = zarr.open_group(store, mode="r")
        except Exception:
            return X_out, y_out

        half = self.seq_len // 2  # number of frames on *either* side of y

        for scn in scenarios:
            if scn not in root:
                continue
            if "x" not in root[scn] or self.target_group not in root[scn]:
                continue

            # ---------- Y (entire time‑series) -----------------------
            y_arr = zarr.open_array(os.path.join(store, scn, self.target_group, self.lpf), mode="r")
            y_all = y_arr[:].astype(np.float32)

            if self.target_group in self.norm_stats[model]:
                mu = self.norm_stats[model][self.target_group]
                ref = mu[np.argmax(mu[:, 23]), 23]
                y_all = (y_all - ref) / 1161159294  # normalisation identical to the old version
            y_all = np.nan_to_num(y_all)

            if self.selected_lats is not None:
                lat_axis = np.arange(-30, 70 + 2.5, 2.5)
                idx = [np.abs(lat_axis - lat).argmin() for lat in self.selected_lats]
                y_all = y_all[:, :, idx]  # (T, depth, n_sel)

            # ---------- X variables ---------------------------------
            X_vars = []
            for var in self.x_vars:
                #key = f"{var}_{self.lpf}"
                key = f"{var}_raw"
                if key not in root[scn]["x"]:
                    break
                x_arr = zarr.open_array(os.path.join(store, scn, "x", key), mode="r")
                x_all = x_arr[:].astype(np.float32)

                if f"{var}_raw" in self.norm_stats[model]:
                    x_all -= self.norm_stats[model][f"{var}_raw"]
                X_vars.append(np.nan_to_num(x_all))
            else:
                x_stack = np.stack(X_vars, axis=1)  # (T, C, H, W)

                # build sliding windows of full length `seq_len`
                win = sliding_window_view(x_stack, window_shape=(self.seq_len,), axis=0)
                # windows start at every time step from 0 … T - seq_len
                starts = np.arange(win.shape[0])

                # skip spin‑up period
                if self.num_skip:
                    win = win[self.num_skip:]
                    starts = starts[self.num_skip:]

                # apply striding
                win = win[::self.stride]
                starts = starts[::self.stride]

                # move seq dim to position 1 → (N, seq, C, H, W)
                X_windows = np.moveaxis(win, -1, 1)

                # ---------- y aligned with centre ------------------
                centre_idx = starts + half  # index of y for each window
                y_windows = y_all[centre_idx]  # (N, depth, n_sel)

                # reduce over depth if required
                if self.output_type == "max":
                    y_max = y_windows.max(axis=1)  # (N, n_sel)
                    if y_max.shape[1] == 1:
                        y_max = y_max[:, 0:1]
                    y_windows = y_max

                X_out.extend(list(X_windows))
                y_out.extend(y_windows.astype(np.float32))

        return X_out, y_out

    # -------------------- PyTorch API -------------------------------
    def __len__(self):
        return len(self.X_cache)

    def __getitem__(self, idx):
        return self.X_cache[idx], self.y_cache[idx]










if __name__ == "__main__":

    dataset = PreprocessedCMIP6DatasetInMem(
        zarr_dir     = "/home/am334/link_am334/moc_mmodel/monthly_1deg_grid_with_coasts_EK_minus",
        models       = [ "ACCESS-ESM1-5"], #"ACCESS-CM2", "GFDL-ESM4", "GFDL-CM4", "MRI-ESM2-0", "INM-CM4-8", "FGOALS-g3", "MIROC6", "CanESM5", "GISS-E2-1-G", "NorESM2-LM", "NorESM2-MM", "HadGEM3-GC31-LL", "UKESM1-1-LL", "CMCC-ESM2", "HadGEM3-GC31-MM", "MPI-ESM1-2-HR" ],
        x_vars       = ["tos"],
        scenarios    = ["piControl", "historical"],
        output_type="max",
        target_group = "y",
        lpf          = "LPF24",

        selected_lats=[26.5],
    )


    print("Number of samples:", len(dataset))
    x_tensor, y_tensor = dataset[0]
    print(y_tensor)
    print("x_tensor shape:", x_tensor.shape)  # Expected shape: [num_vars, ...]
    print("y_tensor shape:", y_tensor.shape)





    '''
    dataset = PreprocessedCMIP6Dataset(
        zarr_dir="/home/am334/link_am334/moc_mmodel/monthly_1deg_grid_with_coasts_EK_minus/",
        models=['ACCESS-ESM1-5'],  # other models can be included too
        x_vars=['tauu'],
        scenarios=["piControl"],
        target_group="y_no_ekman",
        output_type="max",
        selected_lats=[26.5],
        lpf="LPF24"
    )

    print("Number of samples:", len(dataset))
    x_tensor, y_tensor = dataset[0]
    print(y_tensor)
    print("x_tensor shape:", x_tensor.shape)  # Expected shape: [num_vars, ...]
    print("y_tensor shape:", y_tensor.shape)

'''


'''
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, get_worker_info
import zarr

class PreprocessedCMIP6Dataset(Dataset):
    def __init__(self, zarr_dir, models, x_vars, scenarios, output_type="profile", 
                 selected_lats=None, transform=None, lpf="raw", local = False):
        """
        Args:
            zarr_dir (str): Directory containing the Zarr stores (files named "output_data_{model}.zarr").
            models (list): List of model names.
            x_vars (list): List of x-variable names (e.g. ['zos', 'tos']).
            scenarios (list): List of scenarios (e.g. ["ssp245", "piControl"]).
            output_type (str): "profile" or "max".
            selected_lats (list, optional): List of latitudes for selecting y-variable data.
            transform (callable, optional): Transformation to apply to data.
            lpf (str): Filter version key (e.g. "raw", "LPF24", "LPF120").
        """
        self.zarr_dir = zarr_dir
        self.models = models
        self.x_vars = x_vars
        self.scenarios = scenarios
        self.output_type = output_type
        self.selected_lats = selected_lats
        self.transform = transform
        self.lpf = lpf
        self.local = local

        # Set number of initial steps to skip based on lpf.
        self.num_skip = 0
        self.stride = 0
        if lpf == "LPF120":
            self.stride = 12
        if lpf == "LPF24":
            self.stride = 12

        
        if self.stride == 0:
            self.stride = 1

        self.samples = []
        self.norm_stats = {}

        # Iterate over models and scenarios to build sample list and preload normalization stats.
        for model in models:
            file_path = os.path.join(zarr_dir, f"output_data_{model}.zarr")
            try:
                ds_group = zarr.open_group(file_path, mode="r")
            except Exception as e:
                print(f"Could not open Zarr for model {model}: {e}")
                continue

            # Preload normalization stats if available.
            if "piControl_stats" in ds_group:
                stats_grp = ds_group["piControl_stats"]
                self.norm_stats[model] = {"x": {}, "y": {}}
                if "x" in stats_grp:
                    for key in stats_grp["x"].keys():
                        mean_path = os.path.join(file_path, "piControl_stats", "x", key, "mean")
                        std_path  = os.path.join(file_path, "piControl_stats", "x", key, "std")
                        try:
                            mean_arr = zarr.open_array(mean_path, mode="r")
                            std_arr  = zarr.open_array(std_path, mode="r")
                            self.norm_stats[model]["x"][key] = (np.asarray(mean_arr), np.asarray(std_arr))
                        except Exception as e:
                            print(f"Error loading x stats for model {model}, key {key}: {e}")
                if "y" in stats_grp:
                    for key in stats_grp["y"].keys():
                        mean_path = os.path.join(file_path, "piControl_stats", "y", key, "mean")
                        std_path  = os.path.join(file_path, "piControl_stats", "y", key, "std")
                        try:
                            mean_arr = zarr.open_array(mean_path, mode="r")
                            std_arr  = zarr.open_array(std_path, mode="r")
                            self.norm_stats[model]["y"][key] = (np.asarray(mean_arr), np.asarray(std_arr))
                        except Exception as e:
                            print(f"Error loading y stats for model {model}, key {key}: {e}")

            # Build sample list for each scenario.
            for scenario in scenarios:
                if scenario not in ds_group:
                    continue
                scenario_grp = ds_group[scenario]
                if "x" not in scenario_grp or "y" not in scenario_grp:
                    continue
                x_grp = scenario_grp["x"]
                y_grp = scenario_grp["y"]

                # Verify that all requested x_vars exist for the given lpf.
                missing = False
                for var in self.x_vars:
                    key = f"{var}_{self.lpf}"
                    if key not in x_grp:
                        print(f"Warning: {key} not found in model {model}, scenario {scenario}")
                        missing = True
                        break
                if missing:
                    continue

                # Check that y variable with key lpf exists.
                if self.lpf not in y_grp:
                    continue

                # Determine the common time dimension T as the minimum available time among all variables.
                T_x = min([x_grp[f"{var}_{self.lpf}"].shape[0] for var in self.x_vars])
                T_y = y_grp[self.lpf].shape[0]
                T = min(T_x, T_y)


                if self.lpf == "raw":
                    for t in range(self.num_skip, T):
                        self.samples.append({
                            "model": model,
                            "scenario": scenario,
                            "time_idx": t
                        })

                else:


                # Create a sample for each time index (using stride and num_skip).
                    for t in range(self.num_skip, T, self.stride):
                        self.samples.append({
                            "model": model,
                            "scenario": scenario,
                            "time_idx": t
                        })

            ds_group = None

    def __len__(self):
        return len(self.samples)

    def _get_store(self, model):
        file_path = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")
        return xr.open_zarr(file_path)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model = sample["model"]
        scenario = sample["scenario"]
        time_idx = sample["time_idx"]

        store_path = os.path.join(self.zarr_dir, f"output_data_{model}.zarr")

        # Load each x variable and collect into a list.
        x_data_list = []
        for var in self.x_vars:
            key = f"{var}_{self.lpf}"
            x_arr_path = os.path.join(store_path, scenario, "x", key)
            try:
                x_arr = zarr.open_array(x_arr_path, mode="r")
            except Exception as e:
                raise RuntimeError(f"Error opening x array for model {model}, scenario {scenario}, var {var}: {e}")
            x_data = np.asarray(x_arr[time_idx]).astype(np.float32)
            #x_data = x_data - np.nanmean(x_data)
            # Normalize using preloaded stats if available.
            if model in self.norm_stats and key in self.norm_stats[model]["x"]:
                norm_mean, norm_std = self.norm_stats[model]["x"][key]
                x_data = (x_data - norm_mean) / norm_std  #/ norm_std
                #x_data = x_data


            if self.local == True:
                lat_list_x = np.arange(-60.5,83.5,1)
                for sel in self.selected_lats:
                    idx_lat_x = np.abs(lat_list_x - sel).argmin()
                x_data = x_data[ idx_lat_x, : ]

            #if var == 'zos':
            #    x_data = x_data - np.nanmean(x_data)
            x_data = np.nan_to_num(x_data)
            x_data_list.append(x_data)
        # Stack the variables along a new dimension (channel dimension).
        x_data_concat = np.stack(x_data_list, axis=0)

        # Load y data.
        y_arr_path = os.path.join(store_path, scenario, "y", self.lpf)
        try:
            y_arr = zarr.open_array(y_arr_path, mode="r")
        except Exception as e:
            raise RuntimeError(f"Error opening y array for model {model}, scenario {scenario}: {e}")
        y_data = np.asarray(y_arr[time_idx]).astype(np.float32)
        if model in self.norm_stats and self.lpf in self.norm_stats[model]["y"]:
            norm_mean, norm_std = self.norm_stats[model]["y"][self.lpf]
            max_idx = np.argmax(y_data, axis = 0)
            #y_data = y_data/np.nanm(norm_mean) mean = 18.4 Sv, 
            y_data = (y_data - norm_mean) / norm_std #/ norm_mean#norm_std

        #y_data = np.nan_to_num(y_data)


        



        # Optionally select specific latitudes.
        if self.selected_lats is not None:
            # Assume a standard latitude grid from -30 to 70 with 2.5 degree steps.
            lat_arr = np.arange(-30, 70 + 2.5, 2.5)
            profiles = []
            sel_max_idx = []
            for sel in self.selected_lats:
                idx_lat = np.abs(lat_arr - sel).argmin()
                # Assuming y_data shape is (lev, rlat)
                profiles.append(y_data[:, idx_lat])
                sel_max_idx.append(max_idx[idx_lat])
            y_data = np.stack(profiles, axis=1)
            sel_max_idx = np.stack(sel_max_idx, axis=0)
        # Apply any transformation if provided.
        if self.transform:
            x_data_concat = self.transform(x_data_concat)

        if self.output_type == "max":
            #y_data = np.nanmax(y_data, axis=0) 
            y_data = y_data[max_idx[idx_lat]]

        y_data = np.nan_to_num(y_data)

        return torch.from_numpy(x_data_concat), torch.from_numpy(y_data)

if __name__ == "__main__":
    dataset = PreprocessedCMIP6Dataset(
        zarr_dir="/home/am334/link_am334/moc_mmodel",
        models=['ACCESS-ESM1-5'],
        x_vars=['zos', 'tos'],
        scenarios=["piControl"],
        output_type="max",
        selected_lats=[26.5],
        lpf="LPF24"
    )
    print("Number of samples:", len(dataset))
    x_tensor, y_tensor = dataset[0]
    print("x_tensor shape:", x_tensor.shape)  # Expected shape: [2, ...]
    print("y_tensor shape:", y_tensor.shape)
'''
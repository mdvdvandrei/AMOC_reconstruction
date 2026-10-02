#!/usr/bin/env python3
"""
Tables for the unified version of the manuscript.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

ROOT = "/dat1/am334/moc_mmodel"
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ssh_trend_honesty as S  # noqa: E402
from andrews_median_unbiased import rho_hat  # noqa: E402
from gls_blocks_calibration import block_mean, gls_ar1_wald  # noqa: E402
from obs_datasets import lowpass_filter  # noqa: E402

SLOPE = "GLS"  # "GLS" for the generalised slope, "OLS" for the ordinary one
BNN = os.path.join(S.ROOT, "bnn_real_world_rec")
DROP = ("test_model(CESM2)", "CAS-ESM2-0", "INM-CM4-8")
RAPID = {"non-Ekman": 0.48, "+Ekman": 0.44}
SST_120 = ("amoc_from_sst.nc", "amoc_from_sst_mu",
           "amoc_from_sst_sigma_total")
SST_120_ALEA = ("amoc_from_sst.nc", "amoc_from_sst_mu",
                "amoc_from_sst_sigma_aleatoric")
# the 2-year product stores no epistemic sigma, so its rows keep the aleatoric
SST_24 = ("amoc_from_hadisst_lpf24.nc", "amoc_noek", "amoc_alea_std")
# the same product with the Ekman term left in, for the figures that compare
# against CMIP6 members, whose trends are those of the whole transport
SST_24_FULL = ("amoc_from_hadisst_lpf24.nc", "amoc_mean", "amoc_alea_std")
SST_FOLDS = os.path.join(HERE, "R_obs_per_fold_y_new.npz")
FOLD_CACHE = {"amoc_from_sst.nc": SST_FOLDS,
              S.OBS["DUACS"]: os.path.join(HERE, "R_obs_ssh_per_fold.npz")}
SSH_24 = {"non-Ekman": (S.OBS["DUACS"], "amoc_from_ssh_mu_no_ekman",
                        "amoc_from_ssh_sigma_total"),
          "+Ekman": (S.OBS["DUACS"], "amoc_from_ssh_mu",
                     "amoc_from_ssh_sigma_total")}
SSH_24_ALEA = {k: (v[0], v[1], "amoc_from_ssh_sigma_aleatoric")
               for k, v in SSH_24.items()}
BLOCK = {"10-year": 1, "5-year": 1, "2-year": 1}  # annual means throughout
# lag beyond which the reconstruction error has forgotten itself, per filter
ACF_LAG = {"10-year": 10, "5-year": 8, "2-year": 6}


def folds(fn):
    """Per-fold mu and sigma. None if that product has no fold cache."""
    npz = FOLD_CACHE.get(fn)
    if npz is None or not os.path.exists(npz):
        return None
    z = np.load(npz, allow_pickle=True)
    ds = xr.open_dataset(os.path.join(BNN, fn))
    assert np.array_equal(pd.to_datetime(ds.time.values).values,
                          pd.to_datetime(z["time"]).values), \
        f"{os.path.basename(npz)} belongs to a different run than {fn}"
    return z["mu"].astype(float), z["sig"].astype(float)


def pooled_sigma(fn):
    """sqrt(mean_k sigma_k^2). None if there is no fold cache."""
    f = folds(fn)
    return None if f is None else np.sqrt((f[1] ** 2).mean(axis=0))


def fold_trend_spread(src, y0, y1, blk, R, extra_lpf_months=None):
    """sd of the GLS trend across folds. 0 if there is no fold cache."""
    f = folds(src[0])
    if f is None:
        return 0.0
    MU, SG = f
    b = []
    for k in range(MU.shape[0]):
        a = annual(src, y0, y1, extra_lpf_months, sigma=SG[k], mu=MU[k])
        b.append(slope(block_mean(a["mu"].values, blk),
                       block_mean(a["sig"].values, blk), blk, R)[0])
    return float(np.std(b, ddof=1))


def annual(src, y0, y1, extra_lpf_months=None, sigma=None, mu=None):
    """Annual means. sigma and mu override the file. Extra low-pass is zero-phase."""
    fn, muv, sigv = src
    ds = xr.open_dataset(os.path.join(BNN, fn))
    t = pd.to_datetime(ds.time.values)
    mu = ds[muv].values.astype(float) if mu is None else np.asarray(mu, float)
    sig = ds[sigv].values.astype(float) if sigma is None else np.asarray(sigma)
    ok = np.isfinite(mu) & np.isfinite(sig)
    if extra_lpf_months is not None:
        mu = mu.copy()
        mu[ok] = lowpass_filter(mu[ok], 1.0 / extra_lpf_months,
                                pad=2 * extra_lpf_months)
    m = ok & (t >= pd.Timestamp(f"{y0}-01-01")) & (t <= pd.Timestamp(f"{y1}-12-31"))
    return pd.DataFrame({"year": t[m].year, "mu": mu[m],
                         "sig": sig[m]}).groupby("year").mean()


def cmip_rho(art, var, win, y0, blk, extra_lpf_years=None):
    """Median AR(1) the ESMs read over a window of the same length and filter."""
    d = pd.read_parquet(art)
    d = d[(d.scenario == "historical") & (d["var"] == var)
          & ~d.model.isin(DROP)].drop_duplicates(
              subset=["model", "member", "year"])
    out = []
    for _, g in d.groupby(["model", "member"]):
        w = g.sort_values("year")
        w = w[(w.year >= y0) & (w.year <= y0 + win - 1)]
        mu = w.groupby("year").mu.mean().values
        if len(mu) < win - 1 or not np.isfinite(mu).all():
            continue
        sig = w.groupby("year").sig.mean().values
        if extra_lpf_years is not None:
            mu = lowpass_filter(mu, 1.0 / extra_lpf_years,
                                pad=2 * extra_lpf_years)
        out.append(gls_ar1_wald(block_mean(mu, blk), block_mean(sig, blk),
                                blk)[2])
    return float(np.median(out)), len(out)


_ACF_CACHE: dict = {}


def error_corr(art, var, K, drop, n):
    """Median error ACF on held-out windows, Bartlett-tapered past lag K."""
    key = (art, var, K, n)
    if key in _ACF_CACHE:
        return _ACF_CACHE[key]
    from trend_error_direct import windows
    W = windows(art, var, n, [1900] if var == "tos" else
                [y for y, _ in S.WINDOWS], drop)

    def acf(x):
        x = x - x.mean()
        d = (x * x).sum()
        return np.array([(x[:len(x) - h] * x[h:]).sum() / d
                         for h in range(K + 1)])

    r = np.median([acf(w["mu"] - w["tgt"]) for w in W], axis=0)
    r[0] = 1.0
    H = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    R = np.where(H <= K, r[np.minimum(H, K)], 0.0) * np.clip(1 - H / (K + 1), 0, 1)
    _ACF_CACHE[key] = (R, r)
    return R, r


def slope(y, s, blk, R):
    """GLS slope (Sv/century) and its ±2 se from Cov = D R D. A scalar R is AR(1)."""
    n = len(y)
    X = np.c_[np.ones(n), np.arange(n, dtype=float)]
    if np.ndim(R) == 0:  # a bare lag-1 value still means the AR(1) shape
        R = float(R) ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    Si = np.linalg.pinv(R * np.outer(s, s))
    V = np.linalg.pinv(X.T @ Si @ X)
    pm = 2.0 * np.sqrt(V[1, 1]) * 100.0 / blk
    if SLOPE == "OLS":
        b = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    else:
        b = float((V @ X.T @ Si @ y)[1])
    return b * 100.0 / blk, pm


def fit(a, blk, R, rho, tau=0.0):
    """Slope plus se = hypot(se_fit, tau). p is one-sided normal."""
    y, s = block_mean(a["mu"].values, blk), block_mean(a["sig"].values, blk)
    b, pm = slope(y, s, blk, R)
    se = float(np.hypot(pm / 2.0, tau))
    return dict(rho=rho, n=len(y), sigma=float(np.mean(s)), trend=b, se=se,
                se_fit=pm / 2.0, tau=tau, two_se=2 * se,
                lo=b - 2 * se, hi=b + 2 * se, p=float(norm.cdf(b / se)))


def case(group, label, src, y0, y1, blk, art, var, K, extra_lpf_months=None):
    """One series twice: measured ACF, and AR(1) at the series' own lag-1."""
    a = annual(src, y0, y1, extra_lpf_months, sigma=pooled_sigma(src[0]))
    y = block_mean(a["mu"].values, blk)
    n = len(y)
    R, r = error_corr(getattr(S, art), var, K, DROP if var == "tos" else S.DROP,
                      n)
    own = max(float(rho_hat(y)[0]), 0.0)
    rows = []
    for RR, rho, name in ((R, float(r[1]), "ACF"), (own, own, "AR(1)")):
        tau = fold_trend_spread(src, y0, y1, blk, RR, extra_lpf_months)
        row = fit(a, blk, RR, rho, tau)
        row.update(group=group, label=label, block=blk, rho_from=name)
        rows.append(row)
    return rows


def coverage(art, var, win, starts, blk, R, k=1.0, tau=0.0):
    """Fraction of held-out windows whose ±2 se contains the true trend."""
    from trend_error_direct import windows
    hit = []
    for w in windows(getattr(S, art), var, win, starts, DROP if var == "tos"
                     else S.DROP):
        b, pm = slope(block_mean(w["mu"], blk), k * block_mean(w["sig"], blk),
                      blk, R)
        hit.append(abs(b - w["true"]) <= 2 * np.hypot(pm / 2.0, tau))
    return float(np.mean(hit)), len(hit)


def sigma_carry(src, y0, y1):
    """Pooled sigma divided by the sigma stored in the file. 1 if no fold cache."""
    p = pooled_sigma(src[0])
    if p is None:
        return 1.0
    d = xr.open_dataset(os.path.join(BNN, src[0]))
    t = pd.to_datetime(d.time.values)
    m = (t.year >= y0) & (t.year <= y1)
    return float(np.nanmean(p[m]) / np.nanmean(d[src[2]].values[m]))


def build():
    rows = []
    k10 = sigma_carry(SST_120_ALEA, 1900, 2000)
    kssh = sigma_carry(SSH_24_ALEA["non-Ekman"], 1993, 2023)
    print(f"pooled within-fold sigma, as a multiple of the arithmetic mean the "
          f"file stores: SST {k10:.3f}, SSH {kssh:.3f} "
          f"(used as such on the record, carried over in the coverage test)")

    K10, K5, K2 = ACF_LAG["10-year"], ACF_LAG["5-year"], ACF_LAG["2-year"]

    # --- filtering ---
    rows += case("filter", "10-year low-pass", SST_120, 1900, 2000,
                 BLOCK["10-year"], "ART_120", "tos", K10)
    rows += case("filter", "5-year low-pass", SST_24, 1900, 2000,
                 BLOCK["5-year"], "ART_24", "tos", K5,
                 extra_lpf_months=60)
    rows += case("filter", "2-year low-pass", SST_24, 1900, 2000,
                 BLOCK["2-year"], "ART_24", "tos", K2)

    # --- window, 10-year low-pass ---
    for tag, (y0, y1) in (("end", (1900, 1990)), ("end", (1900, 2000)),
                          ("end", (1900, 2010)), ("start", (1890, 2000)),
                          ("start", (1900, 2000)), ("start", (1910, 2000))):
        rows += case(tag, f"{y0}--{y1}", SST_120, y0, y1,
                     BLOCK["10-year"], "ART_120", "tos", K10)

    # --- satellite era ---
    for y0, y1 in ((1993, 2023), (1994, 2023), (1993, 2020), (2000, 2023)):
        for ek in ("non-Ekman", "+Ekman"):
            rows += case("ssh", f"{y0}--{y1}, {ek}", SSH_24[ek], y0, y1,
                         BLOCK["2-year"], "ART_24", "zos_minus_basin_mean", K2)

    # --- the century row of the main table ---
    rows += case("main", "SST 1900--2000, 10-year", SST_120, 1900, 2000,
                 BLOCK["10-year"], "ART_120", "tos", K10)

    # --- the two windows the top panel of Figure 4 draws, so that the band it
    # shades is the same fit as the table rather than a second computation ---
    for y0, y1 in ((1900, 1985), (1900, 2000)):
        rows += case("fig4", f"SST {y0}--{y1}, 10-year", SST_120, y0, y1,
                     BLOCK["10-year"], "ART_120", "tos", K10)

    # --- the manuscript table also quotes the 2-year SST product over the
    # satellite era, which no other table needs ---
    for tag, src in (("non-Ekman", SST_24), ("+Ekman", SST_24_FULL)):
        rows += case("paper", f"SST 1993--2023, {tag}", src, 1993, 2023,
                     BLOCK["2-year"], "ART_24", "tos", K2)
    return pd.DataFrame(rows), (k10, kssh)


def ekman_trend():
    """Trend of the Ekman term alone, which carries no network uncertainty."""
    ds = xr.open_dataset(os.path.join(BNN, S.OBS["DUACS"]))
    t = pd.to_datetime(ds.time.values)
    m = (t >= pd.Timestamp(S.OBS_T0)) & (t <= pd.Timestamp(S.OBS_T1))
    ek = (ds["amoc_from_ssh_mu"].values[m]
          - ds["amoc_from_ssh_mu_no_ekman"].values[m]).astype(float)
    a = pd.DataFrame({"year": t[m].year, "v": ek}).groupby("year").v.mean()
    x = a.index.values.astype(float)
    return float(np.polyfit(x, a.values, 1)[0] * 100.0)


def fmt_p(p):
    s = f"{p:.3f}"
    return f"\\mathbf{{{s}}}" if p <= 0.05 else s


def latex(d, ek):
    def pick(group, label, src):
        r = d[(d.group == group) & (d.label == label) & (d.rho_from == src)]
        return r.iloc[0]

    def line(group, label, name=None):
        o = pick(group, label, "ACF")
        return (f"& {name if name is not None else label} & "
                f"${o.trend:+.2f}$ & $\\pm{o.two_se:.2f}$ & "
                f"${fmt_p(o.p)}$ \\\\")

    # The 5-year rows are still computed and kept in the csv, but not shown:
    # that case is not a reconstruction of its own, only the 2-year one passed
    # through a further low-pass, so it does not test anything the 2-year row
    # does not already test. The AR(1) columns are dropped for the same reason
    # the table no longer quotes rho at all - the correlation is measured, and
    # the geometric alternative is discussed in the text rather than tabulated.
    print("\n% ==== sensitivity table ====")
    print("\\begin{center}\\small")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("& Series & Trend & $\\pm2$ s.e. & $p$ \\\\")
    print("\\midrule")
    print("\\multicolumn{5}{l}{\\emph{Filtering} (SST, 1900--2000)} \\\\")
    for lab in ("10-year low-pass", "2-year low-pass"):
        print(line("filter", lab))
    print("\\addlinespace")
    print("\\multicolumn{5}{l}{\\emph{End of window} (SST, 10-year low-pass)} \\\\")
    for lab in ("1900--1990", "1900--2000", "1900--2010"):
        print(line("end", lab))
    print("\\addlinespace")
    print("\\multicolumn{5}{l}{\\emph{Start of window} (SST, 10-year low-pass)} \\\\")
    for lab in ("1890--2000", "1900--2000", "1910--2000"):
        print(line("start", lab))
    print("\\addlinespace\n\\midrule")
    print("\\multicolumn{5}{l}{\\emph{Satellite era} (SSH, 2-year low-pass)} \\\\")
    for y0, y1 in ((1993, 2023), (1994, 2023), (1993, 2020), (2000, 2023)):
        for ek_tag in ("non-Ekman", "+Ekman"):
            print(line("ssh", f"{y0}--{y1}, {ek_tag}"))
    print("\\bottomrule\n\\end{tabular}\n\\end{center}")
    print("\\noindent(Sv\\,century$^{-1}$. Every row is a generalised least "
          "squares fit on annual means, both the slope and its error. The "
          "weights are the pooled within-fold scale of the ensemble mean, "
          "$\\sigma = [\\,\\overline{\\sigma_k^2}\\,]^{1/2}$ over the folds, and "
          "the error correlation is taken from the errors themselves: the median "
          "autocorrelation of reconstruction minus truth over the held-out CMIP6 "
          "windows. To this is added, the standard deviation of the trend across "
          "the folds, which is the dependence of the dataset on which ESMs the "
          "network was trained on. The $p$-value is one-sided for $H_1$: trend "
          "$<0$, from the normal distribution, under the measured correlation; "
          "bold marks $p \\le 0.05$.)")

    print("\n% ==== manuscript table ====")
    print("\\begin{tabular}{lllcc}")
    print("\\toprule")
    print("\\textbf{Filter} & \\textbf{Reconstruction} & \\textbf{Period} & "
          "\\textbf{Trend} & \\textbf{$p$} \\\\")
    print("\\midrule")
    for i, (y0, y1) in enumerate(((1900, 1985), (1900, 2000))):
        r = pick("fig4", f"SST {y0}--{y1}, 10-year", "ACF")
        print(f"{'10-year' if i == 0 else '       '} & {'SST' if i == 0 else '   '} "
              f"& {y0}--{y1} & ${r.trend:+.2f} \\pm {r.two_se:.2f}$ & "
              f"${fmt_p(r.p)}$ \\\\")
    print("\\addlinespace")
    for group, series in (("paper", "SST"), ("ssh", "SSH")):
        for i, ek_tag in enumerate(("non-Ekman", "+Ekman")):
            key = (f"SST 1993--2023, {ek_tag}" if group == "paper"
                   else f"1993--2023, {ek_tag}")
            r = pick(group, key, "ACF")
            name = f"{series}, {'no Ekman' if i == 0 else 'with Ekman'}"
            print(f"{'2-year' if i == 0 else '      '} & {name} & 1993--2023 & "
                  f"${r.trend:+.2f} \\pm {r.two_se:.2f}$ & "
                  f"${fmt_p(r.p)}$ \\\\")
        if group == "paper":
            print("\\addlinespace")
    print("\\bottomrule\n\\end{tabular}")

    print("\n% ==== main table ====")
    print("\\begin{center}\\small")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Series & Filter & Trend & $\\pm2$ s.e. & Interval "
          "& $p$ (weakening) \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\emph{SSH, 1993--2023}} \\\\")
    for ek_tag in ("non-Ekman", "+Ekman"):
        r = pick("ssh", f"1993--2023, {ek_tag}", "ACF")
        print(f"\\quad {ek_tag} & 2-year & "
              f"${r.trend:+.2f}$ & $\\pm{r.two_se:.2f}$ & "
              f"${r.lo:+.2f},\\,{r.hi:+.2f}$ & ${fmt_p(r.p)}$ \\\\")
    print(f"\\quad Ekman only & 2-year & ${ek:+.2f}$ & --- & --- & --- \\\\")
    print("\\addlinespace")
    print("\\multicolumn{6}{l}{\\emph{SST, 1900--2000}} \\\\")
    r = pick("main", "SST 1900--2000, 10-year", "ACF")
    print(f"\\quad total & 10-year & ${r.trend:+.2f}$ & "
          f"$\\pm{r.two_se:.2f}$ & ${r.lo:+.2f},\\,{r.hi:+.2f}$ & "
          f"${fmt_p(r.p)}$ \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{center}")
    print("\\noindent(Sv\\,century$^{-1}$, same convention as the sensitivity "
          "table: generalised least squares on annual means, weighted by the "
          "pooled within-fold $\\sigma$ of the ensemble mean, with the error "
          "correlation measured on held-out CMIP6 reconstructions rather than "
          "assumed geometric, and with the across-fold spread of the trend "
          "added in quadrature. Applied to the held-out "
          "CMIP6 windows, where the true trend is known, the interval contains "
          "it in 95.0 per cent of the 100 century windows and 90.9 per cent of "
          "the 440 satellite-length windows. The Ekman transport is computed from "
          "ERA5 wind stress and carries no reconstruction uncertainty, so the two "
          "SSH rows share a $\\sigma$ and an interval and differ only in the "
          "slope.)")


def main():
    d, (k10, kssh) = build()
    ek = ekman_trend()

    hdr = (f"{'group':7s} {'series':26s} {'blk':>3s} {'err mod':>8s} "
           f"{'rho1':>5s} {'sigma':>6s} {'trend':>7s} {'se fit':>6s} "
           f"{'tau':>5s} {'2 s.e.':>7s} {'interval':>16s} {'p':>6s}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for _, r in d.iterrows():
        print(f"{r.group:7s} {r.label:26s} {r.block:>3d} {r.rho_from:>8s} "
              f"{r.rho:>5.2f} {r.sigma:>6.3f} {r.trend:>+7.2f} "
              f"{r.se_fit:>6.3f} {r.tau:>5.3f} {r.two_se:>7.2f} "
              f"{'%+.2f, %+.2f' % (r.lo, r.hi):>16s} {r.p:>6.3f}")
    print(f"\nEkman term alone, 1993-2023: {ek:+.2f} Sv/century")

    R10, _ = error_corr(S.ART_120, "tos", ACF_LAG["10-year"], DROP, 101)
    t10 = float(d[(d.group == "main") & (d.rho_from == "ACF")].tau.iloc[0])
    for tau, tag in ((0.0, "fit only"), (t10, "with the across-fold term")):
        c, n = coverage("ART_120", "tos", 101, [1900], BLOCK["10-year"], R10,
                        k=k10, tau=tau)
        print(f"coverage of the true CMIP6 trend, century row, {tag}: "
              f"{c:.3f} over {n} windows")
    R2, _ = error_corr(S.ART_24, "zos_minus_basin_mean", ACF_LAG["2-year"],
                       S.DROP, S.WIN)
    t2 = float(d[(d.group == "ssh") & (d.label == "1993--2023, non-Ekman")
                 & (d.rho_from == "ACF")].tau.iloc[0])
    for tau, tag in ((0.0, "fit only"), (t2, "with the across-fold term")):
        c24, n24 = coverage("ART_24", "zos_minus_basin_mean", S.WIN,
                            [y for y, _ in S.WINDOWS], BLOCK["2-year"], R2,
                            k=kssh, tau=tau)
        print(f"coverage of the satellite-era convention, {tag}: {c24:.3f} "
              f"over {n24} windows of that length")

    latex(d, ek)
    d.to_csv(os.path.join(HERE, "R_tables_unified.csv"), index=False)
    print(f"\n[SAVED] {HERE}/R_tables_unified.csv")


if __name__ == "__main__":
    main()

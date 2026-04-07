"""Per-site damage detection via CUSUM on orbit-normalized z-scores.

Reads one or more CSVs with columns (system:time_start, VH, VV, ...), fuses VV
and VH into a Mahalanobis magnitude m_t against a robust (median/MAD) pre-war
baseline, runs Page's one-sided CUSUM with a persistence guard, reports the
detection date and Page's change-point estimate, and writes a two-panel figure
per input.

Usage:
    python code/cusum_damage_detector.py data/no_speckle.csv data/speckle.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------- detector core --------------------------------


@dataclass
class DetectorConfig:
    k: float = 2.0           # CUSUM reference value (slack)
    h: float = 5.0           # CUSUM decision interval
    persistence: int = 3     # require k_p consecutive m_t > k after alarm
    ref_frac: float = 0.85   # fraction of series treated as pre-war reference
                             # for robust median/MAD baseline (no event labels)


@dataclass
class DetectorResult:
    times: pd.DatetimeIndex
    m: np.ndarray            # fused Mahalanobis magnitude per obs
    s: np.ndarray            # CUSUM statistic
    alarm_idx: int | None    # first index where S_t > h AND persistence holds
    tau_hat_idx: int | None  # Page's change-point estimate
    cfg: DetectorConfig


def robust_z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Re-z-score x using median/MAD computed on ref."""
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    scale = 1.4826 * mad if mad > 0 else np.std(ref) or 1.0
    return (x - med) / scale


def mahalanobis_magnitude(z_vv: np.ndarray, z_vh: np.ndarray,
                          ref_slice: slice) -> np.ndarray:
    """sqrt(z^T Σ_pre^{-1} z) using pre-war covariance."""
    Z = np.column_stack([z_vv, z_vh])
    ref = Z[ref_slice]
    cov = np.cov(ref.T)
    # regularize for safety
    cov += 1e-6 * np.eye(2)
    inv = np.linalg.inv(cov)
    quad = np.einsum("ij,jk,ik->i", Z, inv, Z)
    quad = np.clip(quad, 0, None)
    return np.sqrt(quad)


def cusum(m: np.ndarray, k: float) -> np.ndarray:
    s = np.zeros_like(m)
    for t in range(1, len(m)):
        s[t] = max(0.0, s[t - 1] + (m[t] - k))
    return s


def detect(times: pd.DatetimeIndex, vv: np.ndarray, vh: np.ndarray,
           cfg: DetectorConfig) -> DetectorResult:
    n = len(vv)
    n_ref = max(10, int(cfg.ref_frac * n))
    ref = slice(0, n_ref)

    # 1. Robustify channels with median/MAD on the reference slice
    z_vv = robust_z(vv, vv[ref])
    z_vh = robust_z(vh, vh[ref])

    # 2. Fuse via Mahalanobis magnitude
    m = mahalanobis_magnitude(z_vv, z_vh, ref)

    # 3. CUSUM
    s = cusum(m, cfg.k)

    # 4. Alarm + persistence guard
    alarm_idx: int | None = None
    for t in range(len(s)):
        if s[t] > cfg.h:
            window = m[t : t + cfg.persistence]
            if len(window) >= cfg.persistence and np.all(window > cfg.k):
                alarm_idx = t
                break

    # 5. Page's change-point estimator: last reset to 0 before alarm
    tau_hat_idx: int | None = None
    if alarm_idx is not None:
        zeros = np.where(s[: alarm_idx + 1] == 0)[0]
        tau_hat_idx = (zeros[-1] + 1) if len(zeros) else 0
        tau_hat_idx = min(tau_hat_idx, len(s) - 1)

    return DetectorResult(times, m, s, alarm_idx, tau_hat_idx, cfg)


# ----------------------------- io / plotting --------------------------------


def load(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["system:time_start"], format="%b %d, %Y")
    df = df.sort_values("time").reset_index(drop=True)
    return pd.DatetimeIndex(df["time"]), df["VV"].to_numpy(), df["VH"].to_numpy()


def plot(result: DetectorResult, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax = axes[0]
    ax.plot(result.times, result.m, color="#1f77b4", lw=1.2, marker="o", ms=3)
    ax.axhline(result.cfg.k, color="gray", ls="--", lw=0.8,
               label=f"k = {result.cfg.k}")
    ax.set_ylabel("m_t  (Mahalanobis magnitude)")
    ax.set_title(title)

    ax2 = axes[1]
    ax2.plot(result.times, result.s, color="#d62728", lw=1.2)
    ax2.axhline(result.cfg.h, color="black", ls="--", lw=0.8,
                label=f"h = {result.cfg.h}")
    ax2.set_ylabel("CUSUM  S_t")
    ax2.set_xlabel("date")

    if result.alarm_idx is not None:
        for a in (ax, ax2):
            a.axvline(result.times[result.alarm_idx], color="red", lw=1,
                      label="alarm")
    if result.tau_hat_idx is not None:
        for a in (ax, ax2):
            a.axvline(result.times[result.tau_hat_idx], color="green", lw=1,
                      ls=":", label="τ̂  (change-point)")

    for a in axes:
        a.legend(loc="upper left", fontsize=8)
        a.grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summarize(label: str, result: DetectorResult) -> dict:
    n = len(result.m)
    n_ref = int(result.cfg.ref_frac * n)
    pre = result.m[:n_ref]
    post = result.m[n_ref:]
    alarm_date = (result.times[result.alarm_idx].strftime("%Y-%m-%d")
                  if result.alarm_idx is not None else "—")
    tau_date = (result.times[result.tau_hat_idx].strftime("%Y-%m-%d")
                if result.tau_hat_idx is not None else "—")
    return {
        "file": label,
        "n_obs": n,
        "max_pre_m": round(float(pre.max()), 2),
        "mean_post_m": round(float(post.mean()), 2) if len(post) else float("nan"),
        "alarm_date": alarm_date,
        "tau_hat": tau_date,
    }


# --------------------------------- main -------------------------------------


def calibrate_arl0(cfg: DetectorConfig, n_runs: int = 10_000,
                   run_len: int = 1000, seed: int = 0) -> float:
    """Empirical ARL0 by simulating bivariate N(0, I) → m_t = sqrt(chi^2_2)."""
    rng = np.random.default_rng(seed)
    run_lengths = []
    for _ in range(n_runs):
        z = rng.standard_normal((run_len, 2))
        m = np.sqrt((z ** 2).sum(axis=1))
        s = cusum(m, cfg.k)
        crossings = np.where(s > cfg.h)[0]
        run_lengths.append(crossings[0] if len(crossings) else run_len)
    return float(np.mean(run_lengths))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--h", type=float, default=5.0)
    p.add_argument("--persistence", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--calibrate", action="store_true",
                   help="run Monte-Carlo ARL0 estimate and exit")
    args = p.parse_args()

    cfg = DetectorConfig(k=args.k, h=args.h, persistence=args.persistence)

    if args.calibrate:
        arl0 = calibrate_arl0(cfg)
        print(f"Empirical ARL0 (k={cfg.k}, h={cfg.h}): {arl0:.0f} obs")
        return

    rows = []
    for csv in args.csvs:
        times, vv, vh = load(csv)
        result = detect(times, vv, vh, cfg)
        out_png = args.out_dir / f"cusum_{csv.stem}.png"
        plot(result, title=f"CUSUM damage detector — {csv.stem}",
             out_path=out_png)
        rows.append(summarize(csv.stem, result))
        print(f"wrote {out_png}")

    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

"""Compare CUSUM-on-z-score-timeseries vs PWTT T-statistic.

For each city in `--cities`:
  1. Pull a stratified sample of N damaged + N undamaged building footprints
     (labeled by spatial-join with the same UNOSAT/CDA points eval.py uses).
  2. Build a per-orbit z-score ImageCollection over the post-war window
     (mirrors the normalize_orbit_images logic in pwtt/__init__.py:357).
  3. For each building, reduce regions over each image to get a per-date
     (VV, VH) z-score time series.
  4. Score with CUSUM (max S_t) and with the standard PWTT T-statistic.
  5. Report AUC/F1 for both.

Usage:
    python code/cusum_eval.py --cities Bucha               # smoke
    python code/cusum_eval.py --cities Bucha Hostomel Irpin --n-per-class 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from cusum_damage_detector import cusum  # noqa: E402
from eval import CITIES, run_evaluation  # noqa: E402
from pwtt import detect_damage  # noqa: E402

ee.Initialize(project="ggmap-325812")


# --------------------------- z-score time series ----------------------------


def zscore_collection(aoi: ee.Geometry, war_start: ee.Date,
                      pre_interval: int, post_interval: int) -> ee.ImageCollection:
    """Per-orbit-normalized Sentinel-1 (VV, VH) z-scores over the post-war window.

    Mirrors pwtt/__init__.py:357 (the hotelling/mahalanobis branch) but returns
    only the post-war slice and exposes it as an ImageCollection for time-series
    extraction.
    """
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
          .filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))

    pre_start = war_start.advance(-pre_interval, "month")
    post_end = war_start.advance(post_interval, "month")

    orbits = s1.filterDate(pre_start, post_end) \
        .aggregate_array("relativeOrbitNumber_start").distinct()

    def normalize_one_orbit(orbit):
        orbit_imgs = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
            .filterDate(pre_start, post_end)
        pre = orbit_imgs.filterDate(pre_start, war_start)
        has_pre = pre.select("VV").count().reduceRegion(
            ee.Reducer.max(), aoi, 1000).values().get(0)
        pre_mean = pre.mean()
        pre_sd = pre.reduce(ee.Reducer.stdDev()).rename(["VV", "VH"])
        post = orbit_imgs.filterDate(war_start, post_end)

        def normalize(img):
            return (img.subtract(pre_mean)
                    .divide(pre_sd.max(ee.Image.constant(1e-10)))
                    .copyProperties(img, ["system:time_start"]))

        normalized = post.map(normalize).toList(500)
        return ee.Algorithms.If(ee.Number(has_pre).gt(0), normalized, ee.List([]))

    return ee.ImageCollection(orbits.map(normalize_one_orbit).flatten())


# --------------------------- building sampling ------------------------------


def labeled_footprints(footprints, ground_truth, bounds) -> ee.FeatureCollection:
    """Replicate eval.py's spatial-join labeling: class=1 if any damage point
    falls within a 10m buffer of the footprint, else 0."""
    fp = ee.FeatureCollection(footprints) if isinstance(footprints, str) else footprints
    fp = (fp.filterBounds(bounds)
          .map(lambda f: f.set("area", f.geometry().area(10)))
          .filter(ee.Filter.gt("area", 50)))

    pts = (ee.FeatureCollection(ground_truth)
           if isinstance(ground_truth, str) else ground_truth)
    buffered = pts.map(lambda f: f.buffer(10))

    join = ee.Join.saveAll(matchesKey="dpts", outer=True)
    spatial = ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=10)

    def label(f):
        n = ee.List(f.get("dpts")).size()
        return ee.Algorithms.If(
            ee.Number(n).gt(0), f.set("class", 1), f.set("class", 0))

    return join.apply(fp, buffered, spatial).map(label)


def stratified_sample(labeled_fp: ee.FeatureCollection,
                      n_per_class: int, seed: int = 42) -> ee.FeatureCollection:
    """Take n_per_class damaged + n_per_class undamaged footprints, assign bid."""
    pos = (labeled_fp.filter(ee.Filter.eq("class", 1))
           .randomColumn("r", seed).sort("r").limit(n_per_class))
    neg = (labeled_fp.filter(ee.Filter.eq("class", 0))
           .randomColumn("r", seed).sort("r").limit(n_per_class))
    merged = pos.merge(neg)

    # Assign a stable integer bid via aggregate of system:index
    indexed = merged.toList(merged.size())
    bids = ee.List.sequence(0, merged.size().subtract(1))
    pairs = bids.zip(indexed)

    def with_bid(pair):
        pair = ee.List(pair)
        return ee.Feature(pair.get(1)).set("bid", ee.Number(pair.get(0)))

    return ee.FeatureCollection(pairs.map(with_bid))


# ------------------------- pull per-building series -------------------------


def pull_timeseries(zscore_ic: ee.ImageCollection,
                    sample: ee.FeatureCollection) -> pd.DataFrame:
    """Reduce each z-score image over each footprint, return long DataFrame."""
    # Keep geometry on `sample` so reduceRegions can use it; only drop geometry
    # on the per-image reduced features (which we never need spatially again).
    sample = sample.select(["bid", "class", "area"])

    def reduce_one(img):
        fc = img.reduceRegions(
            collection=sample,
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=8,
        )
        date = img.date().format("YYYY-MM-dd")
        return fc.map(lambda f: f.set("date", date)
                      .select(["bid", "class", "date", "VV", "VH"], None, False))

    all_obs = ee.FeatureCollection(zscore_ic.map(reduce_one)).flatten()
    all_obs = all_obs.filter(ee.Filter.notNull(["VV", "VH"]))

    rows = []
    page_size = 5000
    total = all_obs.size().getInfo()
    print(f"    pulling {total:,} (building × date) observations...")
    offset = 0
    while offset < total:
        page = all_obs.toList(page_size, offset).getInfo()
        for f in page:
            p = f["properties"]
            rows.append((p["bid"], p["class"], p["date"], p["VV"], p["VH"]))
        offset += page_size
    return pd.DataFrame(rows, columns=["bid", "class", "date", "VV", "VH"])


# ------------------------------ scoring -------------------------------------


def cusum_score_per_building(df: pd.DataFrame,
                             k: float = 2.0) -> pd.DataFrame:
    """For each building, compute fused magnitude m_t = sqrt(VV²+VH²),
    then max CUSUM as the damage score."""
    out = []
    for bid, g in df.sort_values("date").groupby("bid"):
        vv = g["VV"].to_numpy()
        vh = g["VH"].to_numpy()
        m = np.sqrt(vv ** 2 + vh ** 2)
        s = cusum(m, k)
        out.append({
            "bid": bid,
            "class": int(g["class"].iloc[0]),
            "n_obs": len(m),
            "max_m": float(m.max()) if len(m) else np.nan,
            "max_cusum": float(s.max()) if len(s) else np.nan,
            "mean_m": float(m.mean()) if len(m) else np.nan,
        })
    return pd.DataFrame(out)


def auc_only(labels, scores, weights=None):
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    return auc(fpr, tpr)


def f1_at_best(labels, scores, weights=None):
    p, r, t = precision_recall_curve(labels, scores, sample_weight=weights)
    with np.errstate(invalid="ignore"):
        f = 2 * p * r / (p + r)
    f = np.nan_to_num(f)
    i = int(np.argmax(f))
    return float(f[i]), float(t[i] if i < len(t) else 0.0)


# ------------------------------ pwtt scores ---------------------------------


def pwtt_t_scores(sample: ee.FeatureCollection, bounds: ee.Geometry,
                  inference_start, war_start, post_interval: int) -> pd.DataFrame:
    """Standard PWTT: detect_damage → reduceRegions → per-building T_statistic."""
    img = detect_damage(
        aoi=bounds,
        inference_start=(inference_start if isinstance(inference_start, ee.Date)
                         else ee.Date(inference_start)),
        war_start=ee.Date(war_start),
        pre_interval=12,
        post_interval=post_interval,
        method="stouffer",
    )
    fc = img.reduceRegions(
        collection=sample.select(["bid", "class"]),
        reducer=ee.Reducer.mean(),
        scale=10,
        tileScale=8,
    ).filter(ee.Filter.notNull(["T_statistic"]))
    fc = fc.select(["bid", "class", "T_statistic"], None, False)

    rows = []
    total = fc.size().getInfo()
    page_size = 5000
    offset = 0
    while offset < total:
        page = fc.toList(page_size, offset).getInfo()
        for f in page:
            p = f["properties"]
            rows.append((p["bid"], p["class"], p["T_statistic"]))
        offset += page_size
    return pd.DataFrame(rows, columns=["bid", "class", "t"])


# ---------------------------------- main ------------------------------------


def run_city(city: dict, n_per_class: int, post_interval: int) -> dict | None:
    name = city["name"]
    print(f"\n=== {name} ===")
    bounds = city["bounds"]
    war_start = city["war_start"]
    inf_start = city["inference_start"]
    inf_date = inf_start if isinstance(inf_start, ee.Date) else ee.Date(inf_start)
    war_date = ee.Date(war_start)

    fp = labeled_footprints(city["footprints"], city["ground_truth"], bounds)
    sample = stratified_sample(fp, n_per_class)
    n_pos = sample.filter(ee.Filter.eq("class", 1)).size().getInfo()
    n_neg = sample.filter(ee.Filter.eq("class", 0)).size().getInfo()
    print(f"  sampled {n_pos} damaged + {n_neg} undamaged buildings")
    if n_pos < 5 or n_neg < 5:
        print("  too few buildings, skipping")
        return None

    # Months from war_start through inference_start + post_interval (post-war window)
    months_span = int(round(
        (inf_date.advance(post_interval, "month").millis().getInfo()
         - war_date.millis().getInfo()) / (1000 * 60 * 60 * 24 * 30.44)
    )) + 1
    zic = zscore_collection(
        aoi=bounds, war_start=war_date,
        pre_interval=12, post_interval=months_span)
    n_imgs = zic.size().getInfo()
    print(f"  {n_imgs} z-score images in post-war window")

    df = pull_timeseries(zic, sample)
    if df.empty:
        print("  no time series data, skipping")
        return None

    # Per-building CUSUM
    cs = cusum_score_per_building(df)
    cs = cs[cs["n_obs"] >= 5]
    print(f"  {len(cs)} buildings have ≥5 observations")

    # PWTT T-statistic for the same buildings
    ts = pwtt_t_scores(sample, bounds, inf_start, war_start, post_interval)
    merged = cs.merge(ts[["bid", "t"]], on="bid", how="inner")
    print(f"  {len(merged)} buildings have both CUSUM and PWTT scores")

    if len(merged) < 10 or merged["class"].sum() < 3:
        print("  insufficient overlap for AUC, skipping")
        return None

    labels = merged["class"].values
    auc_cusum = auc_only(labels, merged["max_cusum"].values)
    auc_t = auc_only(labels, merged["t"].values)
    auc_m = auc_only(labels, merged["max_m"].values)
    f1_cusum, _ = f1_at_best(labels, merged["max_cusum"].values)
    f1_t, _ = f1_at_best(labels, merged["t"].values)

    print(f"  CUSUM      AUC={auc_cusum:.3f}  F1={f1_cusum:.3f}")
    print(f"  max |z|    AUC={auc_m:.3f}")
    print(f"  PWTT (T)   AUC={auc_t:.3f}  F1={f1_t:.3f}")

    return dict(
        city=name, n=len(merged), n_pos=int(labels.sum()),
        auc_cusum=auc_cusum, f1_cusum=f1_cusum,
        auc_max_m=auc_m,
        auc_pwtt=auc_t, f1_pwtt=f1_t,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=["Bucha"])
    p.add_argument("--n-per-class", type=int, default=75)
    p.add_argument("--post-interval", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("data/cusum_eval_results.csv"))
    args = p.parse_args()

    selected = [c for c in CITIES if c["name"] in args.cities]
    if not selected:
        print(f"No cities matched {args.cities}")
        return

    results = []
    for city in selected:
        try:
            r = run_city(city, args.n_per_class, args.post_interval)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  {city['name']} FAILED: {e!r}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        # Weighted average across cities
        w = df["n"].values
        for col in ["auc_cusum", "auc_pwtt", "auc_max_m", "f1_cusum", "f1_pwtt"]:
            print(f"  weighted {col}: {np.average(df[col], weights=w):.3f}")
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

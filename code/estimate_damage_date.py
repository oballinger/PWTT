"""
Estimate the date of damage for flagged buildings by downloading per-orbit
normalized z-score time series from Sentinel-1.

Reads the damage points CSV, filters to T_statistic > threshold, then for
each H3 cell queries EE for the orbit-normalized VV/VH z-scores at every
post-war acquisition date. Exports one CSV per cell to Google Drive, then
downloads and concatenates them locally. Finally, estimates a damage date
per building as the first post-war image where max(|z_vv|, |z_vh|) > z_crit
for two consecutive acquisitions.

Usage:
    python code/estimate_damage_date.py \
        --csv data/iran_damage_points_v20260410.csv \
        --threshold 1.0 \
        --war-start 2026-03-01 \
        --output data/iran_damage_dates.csv
"""

import argparse
import ee
import os
import time
import pandas as pd
import numpy as np


def get_s1_base():
    """Base Sentinel-1 collection (log-scale VV/VH, IW mode)."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
        .map(lambda img: img.log().copyProperties(img, img.propertyNames()))
    )


def build_zscore_collection(aoi, war_start, pre_months=12):
    """Build orbit-normalized z-score ImageCollection over an AOI.

    For each orbit covering the AOI, normalizes all images by that orbit's
    pre-war mean and std (matching the JS split-panel app exactly).
    Returns an ImageCollection with bands ['VV', 'VH'] in z-score space,
    plus properties 'system:time_start' and 'orbit'.
    """
    s1 = get_s1_base().filterBounds(aoi)
    pre_start = war_start.advance(-pre_months, "month")
    orbits = (
        s1.filterDate(pre_start, ee.Date("2099-01-01"))
        .aggregate_array("relativeOrbitNumber_start")
        .distinct()
    )

    def normalize_orbit(orbit):
        orbit_coll = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        pre = orbit_coll.filterDate(pre_start, war_start)
        pre_mean = pre.mean()
        pre_sd = pre.reduce(ee.Reducer.stdDev()).rename(["VV", "VH"])
        safe_sd = pre_sd.max(ee.Image.constant(1e-10))

        return (
            orbit_coll.map(
                lambda img: img.subtract(pre_mean)
                .divide(safe_sd)
                .set("orbit", orbit)
                .copyProperties(img, ["system:time_start"])
            )
            .toList(500)
        )

    return ee.ImageCollection(orbits.map(normalize_orbit).flatten())


def sample_zscore_timeseries(zscore_coll, points_fc, war_start, scale=10):
    """Sample the z-score collection at point locations for post-war dates.

    Returns an EE FeatureCollection with one row per (point, date) pair,
    containing: latitude, longitude, date_millis, z_vv, z_vh, orbit.
    """
    post = zscore_coll.filterDate(war_start, ee.Date("2099-01-01"))

    def sample_one_image(img):
        img = ee.Image(img)
        date_millis = img.get("system:time_start")
        orbit = img.get("orbit")
        sampled = img.select(["VV", "VH"]).sampleRegions(
            collection=points_fc,
            scale=scale,
            geometries=False,
        )
        return sampled.map(
            lambda f: f.set("date_millis", date_millis).set("orbit", orbit)
        )

    return post.map(sample_one_image).flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate damage dates from orbit-normalized S1 z-scores."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to damage points CSV (with T_statistic, latitude, longitude, h3_cell)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum T_statistic to include (default: 1.0)",
    )
    parser.add_argument("--war-start", default="2026-03-01", help="War start date")
    parser.add_argument(
        "--pre-months",
        type=int,
        default=12,
        help="Months of pre-war baseline (default: 12)",
    )
    parser.add_argument(
        "--z-crit",
        type=float,
        default=2.576,
        help="Z-score threshold for damage detection (default: 2.576 = 99%% CI)",
    )
    parser.add_argument(
        "--output", default="data/iran_damage_dates.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--drive-folder",
        default="iran_zscore_timeseries",
        help="Google Drive folder for EE exports",
    )
    parser.add_argument("--project", default="ggmap-325812", help="GEE project ID")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip export, just download existing Drive files and estimate dates",
    )
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    # ── Step 1: Read and filter CSV ──────────────────────────────────────
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    df = df[df["T_statistic"] > args.threshold].copy()
    print(f"  {len(df)} points with T_statistic > {args.threshold}")
    print(f"  {df['h3_cell'].nunique()} H3 cells")

    # ── Step 2: Export z-score time series per H3 cell ───────────────────
    war_start = ee.Date(args.war_start)
    cells = sorted(df["h3_cell"].unique())

    if not args.download_only:
        # Check for existing/running tasks to enable resume
        existing_tasks = set()
        try:
            for t in ee.batch.Task.list():
                if t.state in ("READY", "RUNNING", "COMPLETED"):
                    existing_tasks.add(t.config.get("description", ""))
        except Exception:
            pass

        submitted = 0
        skipped = 0
        for i, cell_id in enumerate(cells):
            task_name = f"zscore_{cell_id}"
            if task_name in existing_tasks:
                skipped += 1
                continue

            cell_df = df[df["h3_cell"] == cell_id]
            # Build EE point collection for this cell
            features = []
            for _, row in cell_df.iterrows():
                f = ee.Feature(
                    ee.Geometry.Point([row["longitude"], row["latitude"]]),
                    {"latitude": row["latitude"], "longitude": row["longitude"]},
                )
                features.append(f)
            points_fc = ee.FeatureCollection(features)

            # Build AOI from H3 cell boundary
            import h3

            boundary = h3.cell_to_boundary(cell_id)
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            aoi = ee.Geometry.Polygon([coords])

            # Build z-score collection and sample
            zscore_coll = build_zscore_collection(aoi, war_start, args.pre_months)
            sampled = sample_zscore_timeseries(zscore_coll, points_fc, war_start)

            # Select output properties
            sampled = sampled.select(
                ["latitude", "longitude", "date_millis", "orbit", "VV", "VH"]
            )

            task = ee.batch.Export.table.toDrive(
                collection=sampled,
                description=task_name,
                folder=args.drive_folder,
                fileFormat="CSV",
            )
            task.start()
            submitted += 1

            if (submitted) % 10 == 0:
                print(f"  Submitted {submitted} tasks ({i+1}/{len(cells)} cells)...")

        print(
            f"  Submitted {submitted} tasks, skipped {skipped} existing. "
            f"Waiting for completion..."
        )

        # ── Step 3: Wait for all tasks ───────────────────────────────────
        while True:
            tasks = ee.batch.Task.list()
            zscore_tasks = [
                t for t in tasks if t.config.get("description", "").startswith("zscore_")
            ]
            running = sum(1 for t in zscore_tasks if t.state in ("READY", "RUNNING"))
            completed = sum(1 for t in zscore_tasks if t.state == "COMPLETED")
            failed = sum(1 for t in zscore_tasks if t.state == "FAILED")
            print(
                f"  Running: {running}  Completed: {completed}  Failed: {failed}",
                end="\r",
            )
            if running == 0:
                print()
                break
            time.sleep(30)

        if failed > 0:
            print(f"  WARNING: {failed} tasks failed")
            for t in zscore_tasks:
                if t.state == "FAILED":
                    print(f"    {t.config.get('description')}: {t.status().get('error_message', '?')}")

    # ── Step 4: Download from Drive ──────────────────────────────────────
    print("Downloading CSVs from Google Drive...")
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    import io

    # Reuse EE credentials for Drive API
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None
    token_path = os.path.expanduser("~/.config/earthengine/drive_token.pickle")
    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Fall back to ee credentials
            ee_creds = ee.data._credentials
            if hasattr(ee_creds, "token"):
                from google.oauth2.credentials import Credentials as OAuthCreds

                creds = ee_creds
            else:
                print(
                    "  Cannot access Drive API. Download CSVs manually from "
                    f"Drive folder '{args.drive_folder}' and re-run with --download-only."
                )
                return

    service = build("drive", "v3", credentials=creds)

    # Find the folder
    folder_query = (
        f"name = '{args.drive_folder}' and mimeType = 'application/vnd.google-apps.folder'"
    )
    folders = service.files().list(q=folder_query, spaces="drive").execute()
    folder_items = folders.get("files", [])
    if not folder_items:
        print(f"  Drive folder '{args.drive_folder}' not found. Trying local fallback...")
        # Try local download directory
        local_drive = os.path.expanduser(f"~/Google Drive/My Drive/{args.drive_folder}")
        if os.path.isdir(local_drive):
            print(f"  Found local sync: {local_drive}")
            all_dfs = []
            for fname in sorted(os.listdir(local_drive)):
                if fname.startswith("zscore_") and fname.endswith(".csv"):
                    chunk = pd.read_csv(os.path.join(local_drive, fname))
                    all_dfs.append(chunk)
            if all_dfs:
                ts_df = pd.concat(all_dfs, ignore_index=True)
                print(f"  Loaded {len(ts_df)} rows from {len(all_dfs)} files")
            else:
                print("  No zscore CSV files found.")
                return
        else:
            print("  No local Drive sync found either. Exiting.")
            return
    else:
        folder_id = folder_items[0]["id"]
        file_query = f"'{folder_id}' in parents and name contains 'zscore_'"
        files = []
        page_token = None
        while True:
            resp = (
                service.files()
                .list(q=file_query, spaces="drive", pageToken=page_token, pageSize=100)
                .execute()
            )
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        print(f"  Found {len(files)} files in Drive")
        all_dfs = []
        for fi in files:
            content = service.files().get_media(fileId=fi["id"]).execute()
            chunk = pd.read_csv(io.BytesIO(content))
            all_dfs.append(chunk)
        ts_df = pd.concat(all_dfs, ignore_index=True)
        print(f"  Downloaded {len(ts_df)} rows")

    # ── Step 5: Estimate damage date ─────────────────────────────────────
    print("Estimating damage dates...")
    ts_df["date"] = pd.to_datetime(ts_df["date_millis"], unit="ms")
    ts_df["z_max"] = ts_df[["VV", "VH"]].abs().max(axis=1)
    ts_df = ts_df.sort_values(["latitude", "longitude", "date"])

    # For each building, find the first date where z_max > z_crit
    # in two consecutive acquisitions (to avoid single-image speckle spikes)
    results = []
    for (lat, lon), grp in ts_df.groupby(["latitude", "longitude"]):
        grp = grp.sort_values("date")
        above = grp["z_max"] > args.z_crit
        # Two consecutive exceedances
        consecutive = above & above.shift(1, fill_value=False)
        if consecutive.any():
            first_idx = consecutive.idxmax()
            # Use the date of the first exceedance (one row before the consecutive one)
            prev_idx = grp.index[grp.index.get_loc(first_idx) - 1]
            damage_date = grp.loc[prev_idx, "date"]
            peak_z = grp.loc[first_idx, "z_max"]
        elif above.any():
            # Single exceedance only — use it but flag as uncertain
            first_idx = above.idxmax()
            damage_date = grp.loc[first_idx, "date"]
            peak_z = grp.loc[first_idx, "z_max"]
        else:
            damage_date = pd.NaT
            peak_z = grp["z_max"].max()

        results.append(
            {
                "latitude": lat,
                "longitude": lon,
                "estimated_damage_date": damage_date,
                "peak_z": peak_z,
                "n_exceedances": above.sum(),
                "n_acquisitions": len(grp),
            }
        )

    result_df = pd.DataFrame(results)

    # Merge back with original damage stats
    orig = pd.read_csv(args.csv)
    orig = orig[orig["T_statistic"] > args.threshold]
    merged = orig.merge(result_df, on=["latitude", "longitude"], how="left")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(f"  {len(merged)} buildings")
    print(f"  {merged['estimated_damage_date'].notna().sum()} with estimated damage date")
    print(
        f"  Date range: {merged['estimated_damage_date'].min()} to "
        f"{merged['estimated_damage_date'].max()}"
    )

    # Also save the raw z-score time series
    ts_output = args.output.replace(".csv", "_timeseries.csv")
    ts_df.to_csv(ts_output, index=False)
    print(f"  Raw time series: {ts_output} ({len(ts_df)} rows)")


if __name__ == "__main__":
    main()

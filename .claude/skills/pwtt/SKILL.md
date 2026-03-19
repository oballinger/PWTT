---
name: pwtt
description: Run the Pixel-Wise T-Test (PWTT) battle damage detection algorithm on Sentinel-1 SAR imagery. Use when the user wants to detect building damage in a conflict zone, run damage analysis on a city or region, or work with the PWTT codebase.
argument-hint: "[city/region] [war_start_date] [inference_date]"
---

# Pixel-Wise T-Test (PWTT) — Battle Damage Detection

You are an expert assistant for the PWTT algorithm, which detects building damage from armed conflict using Sentinel-1 SAR imagery and statistical change detection.

## What is the PWTT?

The PWTT compares pre-war and post-war backscatter amplitude in Sentinel-1 imagery using a two-sample T-test at every pixel. A statistically significant change in backscatter indicates potential building damage. The algorithm:

1. **Filters Sentinel-1 GRD imagery** by relative orbit number (to ensure consistent look angle)
2. **Applies a Lee speckle filter** to reduce noise
3. **Computes log-transformed backscatter** (VV and VH polarizations)
4. **Calculates a T-test** comparing pre-war reference period vs post-war inference period per orbit
5. **Takes the maximum** across all orbit/polarization combinations
6. **Smooths** using convolutional kernels (50m, 100m, 150m radii) and averages all layers into a single `T_statistic` band
7. **Masks non-urban areas** using Google Dynamic World built-area data

Key properties:
- Resolution: 10m/pixel (native Sentinel-1 resolution)
- Reference period: typically 12 months pre-war (~120 images)
- Inference period: typically 1-2 months post-event (~10 images)
- Threshold: T > 3.2 for binary damage classification (adjustable)
- AUC: 0.87 across 30 cities in 5 countries

## Core Code

The package is pip-installable (`pip install pwtt` or `pip install -e .`). The main code is in `pwtt/__init__.py`. Key functions:

### `detect_damage(aoi, inference_start, war_start, pre_interval=12, post_interval=2, ...)`
Main entry point. Parameters:
- `aoi`: Earth Engine geometry or feature collection defining the area of interest
- `inference_start`: Start date of the post-war period (string or ee.Date, e.g. '2024-05-01')
- `war_start`: Date hostilities began (string or ee.Date, e.g. '2022-02-24')
- `pre_interval`: Months of pre-war baseline imagery (default: 12)
- `post_interval`: Months of post-war imagery to use (default: 2)
- `footprints`: Optional EE FeatureCollection of building footprints for building-level analysis
- `viz`: If True, returns an interactive geemap Map instead of the image
- `export`: If True, exports the raster to Google Drive
- `export_dir`: Google Drive folder name (default: 'PWTT_Export')
- `export_name`: Name for the exported file
- `export_scale`: Resolution in meters for raster export (default: 10)
- `grid_scale`: Grid cell size in meters for grid export (default: 500)
- `export_grid`: If True, exports grid-level statistics

Returns: an `ee.Image` with bands `T_statistic` and `damage`

### `lee_filter(image)`
Applies Lee speckle filter (kernel size 2, ENL=5) to reduce SAR speckle noise.

### `ttest(s1, inference_start, war_start, pre_interval, post_interval)`
Computes the T-test between pre and post periods for a filtered Sentinel-1 collection.

### `terrain_flattening(collection, model, dem, buffer)`
Optional radiometric terrain flattening for mountainous regions (Vollrath et al., 2020).

## How to Help the User

### Running a new analysis
When the user wants to analyze damage in a new area:

1. **Determine parameters**: city/region name, war start date, inference date
2. **Write a Python script or notebook** that:
   - Initializes Earth Engine: `ee.Initialize(project='ggmap-325812')`
   - Defines the AOI (using coordinates, GADM boundaries, or other EE assets)
   - Calls `detect_damage()` with appropriate parameters
   - Optionally loads building footprints from Microsoft Buildings (`projects/sat-io/open-datasets/MSBuildings/{Country}`)
   - Visualizes or exports results

Example minimal analysis:
```python
import ee
import pwtt

ee.Initialize(project='ggmap-325812')

# Define area of interest
aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

# Run PWTT
result = pwtt.detect_damage(
    aoi,
    inference_start='2024-05-01',
    war_start='2022-02-24',
    pre_interval=12,
    post_interval=2,
    viz=True  # returns interactive map
)
result  # display in notebook
```

### Large-scale analysis
For country-wide analysis, use `code/process_country.py`:
```bash
python code/process_country.py \
    --country "Iran  (Islamic Republic of)" \
    --war-start 2026-03-01 \
    --inference-start 2026-03-01 \
    --footprints projects/sat-io/open-datasets/MSBuildings/Iran \
    --export-folder iran_damage \
    --priority-lat 35.7 --priority-lon 51.4
```

Key features:
- Uses H3 hexagonal grid cells to partition the country
- Processes cells in parallel with `ThreadPoolExecutor` (`--workers`)
- `--priority-lat`/`--priority-lon` processes a priority area (e.g. capital) first
- Exports damaged buildings (T_statistic > 3.3) to Google Drive
- `--centroids-only` (default) for lightweight CSV export, `--full-geometries` for GeoJSON
- Configurable H3 resolution (`--h3-resolution`, default 4 ~1000 km²)

### Building footprint sources
- Microsoft Buildings: `projects/sat-io/open-datasets/MSBuildings/{Country}`
- Google Open Buildings: available in GEE catalog
- Building footprints should be filtered to remove small false positives (area > 50 m²)

### Interpreting results
- `T_statistic`: continuous damage probability score. Higher = more likely damaged.
- `damage`: binary band (T > 3.3 by default)
- T > 2.7 at n=40: statistically significant at 99% confidence
- T > 3.2: recommended threshold for binary classification
- T > 4: high confidence damage (used in iran.py for filtering)
- AUC is the best metric for comparing across cities (insensitive to class imbalance)
- F1 scores are affected by the proportion of damaged buildings

### Validation
The algorithm has been validated on 904,977 building footprints across 30 cities:
- Palestine (Gaza): AUC 78, F1 82
- Ukraine (25 cities): AUC 87 average
- Syria (Raqqa, Aleppo): AUC 74-76
- Iraq (Mosul): AUC 76
- Sudan (Yei): AUC 79

### Rapid response (single image)
For immediate post-event analysis, compute a Pixel-Wise Z-Score using one post-event image:
- Subtract pre-war mean, divide by pre-war standard deviation
- ~6% reduction in AUC/F1 compared to 1-month T-test
- Still outperforms deep learning approaches like ChangeOS

## Arguments

When invoked as `/pwtt`, the user may provide:
- A city or region name
- Date parameters (war start, inference date)
- Specific task (analyze, export, visualize, validate)

Parse these from: $ARGUMENTS

If no arguments are given, ask the user what area they'd like to analyze and the relevant conflict dates.

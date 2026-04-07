#!/bin/bash
# Nightly Iran damage detection pipeline
# Runs hotelling (which includes Z-test values), then merges and publishes.

set -e

PROJECT_DIR="/Users/ollieballinger/Google Drive/Work/UCL/Research/PWTT"
ASSET_FOLDER="projects/ggmap-325812/assets/iran_hotelling"
MERGED_ASSET="projects/ggmap-325812/assets/iran_damage_merged"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate geo

cd "$PROJECT_DIR"

echo "$(date): Starting nightly Iran pipeline"

# Step 1: Clean asset folder
echo "Cleaning asset folder..."
python -c "
import ee
ee.Initialize(project='ggmap-325812')
folder = '$ASSET_FOLDER'
try:
    assets = ee.data.listAssets({'parent': folder})
    for a in assets.get('assets', []):
        ee.data.deleteAsset(a['id'])
except: pass
try:
    ee.data.createAsset({'type': 'FOLDER'}, folder)
except: pass
print(f'Ready: {folder}')
"

# Step 2: Submit export tasks (hotelling includes Z_statistic and Z_p_value)
echo "Submitting export tasks..."
python code/process_country.py \
    --country "Iran (Islamic Republic of)" \
    --war-start 2026-03-01 \
    --inference-start 2026-03-01 \
    --footprints projects/sat-io/open-datasets/MSBuildings/Iran \
    --export-folder iran_hotelling \
    --method hotelling \
    --to-asset "$ASSET_FOLDER" \
    --h3-resolution 4 \
    --lee-mode composite

# Step 3: Wait for all tasks
echo "Waiting for EE tasks to complete..."
python code/wait_for_tasks.py --prefix iran_ --max-age-hours 12

# Step 4: Merge and publish
echo "Merging and publishing..."
python code/merge_and_publish.py \
    --source "$ASSET_FOLDER" \
    --destination "$MERGED_ASSET" \
    --public

echo "$(date): Nightly pipeline complete"

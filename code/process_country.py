"""
Run PWTT damage detection across an entire country using H3 spatial indexing.

Tiles the country into H3 hexagonal cells, runs damage analysis on each cell,
and exports results (damaged building footprints) to Google Drive.

Usage:
    python process_country.py --country "Iran" --war-start 2026-03-01 \
        --inference-start 2026-03-01 --footprints projects/sat-io/open-datasets/MSBuildings/Iran \
        --export-folder iran_damage --priority-lat 35.7 --priority-lon 51.4

    python process_country.py --country "Ukraine" --war-start 2022-02-22 \
        --inference-start 2024-07-01 --footprints projects/sat-io/open-datasets/MSBuildings/Ukraine \
        --export-folder ukraine_damage --priority-lat 50.45 --priority-lon 30.52
"""

import argparse
import ee
import h3
from concurrent.futures import ThreadPoolExecutor, as_completed

from pwtt import detect_damage


def get_country_cells(country_name, h3_resolution, buffer_m=20000, simplify_m=10000):
    """Get H3 cells covering a country using GAUL boundaries from GEE."""
    boundary = ee.FeatureCollection('FAO/GAUL/2015/level0') \
        .filter(ee.Filter.eq('ADM0_NAME', country_name))
    geojson = boundary.geometry().buffer(buffer_m).simplify(simplify_m).getInfo()

    if geojson['type'] == 'GeometryCollection':
        cells = set()
        for geom in geojson['geometries']:
            if geom['type'] in ('Polygon', 'MultiPolygon'):
                cells |= set(h3.geo_to_cells(geom, h3_resolution))
    else:
        cells = set(h3.geo_to_cells(geojson, h3_resolution))
    return cells


def order_cells(all_cells, priority_lat=None, priority_lon=None, h3_resolution=4, priority_ring=2):
    """Order cells so priority area (e.g. capital) is processed first."""
    if priority_lat is not None and priority_lon is not None:
        center = h3.latlng_to_cell(priority_lat, priority_lon, h3_resolution)
        priority_cells = set(h3.grid_disk(center, priority_ring)) & all_cells
        remaining = all_cells - priority_cells
        return sorted(priority_cells) + sorted(remaining), priority_cells
    return sorted(all_cells), set()


def process_cell(h3_index, footprints_asset, war_start, inference_start,
                 pre_interval, post_interval, export_folder, country_prefix,
                 centroids_only):
    """Process a single H3 cell: run PWTT and export damaged buildings."""
    boundary = h3.cell_to_boundary(h3_index)
    coords = [[lon, lat] for lat, lon in boundary]
    coords.append(coords[0])
    aoi = ee.Geometry.Polygon([coords])

    footprints = ee.FeatureCollection(footprints_asset) \
        .filterBounds(aoi) \
        .map(lambda feat: feat.set('area', feat.geometry().simplify(10).area(10))) \
        .filter(ee.Filter.gt('area', 50))

    if footprints.limit(1).size().getInfo() == 0:
        return f"  {h3_index}: skipped (no buildings)"

    image = detect_damage(
        aoi,
        inference_start=inference_start,
        war_start=war_start,
        pre_interval=pre_interval,
        post_interval=post_interval,
    )

    fp_sample = image.reduceRegions(
        collection=footprints,
        reducer=ee.Reducer.mean(),
        scale=10,
        tileScale=8,
    )

    damaged = fp_sample.filter(ee.Filter.gt('T_statistic', 3))

    if centroids_only:
        def to_centroid(f):
            centroid = f.geometry().centroid(1).coordinates()
            return f.setGeometry(None) \
                .set('longitude', centroid.get(0)) \
                .set('latitude', centroid.get(1))
        damaged = damaged.map(to_centroid).select(
            propertySelectors=['T_statistic', 'area', 'damage', 'p_value', 'post_n', 'longitude', 'latitude']
        )

    task = ee.batch.Export.table.toDrive(
        collection=damaged,
        description=f'{country_prefix}_{h3_index}_damaged',
        folder=export_folder,
        fileFormat='CSV' if centroids_only else 'GEOJSON',
    )
    task.start()
    return f"  {h3_index}: export task submitted"


def main():
    parser = argparse.ArgumentParser(
        description='Run PWTT damage detection across a country using H3 tiling.')
    parser.add_argument('--country', required=True,
                        help='Country name as it appears in FAO/GAUL (e.g. "Iran  (Islamic Republic of)")')
    parser.add_argument('--war-start', required=True, help='War start date (YYYY-MM-DD)')
    parser.add_argument('--inference-start', required=True, help='Inference start date (YYYY-MM-DD)')
    parser.add_argument('--footprints', required=True,
                        help='EE asset path for building footprints')
    parser.add_argument('--export-folder', default='pwtt_export',
                        help='Google Drive folder for exports')
    parser.add_argument('--pre-interval', type=int, default=12,
                        help='Months of pre-war baseline (default: 12)')
    parser.add_argument('--post-interval', type=int, default=2,
                        help='Months of post-war imagery (default: 2)')
    parser.add_argument('--h3-resolution', type=int, default=4,
                        help='H3 resolution for tiling (default: 4, ~1000 km²)')
    parser.add_argument('--priority-lat', type=float, default=None,
                        help='Latitude of priority area to process first')
    parser.add_argument('--priority-lon', type=float, default=None,
                        help='Longitude of priority area to process first')
    parser.add_argument('--workers', type=int, default=15,
                        help='Number of parallel threads (default: 15)')
    parser.add_argument('--centroids-only', action='store_true', default=True,
                        help='Export centroids as CSV instead of full geometries (default: True)')
    parser.add_argument('--full-geometries', action='store_true',
                        help='Export full geometries as GeoJSON')
    parser.add_argument('--project', default='ggmap-325812',
                        help='GEE cloud project ID')
    args = parser.parse_args()

    centroids_only = not args.full_geometries

    ee.Initialize(project=args.project)

    print(f"Tiling {args.country} with H3 resolution {args.h3_resolution}...")
    all_cells = get_country_cells(args.country, args.h3_resolution)
    ordered, priority_cells = order_cells(
        all_cells, args.priority_lat, args.priority_lon, args.h3_resolution)

    n_priority = len(priority_cells)
    n_total = len(ordered)
    print(f"Total H3 cells: {n_total} ({n_priority} priority, {n_total - n_priority} remaining)")

    country_prefix = args.country.split()[0].lower()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_cell, h3_index, args.footprints, args.war_start,
                args.inference_start, args.pre_interval, args.post_interval,
                args.export_folder, country_prefix, centroids_only
            ): (i, h3_index)
            for i, h3_index in enumerate(ordered)
        }
        for future in as_completed(futures):
            i, h3_index = futures[future]
            label = "priority" if h3_index in priority_cells else "standard"
            try:
                result = future.result()
                print(f"[{i+1}/{n_total}] [{label}] {result}")
            except Exception as e:
                print(f"[{i+1}/{n_total}] [{label}] {h3_index} ERROR: {e}")


if __name__ == '__main__':
    main()

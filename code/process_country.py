"""
Run PWTT damage detection across an entire country using H3 spatial indexing.

Builds a single server-side computation graph for the entire country, then
partitions the export into H3 chunks. This eliminates per-cell getInfo() calls
and shares the computation graph across all chunks.

Usage:
    python process_country.py --country "Palestine" --war-start 2023-10-10 \
        --inference-start 2026-01-01 \
        --footprints projects/sat-io/open-datasets/MSBuildings/Gaza_Strip \
        --export-folder gaza_damage

    python process_country.py --country "Ukraine" --war-start 2022-02-22 \
        --inference-start 2024-07-01 \
        --footprints projects/sat-io/open-datasets/MSBuildings/Ukraine \
        --export-folder ukraine_damage --priority-lat 50.45 --priority-lon 30.52
"""

import argparse
import ee
import h3

from pwtt import detect_damage


def get_country_cells(country_geojson, h3_resolution):
    """Get H3 cells covering a country geometry."""
    if country_geojson['type'] == 'GeometryCollection':
        cells = set()
        for geom in country_geojson['geometries']:
            if geom['type'] in ('Polygon', 'MultiPolygon'):
                cells |= set(h3.geo_to_cells(geom, h3_resolution))
    else:
        cells = set(h3.geo_to_cells(country_geojson, h3_resolution))
    return cells


def order_cells(all_cells, priority_lat=None, priority_lon=None, h3_resolution=4, priority_ring=2):
    """Order cells so priority area (e.g. capital) is processed first."""
    if priority_lat is not None and priority_lon is not None:
        center = h3.latlng_to_cell(priority_lat, priority_lon, h3_resolution)
        priority_cells = set(h3.grid_disk(center, priority_ring)) & all_cells
        remaining = all_cells - priority_cells
        return sorted(priority_cells) + sorted(remaining), priority_cells
    return sorted(all_cells), set()


def h3_to_ee_geometry(h3_index):
    """Convert an H3 cell index to an ee.Geometry.Polygon."""
    boundary = h3.cell_to_boundary(h3_index)
    coords = [[lon, lat] for lat, lon in boundary]
    coords.append(coords[0])
    return ee.Geometry.Polygon([coords])


def main():
    parser = argparse.ArgumentParser(
        description='Run PWTT damage detection across a country using H3 tiling.')
    parser.add_argument('--country', required=True,
                        help='Country name as it appears in FAO/GAUL')
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
    parser.add_argument('--centroids-only', action='store_true', default=True,
                        help='Export centroids as CSV instead of full geometries (default: True)')
    parser.add_argument('--full-geometries', action='store_true',
                        help='Export full geometries as GeoJSON')
    parser.add_argument('--project', default='ggmap-325812',
                        help='GEE cloud project ID')
    args = parser.parse_args()

    centroids_only = not args.full_geometries

    ee.Initialize(project=args.project)

    # 1. Get country boundary
    print(f"Loading boundary for {args.country}...")
    country = ee.FeatureCollection('FAO/GAUL/2015/level0') \
        .filter(ee.Filter.eq('ADM0_NAME', args.country))
    country_geom = country.geometry().buffer(20000).simplify(10000)

    # 2. Single detect_damage call for entire country (no clipping — lazy evaluation)
    print("Building computation graph...")
    image = detect_damage(
        aoi=country_geom,
        inference_start=args.inference_start,
        war_start=args.war_start,
        pre_interval=args.pre_interval,
        post_interval=args.post_interval,
        clip=False,
    )

    # 3. Get H3 cells for export chunking (single getInfo call)
    print(f"Tiling with H3 resolution {args.h3_resolution}...")
    geojson = country_geom.getInfo()
    all_cells = get_country_cells(geojson, args.h3_resolution)
    ordered, priority_cells = order_cells(
        all_cells, args.priority_lat, args.priority_lon, args.h3_resolution)

    n_priority = len(priority_cells)
    n_total = len(ordered)
    print(f"Total H3 cells: {n_total} ({n_priority} priority, {n_total - n_priority} remaining)")

    country_prefix = args.country.split()[0].lower()

    # 4. Submit export tasks (no getInfo, no blocking, no threads needed)
    def to_centroid(f):
        centroid = f.geometry().centroid(1).coordinates()
        return f.setGeometry(None) \
            .set('longitude', centroid.get(0)) \
            .set('latitude', centroid.get(1))

    for i, h3_index in enumerate(ordered):
        chunk_geom = h3_to_ee_geometry(h3_index)

        footprints = ee.FeatureCollection(args.footprints) \
            .filterBounds(chunk_geom) \
            .map(lambda feat: feat.set('area', feat.geometry().simplify(10).area(10))) \
            .filter(ee.Filter.gt('area', 50))

        result = image.reduceRegions(
            collection=footprints,
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=8,
        )

        damaged = result.filter(ee.Filter.gt('T_statistic', 3.3))

        if centroids_only:
            damaged = damaged.map(to_centroid).select(
                propertySelectors=['T_statistic', 'area', 'damage', 'p_value', 'n_post', 'longitude', 'latitude']
            )

        task = ee.batch.Export.table.toDrive(
            collection=damaged,
            description=f'{country_prefix}_{h3_index}_damaged',
            folder=args.export_folder,
            fileFormat='CSV' if centroids_only else 'GEOJSON',
        )
        task.start()

        label = "priority" if h3_index in priority_cells else "standard"
        print(f"[{i+1}/{n_total}] [{label}] {h3_index}: export task submitted")

    print(f"\nAll {n_total} export tasks submitted. Monitor at https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    main()

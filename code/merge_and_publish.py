"""
Merge per-region EE assets into a single FeatureCollection, optionally
joining ztest results alongside, and make it public.

Usage:
    # Hotelling only
    python merge_and_publish.py \
        --source projects/ggmap-325812/assets/iran_hotelling \
        --destination projects/ggmap-325812/assets/iran_damage_merged \
        --public

    # Hotelling + ztest join
    python merge_and_publish.py \
        --source projects/ggmap-325812/assets/iran_hotelling \
        --ztest-source projects/ggmap-325812/assets/iran_ztest_nightly \
        --destination projects/ggmap-325812/assets/iran_damage_merged \
        --public
"""

import argparse
import time
import ee


def merge_folder(folder):
    """Merge all assets in a folder into a single FeatureCollection."""
    assets_list = ee.data.listAssets({'parent': folder})
    asset_ids = [a['id'] for a in assets_list.get('assets', [])]
    print(f"  Found {len(asset_ids)} assets in {folder}")
    if not asset_ids:
        return None
    return ee.FeatureCollection(
        [ee.FeatureCollection(aid) for aid in asset_ids]
    ).flatten()


def wait_for_task(task):
    """Poll until an export task finishes."""
    while task.status()['state'] in ('READY', 'RUNNING', 'UNSUBMITTED'):
        print(f"  Status: {task.status()['state']}")
        time.sleep(60)
    return task.status()['state']


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-region EE assets into a single public FeatureCollection.')
    parser.add_argument('--source', required=True,
                        help='Asset folder containing primary (hotelling) per-region tables')
    parser.add_argument('--ztest-source', default=None,
                        help='Asset folder containing ztest per-region tables (optional)')
    parser.add_argument('--destination', required=True,
                        help='Output asset ID for merged table')
    parser.add_argument('--public', action='store_true',
                        help='Set merged asset ACL to all_users_can_read')
    parser.add_argument('--project', default='ggmap-325812',
                        help='GEE cloud project ID')
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    # Merge primary (hotelling) assets
    print("Merging primary assets...")
    merged = merge_folder(args.source)
    if merged is None:
        print("No primary assets found. Exiting.")
        return

    # Optionally join ztest results
    if args.ztest_source:
        print("Merging ztest assets...")
        ztest_merged = merge_folder(args.ztest_source)

        if ztest_merged is not None:
            print("Joining ztest values to primary results...")

            # Rename ztest columns to avoid collision
            def rename_ztest(f):
                return ee.Feature(f.geometry(), {
                    'Z_statistic': f.get('T_statistic'),
                    'Z_p_value': f.get('p_value'),
                    'z_longitude': f.get('longitude'),
                    'z_latitude': f.get('latitude'),
                })

            ztest_renamed = ztest_merged.map(rename_ztest)

            # Spatial join: match on same lat/lon (same building centroids)
            join_filter = ee.Filter.And(
                ee.Filter.maxDifference(
                    difference=0.0001,
                    leftField='latitude',
                    rightField='z_latitude',
                ),
                ee.Filter.maxDifference(
                    difference=0.0001,
                    leftField='longitude',
                    rightField='z_longitude',
                ),
            )

            join = ee.Join.inner('primary', 'ztest')
            joined = join.apply(merged, ztest_renamed, join_filter)

            # Flatten joined pairs into single features
            def flatten_pair(pair):
                primary = ee.Feature(pair.get('primary'))
                ztest = ee.Feature(pair.get('ztest'))
                return primary.set({
                    'Z_statistic': ztest.get('Z_statistic'),
                    'Z_p_value': ztest.get('Z_p_value'),
                })

            merged = ee.FeatureCollection(joined.map(flatten_pair))
            print("Join complete (server-side)")
        else:
            print("No ztest assets found, exporting primary only")

    # Delete old destination if it exists
    try:
        ee.data.deleteAsset(args.destination)
        print(f"Deleted existing asset: {args.destination}")
    except ee.EEException:
        pass

    # Export merged collection to asset
    task = ee.batch.Export.table.toAsset(
        collection=merged,
        description='merge_' + args.destination.split('/')[-1],
        assetId=args.destination,
    )
    task.start()
    print(f"Export task started: {task.id}")

    state = wait_for_task(task)
    if state == 'COMPLETED':
        print("Merge export completed successfully")
    else:
        print(f"Merge export failed: {task.status()}")
        return

    # Set public ACL
    if args.public:
        ee.data.setAssetAcl(args.destination, {'all_users_can_read': True})
        print(f"Asset set to public: {args.destination}")

    print("Done.")


if __name__ == '__main__':
    main()

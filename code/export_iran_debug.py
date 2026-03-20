"""Export PWTT results for 3 problematic Iran locations for debugging."""

import ee
from pwtt import detect_damage

ee.Initialize(project='ggmap-325812')

WAR_START = '2026-03-01'
PRE_INTERVAL = 12
POST_INTERVAL = 2
FOOTPRINTS = 'projects/sat-io/open-datasets/MSBuildings/Iran'
EXPORT_FOLDER = 'iran_debug'

# The 3 locations: (lon, lat, name)
LOCATIONS = [
    (55.679, 29.4449, 'loc_55_29'),
    (49.5865, 37.2809, 'loc_49_37'),
    (51.37, 35.70, 'tehran'),
]

BUFFER_M = 15000  # 15km radius around each point


def to_centroid(f):
    centroid = f.geometry().centroid(1).coordinates()
    return f.setGeometry(None) \
        .set('longitude', centroid.get(0)) \
        .set('latitude', centroid.get(1))


for lon, lat, name in LOCATIONS:
    print(f"\n--- {name} ({lon}, {lat}) ---")

    aoi = ee.Geometry.Point(lon, lat).buffer(BUFFER_M)

    image = detect_damage(
        aoi=aoi,
        inference_start=WAR_START,
        war_start=WAR_START,
        pre_interval=PRE_INTERVAL,
        post_interval=POST_INTERVAL,
        clip=False,
    )

    # Same quality mask as process_country.py
    quality_mask = image.select('T_statistic').gt(3.0).And(
        image.select('n_post').gte(3)
    )
    image = image.updateMask(quality_mask)

    footprints = ee.FeatureCollection(FOOTPRINTS) \
        .filterBounds(aoi) \
        .map(lambda feat: feat.set('area', feat.geometry().simplify(10).area(10))) \
        .filter(ee.Filter.gt('area', 50))

    result = image.reduceRegions(
        collection=footprints,
        reducer=ee.Reducer.mean(),
        scale=10,
        tileScale=8,
    )

    damaged = result.filter(ee.Filter.And(
        ee.Filter.gt('T_statistic', 3.3),
        ee.Filter.lt('p_value', 0.05),
        ee.Filter.gte('n_post', 3),
    ))

    damaged = damaged.map(to_centroid).select(
        propertySelectors=['T_statistic', 'area', 'damage', 'p_value',
                           'n_pre', 'n_post', 'longitude', 'latitude']
    )

    task = ee.batch.Export.table.toDrive(
        collection=damaged,
        description=f'iran_debug_{name}',
        folder=EXPORT_FOLDER,
        fileFormat='CSV',
    )
    task.start()
    print(f"  Export submitted: iran_debug_{name}")

    # Also export ALL footprints (not just damaged) so we can inspect
    all_result = result.filter(ee.Filter.notNull(['T_statistic']))
    all_result = all_result.map(to_centroid).select(
        propertySelectors=['T_statistic', 'area', 'damage', 'p_value',
                           'n_pre', 'n_post', 'longitude', 'latitude']
    )

    task_all = ee.batch.Export.table.toDrive(
        collection=all_result,
        description=f'iran_debug_{name}_all',
        folder=EXPORT_FOLDER,
        fileFormat='CSV',
    )
    task_all.start()
    print(f"  Export submitted: iran_debug_{name}_all (all footprints)")

print(f"\nAll exports submitted. Monitor at https://code.earthengine.google.com/tasks")

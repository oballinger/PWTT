"""
Z-test for Iran debug locations: compare the single latest S1 image
against the pre-war baseline distribution using detect_damage(method='ztest').
"""

import ee
from pwtt import detect_damage

ee.Initialize(project='ggmap-325812')

WAR_START = '2026-03-01'
PRE_INTERVAL = 12
FOOTPRINTS = 'projects/sat-io/open-datasets/MSBuildings/Iran'
EXPORT_FOLDER = 'iran_ztest'
BUFFER_M = 15000

LOCATIONS = [
    (55.679, 29.4449, 'loc_55_29'),
    (49.5865, 37.2809, 'loc_49_37'),
    (51.37, 35.70, 'tehran'),
]

PROPS = ['T_statistic', 'area', 'damage', 'p_value',
         'n_pre', 'n_post', 'longitude', 'latitude']


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
        method='ztest',
    )

    # Quality mask
    quality_mask = image.select('T_statistic').gt(3.0).And(
        image.select('n_pre').gte(3)
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

    # Export all footprints with scores
    all_result = result.filter(ee.Filter.notNull(['T_statistic']))
    all_result = all_result.map(to_centroid).select(propertySelectors=PROPS)

    task = ee.batch.Export.table.toDrive(
        collection=all_result,
        description=f'iran_ztest_{name}_all',
        folder=EXPORT_FOLDER,
        fileFormat='CSV',
    )
    task.start()
    print(f"  Export submitted: iran_ztest_{name}_all")

    # Export just damaged
    damaged = result.filter(ee.Filter.And(
        ee.Filter.gt('T_statistic', 3.3),
        ee.Filter.lt('p_value', 0.05),
        ee.Filter.gte('n_pre', 3),
    ))
    damaged = damaged.map(to_centroid).select(propertySelectors=PROPS)

    task2 = ee.batch.Export.table.toDrive(
        collection=damaged,
        description=f'iran_ztest_{name}',
        folder=EXPORT_FOLDER,
        fileFormat='CSV',
    )
    task2.start()
    print(f"  Export submitted: iran_ztest_{name}")

print(f"\nAll exports submitted. Monitor at https://code.earthengine.google.com/tasks")

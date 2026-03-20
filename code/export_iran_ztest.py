"""
Z-test for Iran debug locations: compare the single latest S1 image
against the pre-war baseline distribution.

z = (x_post - mean_pre) / sd_pre   per pixel, per orbit
"""

import ee
from pwtt import lee_filter, normal_cdf_approx, two_tailed_pvalue

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


def to_centroid(f):
    centroid = f.geometry().centroid(1).coordinates()
    return f.setGeometry(None) \
        .set('longitude', centroid.get(0)) \
        .set('latitude', centroid.get(1))


def ztest_latest(aoi, war_start, pre_interval):
    """Run z-test using the single latest S1 image vs pre-war baseline."""
    war_start = ee.Date(war_start)
    pre_start = war_start.advance(ee.Number(pre_interval).multiply(-1), 'month')

    s1_base = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filterBounds(aoi)

    # Get orbits that cover this AOI in post-war period
    orbits = s1_base.filterDate(war_start, ee.Date('2026-12-31')) \
        .aggregate_array('relativeOrbitNumber_start').distinct()

    def map_orbit(orbit):
        s1 = s1_base.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
            .map(lee_filter).select(['VV', 'VH']) \
            .map(lambda image: image.log())

        # Pre-war baseline stats
        pre = s1.filterDate(pre_start, war_start)
        pre_mean = pre.mean()
        pre_sd = pre.reduce(ee.Reducer.stdDev())
        pre_n = pre.select('VV').count()

        # Latest single image
        latest = s1.filterDate(war_start, ee.Date('2026-12-31')) \
            .sort('system:time_start', False).first()

        # Z-score: (latest - pre_mean) / pre_sd
        z_vv = latest.select('VV').subtract(pre_mean.select('VV')) \
            .divide(pre_sd.select('VV_stdDev')).abs().rename('VV')
        z_vh = latest.select('VH').subtract(pre_mean.select('VH')) \
            .divide(pre_sd.select('VH_stdDev')).abs().rename('VH')

        # p-values from z-scores
        p_vv = two_tailed_pvalue(z_vv).rename('VV_pvalue')
        p_vh = two_tailed_pvalue(z_vh).rename('VH_pvalue')

        # Mask where pre has < 3 images
        valid = pre_n.gte(3)
        z_vv = z_vv.updateMask(valid)
        z_vh = z_vh.updateMask(valid)

        return z_vv.addBands(z_vh).addBands(p_vv).addBands(p_vh) \
            .addBands(pre_n.toFloat().rename('n_pre')) \
            .addBands(ee.Image.constant(1).toFloat().rename('n_post')) \
            .set('orbit', orbit) \
            .set('date', latest.date().format('YYYY-MM-dd'))

    orbit_images = ee.ImageCollection(orbits.map(map_orbit))

    # Combine orbits: max z-score, min p-value, Bonferroni
    z_max = orbit_images.select(['VV', 'VH']).max()
    p_min = orbit_images.select(['VV_pvalue', 'VH_pvalue']).min()
    n_pre = orbit_images.select('n_pre').max()

    max_change = z_max.select('VV').max(z_max.select('VH')).rename('max_change')
    p_value = p_min.select('VV_pvalue').min(p_min.select('VH_pvalue')).rename('p_value')
    n_orbits = orbits.size()
    p_value = p_value.multiply(n_orbits).min(ee.Image.constant(1)).rename('p_value')

    # Urban mask
    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(pre_start, war_start).select('built').mean()

    # Spatial smoothing (same as detect_damage)
    raw_mask = max_change.mask()
    t_smooth = max_change.focalMedian(10, 'gaussian', 'meters')
    t_smooth = t_smooth.updateMask(urban.gt(0.1)).updateMask(raw_mask)

    k50 = t_smooth.convolve(ee.Kernel.circle(50, 'meters', True)).rename('k50')
    k100 = t_smooth.convolve(ee.Kernel.circle(100, 'meters', True)).rename('k100')
    k150 = t_smooth.convolve(ee.Kernel.circle(150, 'meters', True)).rename('k150')

    T_statistic = t_smooth.add(k50).add(k100).add(k150).divide(4).rename('T_statistic')
    T_statistic = T_statistic.updateMask(raw_mask)
    damage = t_smooth.gt(3.3).rename('damage')
    p_value = p_value.updateMask(urban.gt(0.1))

    image = T_statistic.addBands(damage).addBands(p_value) \
        .addBands(n_pre).addBands(ee.Image.constant(1).toFloat().rename('n_post')).toFloat()

    return image, orbit_images


for lon, lat, name in LOCATIONS:
    print(f"\n--- {name} ({lon}, {lat}) ---")
    aoi = ee.Geometry.Point(lon, lat).buffer(BUFFER_M)

    image, orbit_imgs = ztest_latest(aoi, WAR_START, PRE_INTERVAL)

    # Print orbit dates for debugging
    orbit_info = orbit_imgs.aggregate_array('date').getInfo()
    print(f"  Latest image dates per orbit: {orbit_info}")

    # Quality mask (same as process_country)
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
    all_result = all_result.map(to_centroid).select(
        propertySelectors=['T_statistic', 'area', 'damage', 'p_value',
                           'n_pre', 'n_post', 'longitude', 'latitude']
    )

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
    damaged = damaged.map(to_centroid).select(
        propertySelectors=['T_statistic', 'area', 'damage', 'p_value',
                           'n_pre', 'n_post', 'longitude', 'latitude']
    )

    task2 = ee.batch.Export.table.toDrive(
        collection=damaged,
        description=f'iran_ztest_{name}',
        folder=EXPORT_FOLDER,
        fileFormat='CSV',
    )
    task2.start()
    print(f"  Export submitted: iran_ztest_{name}")

print(f"\nAll exports submitted. Monitor at https://code.earthengine.google.com/tasks")

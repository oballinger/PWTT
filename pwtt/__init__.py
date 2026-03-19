"""
Pixel-Wise T-Test (PWTT) - Battle damage detection using Sentinel-1 SAR imagery.

Usage:
    import ee
    import pwtt

    ee.Authenticate()
    ee.Initialize(project='your-project')

    gaza = ee.Geometry.Rectangle([34.21, 31.21, 34.57, 31.60])
    pwtt.detect_damage(aoi=gaza, war_start='2023-10-10', inference_start='2024-07-01', viz=True)
"""

import math
import datetime

import ee
import geemap


__version__ = "0.1.0"


def normal_cdf_approx(x_image):
    """Approximate standard normal CDF for positive x using Abramowitz & Stegun 26.2.17.
    Max error < 7.5e-8. Operates entirely on ee.Image objects (server-side).
    """
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.882575977
    b5 = 1.330274429

    t = ee.Image.constant(1).divide(
        ee.Image.constant(1).add(ee.Image.constant(0.2316419).multiply(x_image))
    )
    phi = x_image.pow(2).multiply(-0.5).exp().divide(math.sqrt(2 * math.pi))

    # Horner's method: poly = t*(b1 + t*(b2 + t*(b3 + t*(b4 + t*b5))))
    poly = t.multiply(
        ee.Image.constant(b1).add(t.multiply(
            ee.Image.constant(b2).add(t.multiply(
                ee.Image.constant(b3).add(t.multiply(
                    ee.Image.constant(b4).add(t.multiply(b5))
                ))
            ))
        ))
    )
    return ee.Image.constant(1).subtract(phi.multiply(poly))


def two_tailed_pvalue(t_image):
    """Compute two-tailed p-value from absolute t-values using normal approximation.
    Valid for large degrees of freedom (df > 30).
    """
    cdf = normal_cdf_approx(t_image)
    return ee.Image.constant(2).multiply(ee.Image.constant(1).subtract(cdf)).max(ee.Image.constant(1e-10))


def lee_filter(image):
    KERNEL_SIZE = 2
    band_names = image.bandNames().remove('angle')

    # S1-GRD images are multilooked 5 times in range
    enl = 5

    # Compute the speckle standard deviation
    eta = 1.0 / enl ** 0.5
    eta = ee.Image.constant(eta)

    # MMSE estimator
    # Neighbourhood mean and variance
    one_img = ee.Image.constant(1)

    reducers = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.variance(),
        sharedInputs=True
    )
    stats = image.select(band_names).reduceNeighborhood(
        reducer=reducers,
        kernel=ee.Kernel.square(KERNEL_SIZE / 2, 'pixels'),
        optimization='window'
    )

    mean_band = band_names.map(lambda band_name: ee.String(band_name).cat('_mean'))
    var_band = band_names.map(lambda band_name: ee.String(band_name).cat('_variance'))

    z_bar = stats.select(mean_band)
    varz = stats.select(var_band)

    # Estimate weight
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(one_img.add(eta.pow(2)))
    b = varx.divide(varz)

    # If b is negative, set it to zero
    new_b = b.where(b.lt(0), 0)

    output = one_img.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(image.select(band_names)))
    output = output.rename(band_names)

    return image.addBands(output, None, True)


def ttest(s1, inference_start, war_start, pre_interval, post_interval):
    # Convert the inference_start date to a date object
    inference_start = ee.Date(inference_start)

    # Filter the image collection to the pre-event period
    pre = s1.filterDate(
        war_start.advance(ee.Number(pre_interval).multiply(-1), "month"),
        war_start
    )

    # Filter the image collection to the post-event period
    post = s1.filterDate(inference_start, inference_start.advance(post_interval, "month"))

    # Calculate the mean, standard deviation, and number of images for the pre-event period
    pre_mean = pre.mean()
    pre_sd = pre.reduce(ee.Reducer.stdDev())
    pre_n = pre.select('VV').count()

    # Calculate the mean, standard deviation, and number of images for the post-event period
    post_mean = post.mean()
    post_sd = post.reduce(ee.Reducer.stdDev())
    post_n = post.select('VV').count()

    # Calculate the pooled standard deviation
    pooled_sd = (pre_sd.pow(2)
                 .multiply(pre_n.subtract(1))
                 .add(post_sd.pow(2).multiply(post_n.subtract(1)))).divide(pre_n.add(post_n).subtract(2)).sqrt()

    # Calculate the denominator of the t-test
    denom = pooled_sd.multiply(
        ee.Image(1).divide(pre_n).add(ee.Image(1).divide(post_n)).sqrt()
    )

    # Calculate the Degrees of Freedom, which is the number of observations minus 2
    df = pre_n.add(post_n).subtract(2)
    # Calculate the t-test using the mean of the pre-event period, the mean of the post-event period, and the pooled standard deviation
    change = post_mean.subtract(pre_mean).divide(denom).abs()

    # Compute two-tailed p-values (normal approx, valid for df > 30)
    p_values = two_tailed_pvalue(change).rename(['VV_pvalue', 'VH_pvalue'])

    # Return t-values, p-values, and sample sizes
    return change.addBands(p_values).addBands(pre_n.toFloat().rename('n_pre')).addBands(post_n.toFloat().rename('n_post'))


def detect_damage(aoi, inference_start, war_start, pre_interval=12, post_interval=2, footprints=None, viz=False, export=False, export_dir='PWTT_Export', export_name=None, export_scale=10, grid_scale=500, export_grid=False):
    inference_start = ee.Date(inference_start)
    war_start = ee.Date(war_start)

    orbits = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filterBounds(aoi) \
        .filterDate(inference_start, inference_start.advance(post_interval, 'months')) \
        .aggregate_array('relativeOrbitNumber_start') \
        .distinct()

    def map_orbit(orbit):
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
            .map(lee_filter) \
            .select(['VV', 'VH'])\
            .map(lambda image: image.log())\
            .filterBounds(aoi) \

        image = ttest(s1, inference_start, war_start, pre_interval, post_interval)
        return image

    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(
        war_start.advance(-1 * pre_interval, 'months'), war_start).select('built').mean()

    orbit_images = ee.ImageCollection(orbits.map(map_orbit))
    # Max t-value across orbits; min p-value across orbits; max sample sizes
    t_max = orbit_images.select(['VV', 'VH']).max()
    p_min = orbit_images.select(['VV_pvalue', 'VH_pvalue']).min()
    n_pre = orbit_images.select('n_pre').max()
    n_post = orbit_images.select('n_post').max()
    image = t_max.addBands(p_min)

    # Combine polarizations: max t-value, min p-value
    max_change = image.select('VV').max(image.select('VH')).rename('max_change')
    p_value = image.select('VV_pvalue').min(image.select('VH_pvalue')).rename('p_value')
    image = max_change.addBands(p_value)

    # Bonferroni correction: multiply p-value by number of orbits, cap at 1
    n_orbits = orbits.size()
    p_value = p_value.multiply(n_orbits).min(ee.Image.constant(1)).rename('p_value')

    # Spatial smoothing applies only to t-values
    t_smooth = max_change.focalMedian(10, 'gaussian', 'meters').clip(aoi).updateMask(urban.gt(0.1))
    k50 = t_smooth.convolve(ee.Kernel.circle(50, 'meters', True)).rename('k50')
    k100 = t_smooth.convolve(ee.Kernel.circle(100, 'meters', True)).rename('k100')
    k150 = t_smooth.convolve(ee.Kernel.circle(150, 'meters', True)).rename('k150')

    damage = t_smooth.gt(3.3).rename('damage')
    T_statistic = (t_smooth.add(k50).add(k100).add(k150)).divide(4).rename('T_statistic')

    # Mask p-values with urban mask
    p_value = p_value.clip(aoi).updateMask(urban.gt(0.1))

    image = T_statistic.addBands(damage).addBands(p_value).addBands(n_pre).addBands(n_post).toFloat()
    image = image.clip(aoi)

    if export_grid:
        grid = aoi.geometry().bounds().coveringGrid('EPSG:3857', grid_scale)
        grid = image.reduceRegions({
            'collection': grid,
            'reducer': ee.Reducer.mean(),
            'scale': 10,
            'tileScale': 8,
        })
        task_grid = ee.batch.Export.table.toDrive(
            collection=grid,
            description=export_name + '_grid',
            folder=export_dir,
            fileFormat='CSV'
        )

    if viz:
        Map = geemap.Map()
        Map.add_basemap('SATELLITE')
        Map.addLayer(image.select('T_statistic'), {'min': 3, 'max': 5, 'opacity': 0.5, 'palette': ["yellow", "red", "purple"]}, "T-test")
        Map.centerObject(aoi)
        return Map

    if type(footprints) != type(None):
        fc = ee.FeatureCollection(footprints).filterBounds(aoi)
        fp = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=8,
        )

        task_fp = ee.batch.Export.table.toDrive(
            collection=fp,
            description=export_name,
            folder=export_dir,
            fileFormat='GEOJSON'
        )
        task_fp.start()

    if export:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=export_name,
            folder=export_dir,
            scale=export_scale,
            maxPixels=1e13,
        )
        task.start()
    return image


def terrain_flattening(collection, TERRAIN_FLATTENING_MODEL, DEM, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER):
    '''
    Terrain Flattening

    Vollrath, A., Mullissa, A., & Reiche, J. (2020). Angular-Based Radiometric Slope Correction for Sentinel-1 on Google Earth Engine.
    Remote Sensing, 12(11), [1867]. https://doi.org/10.3390/rs12111867
    '''
    ninetyRad = ee.Image.constant(90).multiply(3.14159265359 / 180)

    def volumetric_model_SCF(theta_iRad, alpha_rRad):
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = alpha_azRad.cos().multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos())
        return nominator.divide(denominator)

    def erode(image, distance):
        d = (image.Not().unmask(1)
             .fastDistanceTransform(30).sqrt()
             .multiply(ee.Image.pixelArea().sqrt()))
        return image.updateMask(d.gt(distance))

    def masking(alpha_rRad, theta_iRad, buffer):
        layover = alpha_rRad.lt(theta_iRad).rename('layover')
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
        mask = layover.And(shadow)
        if buffer > 0:
            mask = erode(mask, buffer)
        return mask.rename('no_data_mask')

    def correct(image):
        bandNames = image.bandNames()
        geom = image.geometry()
        proj = image.select(1).projection()

        elevation = DEM.resample('bilinear').reproject({
            'crs': proj,
            'scale': 10
        }).clip(geom)

        heading = ee.Terrain.aspect(image.select('angle'))\
            .reduceRegion(ee.Reducer.mean(), image.geometry(), 1000)\
            .get('aspect')

        heading = ee.Number(heading).where(ee.Number(heading).gt(180), ee.Number(heading).subtract(360))
        theta_iRad = image.select('angle').multiply(3.14159265359 / 180)
        phi_iRad = ee.Image.constant(heading).multiply(3.14159265359 / 180)

        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(3.14159265359 / 180)

        aspect = ee.Terrain.aspect(elevation).select('aspect').clip(geom)
        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)
        phi_sRad = aspect.updateMask(aspect.lte(180)).unmask().add(aspect_minus.unmask())\
            .multiply(-1).multiply(3.14159265359 / 180)

        phi_rRad = phi_iRad.subtract(phi_sRad)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()
        theta_liaRad = (alpha_azRad.cos().multiply((theta_iRad.subtract(alpha_rRad)).cos())).acos()
        theta_liaDeg = theta_liaRad.multiply(180 / 3.14159265359)

        gamma0 = image.divide(theta_iRad.cos())

        if TERRAIN_FLATTENING_MODEL == 'VOLUME':
            scf = volumetric_model_SCF(theta_iRad, alpha_rRad)
        elif TERRAIN_FLATTENING_MODEL == 'DIRECT':
            scf = direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        gamma0_flat = gamma0.multiply(scf)
        mask = masking(alpha_rRad, theta_iRad, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER)

        output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)
        output = ee.Image(output).addBands(image.select('angle'), None, True)

        return output.set('system:time_start', image.get('system:time_start'))

    return collection.map(correct)

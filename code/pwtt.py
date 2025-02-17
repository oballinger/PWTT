import datetime
import ee
import geemap
import datetime 


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
    #pre_n = pre.count()
    pre_n = ee.Number(pre.aggregate_array('orbitNumber_start').distinct().size());
    
    # Calculate the mean, standard deviation, and number of images for the pre-event period
    post_mean = post.mean()
    post_sd = post.reduce(ee.Reducer.stdDev())
    #post_n = post.count()
    post_n = ee.Number(post.aggregate_array('orbitNumber_start').distinct().size());

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

    # Return the t-values for each pixel
    return change

def filter_s1(aoi,inference_start,war_start, pre_interval=12, post_interval=2, footprints=None, viz=False, export=False,  export_dir='PWTT_Export', export_name=None, export_scale=10, grid_scale=500, export_grid=False):
    # Filter the image collection to the ascending or descending orbit
    #turn aoi in to a feature collection
    inference_start=ee.Date(inference_start)
    war_start=ee.Date(war_start)
    #aoi = ee.FeatureCollection(aoi)

    orbits = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.contains('.geo', ee.FeatureCollection(aoi).geometry()))\
        .filterDate(ee.Date(inference_start), ee.Date(inference_start).advance(post_interval, 'months')) \
        .aggregate_array('relativeOrbitNumber_start') \
        .distinct()

    #orbits.getInfo()  # Print the orbits

    def map_orbit(orbit):
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
            .map(lee_filter) \
            .select(['VV', 'VH'])\
            .map(lambda image: image.log())\
            .filterBounds(aoi) \

        image = ttest(s1,inference_start, war_start, pre_interval, post_interval)
        return image
    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(
        war_start.advance(-1 * pre_interval, 'months'), war_start).select('built').mean()
    
    image = ee.ImageCollection(orbits.map(map_orbit)).max()
    #image=image.addBands((image.select('VV').add(image.select('VH')).divide(2)).rename('mean_change')).select('mean_change')
    image=image.addBands(image.select('VV').max(image.select('VH')).rename('max_change')).select('max_change')

    image=image.focalMedian(10, 'gaussian', 'meters').clip(aoi).updateMask(urban.gt(0.1))

    #k20=image.convolve(ee.Kernel.circle(20,'meters',True)).rename('k20')
    k50=image.convolve(ee.Kernel.circle(50,'meters',True)).rename('k50')
    k100=image.convolve(ee.Kernel.circle(100,'meters',True)).rename('k100')
    k150=image.convolve(ee.Kernel.circle(150,'meters',True)).rename('k150')

    damage=image.select('max_change').gt(3).rename('damage')
    image=image.addBands(damage)
    image=image.addBands([k50,k100,k150])
    #calculate mean of all four bands
    image=image.addBands((image.select('max_change').add(image.select('k50')).add(image.select('k100')).add(image.select('k150')).divide(4)).rename('T_statistic'))#.select('mean_change')
    image=image.select('T_statistic', 'damage').toFloat()
    image=image.clip(aoi)
    
    if export_grid:
        grid=aoi.geometry().bounds().coveringGrid('EPSG:3857', grid_scale)
        grid=image.reduceRegions({
            'collection': grid,
            'reducer': ee.Reducer.mean(),
            'scale': 10,
            'tileScale': 8,
        })
        task_grid = ee.batch.Export.table.toDrive(
            collection=grid,
            description=export_name+'_grid',
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
        fc=ee.FeatureCollection(footprints).filterBounds(aoi)
        fp=image.reduceRegions(
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

def run(name, pre_interval, post_interval, datestr='2024-01-01'):
    sensor_date = ee.Date(datestr)
    footprints = ee.FeatureCollection('projects/sat-io/open-datasets/MSBuildings/Ukraine')
    war_start = ee.Date('2022-02-22')
    print(datestr.replace('-', ''))
    bounds = adm3.filter(ee.Filter.eq('ADM3_PCODE', name)).geometry()
    
    print(name)
    
    footprints = footprints.filterBounds(bounds) \
                          .map(lambda feat: feat.set('area', feat.geometry().simplify(10).area(10))) \
                          .filter(ee.Filter.gt('area', 50))\
                            #.select(['area','.geo'])
    
    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(
        war_start.advance(-1 * pre_interval, 'months'), war_start).select('built').mean()
    
    dates = [datetime.date(y, m, 1) for y in range(2022, 2024) for m in range(1, 13, 2)]
    dates = [str(d) for d in dates]
    dates.append('2024-01-01')
    dates = ee.List(dates).map(lambda d: ee.Date(d))

    images = dates.map(lambda d: 
                       filter_s1(bounds, d, war_start, pre_interval, post_interval)
                       .focalMedian(20, 'gaussian', 'meters').clip(bounds)
                       .updateMask(urban.gt(0.1)))
    
    images = ee.ImageCollection(images)
    images=images.map(lambda image: image.addBands((image.select('VV').add(image.select('VH')).divide(2)).rename('mean_change')).select('mean_change'))
    image = images.toBands().rename(dates.map(lambda d: ee.Date(d).format('YYYY-MM-dd')).flatten())
    #cutoff=1
    #image = image.updateMask(cutoff).clip(bounds)
    
    fp_sample = image.reduceRegions(collection=footprints, reducer=ee.Reducer.mean(), scale=10, tileScale=8)
    
    """Map=geemap.Map()
    #Map.addLayer(image, {'min': 1.5, 'max': 5, 'opacity': 0.8, 'palette': ["yellow", "red", "purple"]}, "T-test")
    Map.addLayer(fp_sample, {}, "Footprints")
    Map.centerObject(bounds, 12)
    return Map
    """
    task_fp_sample = ee.batch.Export.table.toDrive(
            collection=fp_sample,
            description=name +"_footprints",
            folder='ukraine_damage',
            fileFormat='CSV'
        )
    task_fp_sample.start()
        
    task_image = ee.batch.Export.image.toDrive(
        image=image,
        description=name  + "_raster",
        folder='ukraine_damage',
        scale=10
    )
    #task_image.start()


    import ee


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

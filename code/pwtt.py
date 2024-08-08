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

def filter_s1(aoi,inference_start,war_start, pre_interval=12, post_interval=2, footprints=None, viz=False, export=False,  export_dir='PWTT_Export', export_name=None, export_scale=10):
    # Filter the image collection to the ascending or descending orbit
    #turn aoi in to a feature collection
    inference_start=ee.Date(inference_start)
    war_start=ee.Date(war_start)
    aoi = ee.FeatureCollection(aoi)
    orbits = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filterBounds(aoi) \
        .filterDate(ee.Date(inference_start), ee.Date(inference_start).advance(2, 'months')) \
        .aggregate_array('relativeOrbitNumber_start') \
        .distinct()
        #.filter(ee.Filter.contains('.geo', aoi.geometry())) \
    #orbits.getInfo()  # Print the orbits

    def map_orbit(orbit):
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
            .map(lee_filter) \
            .select(['VV', 'VH'])

        image = ttest(s1,inference_start, war_start, pre_interval, post_interval)
        return image
    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(
        war_start.advance(-1 * pre_interval, 'months'), war_start).select('built').mean()
    
    image = ee.ImageCollection(orbits.map(map_orbit)).max()
    #image=image.addBands((image.select('VV').add(image.select('VH')).divide(2)).rename('mean_change')).select('mean_change')
    image=image.addBands(image.select('VV').max(image.select('VH')).rename('max_change')).select('max_change')

    image=image.focalMedian(10, 'gaussian', 'meters').clip(aoi).updateMask(urban.gt(0.1))

    k20=image.convolve(ee.Kernel.circle(20,'meters',True)).rename('k20')
    k50=image.convolve(ee.Kernel.circle(50,'meters',True)).rename('k50')
    k100=image.convolve(ee.Kernel.circle(100,'meters',True)).rename('k100')

    image=image.addBands([k20,k50,k100])
    #calculate mean of all four bands
    image=image.addBands((image.select('max_change').add(image.select('k20')).add(image.select('k50')).add(image.select('k100')).divide(4)).rename('mean_change')).select('mean_change')

    if viz:
        Map = geemap.Map()
        Map.add_basemap('SATELLITE')
        Map.addLayer(image.select('mean_change'), {'min': 3, 'max': 5, 'opacity': 0.5, 'palette': ["yellow", "red", "purple"]}, "T-test")
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
            fileFormat='CSV'
        )
        task_fp.start()

    else:
        if export:
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=export_name,
                folder=export_dir,
                scale=export_scale
            )
            task.start
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
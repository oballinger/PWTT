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
__all__ = ['detect_damage', 'lee_filter', 'ttest', 'ztest', 'hotelling_t2', 'terrain_flattening', '__version__']


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


def ttest(s1, inference_start, war_start, pre_interval, post_interval, ttest_type='welch'):
    inference_start = ee.Date(inference_start)

    # Filter to pre-event and post-event periods
    pre = s1.filterDate(
        war_start.advance(ee.Number(pre_interval).multiply(-1), "month"),
        war_start
    )
    post = s1.filterDate(inference_start, inference_start.advance(post_interval, "month"))

    # Per-period statistics
    pre_mean = pre.mean()
    pre_sd = pre.reduce(ee.Reducer.stdDev())
    pre_n = pre.select('VV').count()

    post_mean = post.mean()
    post_sd = post.reduce(ee.Reducer.stdDev())
    post_n = post.select('VV').count()

    if ttest_type == 'welch':
        # Welch's t-test: does not assume equal variance
        var_pre_n = pre_sd.pow(2).divide(pre_n)
        var_post_n = post_sd.pow(2).divide(post_n)
        sum_var = var_pre_n.add(var_post_n)
        denom = sum_var.sqrt()

        # Welch-Satterthwaite degrees of freedom (per pixel, per band)
        df = sum_var.pow(2).divide(
            var_pre_n.pow(2).divide(pre_n.subtract(1))
            .add(var_post_n.pow(2).divide(post_n.subtract(1)))
        )
    else:
        # Pooled t-test (original): assumes equal variance
        pooled_sd = (pre_sd.pow(2)
                     .multiply(pre_n.subtract(1))
                     .add(post_sd.pow(2).multiply(post_n.subtract(1)))) \
            .divide(pre_n.add(post_n).subtract(2)).sqrt()
        denom = pooled_sd.multiply(
            ee.Image(1).divide(pre_n).add(ee.Image(1).divide(post_n)).sqrt()
        )
        # Pooled df is scalar per pixel (same for VV and VH)
        pooled_df = pre_n.add(post_n).subtract(2)
        df = pooled_df.addBands(pooled_df)

    change = post_mean.subtract(pre_mean).divide(denom).abs()

    # Compute two-tailed p-values (normal approx, valid for df > 30)
    p_values = two_tailed_pvalue(change).rename(['VV_pvalue', 'VH_pvalue'])

    # Rename df bands for downstream use (Stouffer weighting)
    df = df.rename(['df_VV', 'df_VH'])

    # Mask out pixels with insufficient observations
    valid_mask = pre_n.gte(3).And(post_n.gte(2))
    change = change.updateMask(valid_mask)
    p_values = p_values.updateMask(valid_mask)
    df = df.updateMask(valid_mask)

    return change.addBands(p_values).addBands(pre_n.toFloat().rename('n_pre')) \
        .addBands(post_n.toFloat().rename('n_post')).addBands(df.toFloat())


def ztest(s1, inference_start, war_start, pre_interval):
    """Z-test: compare the single latest post-event image to the pre-war baseline.
    z = |x_latest - mean_pre| / sd_pre, per pixel.
    """
    inference_start = ee.Date(inference_start)

    pre = s1.filterDate(
        war_start.advance(ee.Number(pre_interval).multiply(-1), "month"),
        war_start
    )

    pre_mean = pre.mean()
    pre_sd = pre.reduce(ee.Reducer.stdDev())
    pre_n = pre.select('VV').count()

    # Latest single image after inference_start
    post = s1.filterDate(inference_start, ee.Date('2099-01-01'))
    latest = post.sort('system:time_start', False).mosaic()

    z_vv = latest.select('VV').subtract(pre_mean.select('VV')) \
        .divide(pre_sd.select('VV_stdDev')).abs().rename('VV')
    z_vh = latest.select('VH').subtract(pre_mean.select('VH')) \
        .divide(pre_sd.select('VH_stdDev')).abs().rename('VH')

    p_vv = two_tailed_pvalue(z_vv).rename('VV_pvalue')
    p_vh = two_tailed_pvalue(z_vh).rename('VH_pvalue')

    valid_mask = pre_n.gte(3)
    z_vv = z_vv.updateMask(valid_mask)
    z_vh = z_vh.updateMask(valid_mask)
    p_vv = p_vv.updateMask(valid_mask)
    p_vh = p_vh.updateMask(valid_mask)

    # df bands: z-test has no meaningful df, use pre_n as placeholder for Stouffer weight
    df_vv = pre_n.toFloat().rename('df_VV')
    df_vh = pre_n.toFloat().rename('df_VH')

    return z_vv.addBands(z_vh).addBands(p_vv).addBands(p_vh) \
        .addBands(pre_n.toFloat().rename('n_pre')) \
        .addBands(ee.Image.constant(1).toFloat().rename('n_post')) \
        .addBands(df_vv).addBands(df_vh)


def hotelling_t2(s1, inference_start, war_start, pre_interval, post_interval, ttest_type='welch'):
    """Hotelling's T-squared: joint multivariate test on VV and VH.
    Uses closed-form 2x2 inverse of pooled covariance matrix.
    Output: sqrt(T2) in both VV and VH bands for compatibility with downstream max(VV,VH).
    """
    inference_start = ee.Date(inference_start)

    pre = s1.filterDate(
        war_start.advance(ee.Number(pre_interval).multiply(-1), "month"),
        war_start
    )
    post = s1.filterDate(inference_start, inference_start.advance(post_interval, "month"))

    pre_mean = pre.mean()
    post_mean = post.mean()
    pre_n = pre.select('VV').count()
    post_n = post.select('VV').count()

    # Per-pixel variances
    pre_sd = pre.reduce(ee.Reducer.stdDev())
    post_sd = post.reduce(ee.Reducer.stdDev())
    pre_var_vv = pre_sd.select('VV_stdDev').pow(2)
    pre_var_vh = pre_sd.select('VH_stdDev').pow(2)
    post_var_vv = post_sd.select('VV_stdDev').pow(2)
    post_var_vh = post_sd.select('VH_stdDev').pow(2)

    # Per-pixel cross-covariance: cov(VV, VH) = E[(VV-mu_VV)(VH-mu_VH)]
    pre_cov = pre.map(lambda img:
        img.select('VV').subtract(pre_mean.select('VV'))
        .multiply(img.select('VH').subtract(pre_mean.select('VH')))
        .rename('cov')
    ).mean().multiply(pre_n).divide(pre_n.subtract(1))  # Bessel correction

    post_cov = post.map(lambda img:
        img.select('VV').subtract(post_mean.select('VV'))
        .multiply(img.select('VH').subtract(post_mean.select('VH')))
        .rename('cov')
    ).mean().multiply(post_n).divide(post_n.subtract(1))  # Bessel correction

    # Pooled covariance matrix elements: S_pooled = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    denom_pool = pre_n.add(post_n).subtract(2)
    s11 = pre_var_vv.multiply(pre_n.subtract(1)).add(post_var_vv.multiply(post_n.subtract(1))).divide(denom_pool)
    s22 = pre_var_vh.multiply(pre_n.subtract(1)).add(post_var_vh.multiply(post_n.subtract(1))).divide(denom_pool)
    s12 = pre_cov.multiply(pre_n.subtract(1)).add(post_cov.multiply(post_n.subtract(1))).divide(denom_pool)

    # Determinant with epsilon floor
    det = s11.multiply(s22).subtract(s12.pow(2)).max(ee.Image.constant(1e-10))

    # Mean differences
    d_vv = post_mean.select('VV').subtract(pre_mean.select('VV'))
    d_vh = post_mean.select('VH').subtract(pre_mean.select('VH'))

    # T² = (n1*n2/(n1+n2)) * d' * S_pooled^{-1} * d
    # For 2x2: d'*S^{-1}*d = (d_vv²*s22 - 2*d_vv*d_vh*s12 + d_vh²*s11) / det
    quad_form = d_vv.pow(2).multiply(s22) \
        .subtract(d_vv.multiply(d_vh).multiply(s12).multiply(2)) \
        .add(d_vh.pow(2).multiply(s11)) \
        .divide(det)
    t2 = pre_n.multiply(post_n).divide(pre_n.add(post_n)).multiply(quad_form)

    # P-value: chi-squared(2) approximation — p = exp(-T²/2)
    p_value = t2.multiply(-0.5).exp().max(ee.Image.constant(1e-10))

    # Output sqrt(T²) as test statistic
    t_stat = t2.sqrt()
    change_vv = t_stat.rename('VV')
    change_vh = t_stat.rename('VH')  # same value in both bands
    p_vv = p_value.rename('VV_pvalue')
    p_vh = p_value.rename('VH_pvalue')

    # Degrees of freedom: use combined n for Stouffer weighting
    combined_df = pre_n.add(post_n).subtract(2).toFloat()
    df_vv = combined_df.rename('df_VV')
    df_vh = combined_df.rename('df_VH')

    # Valid mask: need enough observations for covariance estimation
    valid_mask = pre_n.gte(3).And(post_n.gte(2))
    change_vv = change_vv.updateMask(valid_mask)
    change_vh = change_vh.updateMask(valid_mask)
    p_vv = p_vv.updateMask(valid_mask)
    p_vh = p_vh.updateMask(valid_mask)

    return change_vv.addBands(change_vh).addBands(p_vv).addBands(p_vh) \
        .addBands(pre_n.toFloat().rename('n_pre')) \
        .addBands(post_n.toFloat().rename('n_post')) \
        .addBands(df_vv).addBands(df_vh)


S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11']


def _build_s2_collection(aoi, start, end):
    """Sentinel-2 SR with s2cloudless + cloud-shadow projection + SCL fallback.

    Returns an ImageCollection of cloud-masked, scaled (0–1) reflectance images
    with bands [B2, B3, B4, B8, B11] and the MGRS_TILE property carried through.
    Mirrors the official GEE s2cloudless tutorial.
    """
    start = ee.Date(start)
    end = ee.Date(end)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)))

    s2c = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
           .filterBounds(aoi)
           .filterDate(start, end))

    joined = ee.ImageCollection(
        ee.Join.saveFirst('cloud_prob').apply(
            primary=s2,
            secondary=s2c,
            condition=ee.Filter.equals(leftField='system:index', rightField='system:index'),
        )
    )

    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 1  # km

    def mask_one(img):
        cloud_prob = ee.Image(img.get('cloud_prob')).select('probability')
        is_cloud = cloud_prob.gt(CLD_PRB_THRESH).rename('clouds')

        # Solar azimuth → projection vector
        scaled = img.select('B8').divide(10000)
        dark_pixels = scaled.lt(NIR_DRK_THRESH).multiply(
            img.select('SCL').neq(6)  # exclude water
        ).rename('dark_pixels')

        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
        )
        cld_proj = (is_cloud.directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                    .reproject(crs=img.select(0).projection(), scale=100)
                    .select('distance').mask().rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # SCL belt-and-braces: 3=shadow, 8/9=clouds, 10=cirrus, 11=snow
        scl = img.select('SCL')
        scl_mask = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9))
                    .Or(scl.eq(10)).Or(scl.eq(11)))

        cloud_or_shadow = is_cloud.Or(shadows).Or(scl_mask)

        # Buffer mask by ~50 m
        buffered = (cloud_or_shadow.focalMin(2).focalMax(10)
                    .reproject(crs=img.select(0).projection(), scale=20)
                    .rename('cloud_mask'))

        # Scale to [0,1] reflectance, then add 0 to drop the inferred range so
        # the band type is plain Float (matches the masked sentinel later).
        scaled_bands = (img.select(S2_BANDS).divide(10000)
                        .add(ee.Image.constant(0)).toFloat()
                        .rename(S2_BANDS))
        return (scaled_bands.updateMask(buffered.Not())
                .copyProperties(img, ['system:time_start', 'MGRS_TILE']))

    return joined.map(mask_one)


def detect_damage(aoi, inference_start, war_start, pre_interval=12, post_interval=2, footprints=None, viz=False, export=False, export_dir='PWTT_Export', export_name=None, export_scale=10, grid_scale=500, export_grid=False, clip=True, method='stouffer', threshold=3.3, ttest_type='welch', smoothing='default', mask_before_smooth=True, lee_mode='per_image', sensor='s1'):
    import warnings

    if (export or export_grid) and export_name is None:
        raise ValueError("export_name is required when export=True or export_grid=True")

    # Warn if inference_start is before war_start (likely a mistake)
    if isinstance(inference_start, str) and isinstance(war_start, str):
        inf = datetime.datetime.strptime(inference_start, '%Y-%m-%d')
        war = datetime.datetime.strptime(war_start, '%Y-%m-%d')
        if inf < war:
            warnings.warn(
                f"inference_start ({inference_start}) is before war_start ({war_start}). "
                "The post-war period will use pre-war imagery."
            )

    if sensor not in ('s1', 's2', 'combined'):
        raise ValueError(f"sensor must be 's1', 's2', or 'combined', got '{sensor}'")
    if sensor in ('s2', 'combined') and method not in ('mahalanobis', 'hotelling'):
        raise ValueError(
            f"sensor='{sensor}' only supports method in ('mahalanobis', 'hotelling'), got '{method}'"
        )

    inference_start = ee.Date(inference_start)
    war_start = ee.Date(war_start)

    def _setup_s1():
        bl = ['VV', 'VH']
        orbs = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filterBounds(aoi) \
            .filterDate(inference_start, inference_start.advance(post_interval, 'months')) \
            .aggregate_array('relativeOrbitNumber_start') \
            .distinct()

        def make_s1(orbit):
            s1 = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
                .filter(ee.Filter.eq("instrumentMode", "IW")) \
                .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
                .filterBounds(aoi)
            if lee_mode == 'per_image':
                s1 = s1.map(lee_filter)
            return s1.select(['VV', 'VH']).map(lambda image: image.log())

        return bl, orbs, make_s1

    def _setup_s2():
        bl = list(S2_BANDS)
        window_start = war_start.advance(ee.Number(pre_interval).multiply(-1), 'month')
        window_end = inference_start.advance(post_interval, 'month')
        s2_full = _build_s2_collection(aoi, window_start, window_end)
        grps = s2_full.aggregate_array('MGRS_TILE').distinct()

        def make_s2(tile):
            return s2_full.filter(ee.Filter.eq('MGRS_TILE', tile))

        return bl, grps, make_s2

    if sensor == 's1':
        band_list, groups, make_group_collection = _setup_s1()
    elif sensor == 's2':
        band_list, groups, make_group_collection = _setup_s2()
    else:  # combined
        s1_band_list, s1_groups, s1_make = _setup_s1()
        s2_band_list, s2_groups, s2_make = _setup_s2()
        # Used downstream for the empty-fallback orbit-size check
        groups = s1_groups

    if sensor == 's1':
        # Fallback image for orbits with no coverage — all bands present but fully masked
        empty_orbit = ee.Image.constant([0, 0, 0, 0, 0, 0, 0, 0]).rename(
            ['VV', 'VH', 'VV_pvalue', 'VH_pvalue', 'n_pre', 'n_post', 'df_VV', 'df_VH']
        ).updateMask(ee.Image.constant(0)).toFloat()

        def map_orbit_ttest(orbit):
            s1 = make_orbit_s1(orbit)
            result = ttest(s1, inference_start, war_start, pre_interval, post_interval, ttest_type=ttest_type)
            return ee.Image(ee.Algorithms.If(result.bandNames().size().gt(0), result, empty_orbit))

        def map_orbit_ztest(orbit):
            s1 = make_orbit_s1(orbit)
            result = ztest(s1, inference_start, war_start, pre_interval)
            return ee.Image(ee.Algorithms.If(result.bandNames().size().gt(0), result, empty_orbit))

    urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(
        war_start.advance(-1 * pre_interval, 'months'), war_start).select('built').mean()

    if method in ('hotelling', 'mahalanobis', 'cusum'):

        def _compute_for_sensor(bl, grps, make_group):
            """Run per-group z-normalization + pooled Mahalanobis for one sensor.

            Returns a dict with t2, quad_form, pre_n, post_n, valid_mask, p,
            and z_max (latest-image absolute-z statistic, unmasked).
            """
            p = len(bl)

            def normalize_group_images(group_id):
                coll = make_group(group_id)
                pre = coll.filterDate(
                    war_start.advance(ee.Number(pre_interval).multiply(-1), "month"), war_start)
                has_pre = pre.select(bl[0]).count().reduceRegion(
                    ee.Reducer.max(), aoi, 1000).values().get(0)
                pre_mean = pre.mean()
                pre_sd = pre.reduce(ee.Reducer.stdDev()).rename(bl)
                normalized = coll.map(lambda img:
                    img.select(bl).subtract(pre_mean).divide(pre_sd.max(ee.Image.constant(1e-10)))
                    .copyProperties(img, ['system:time_start'])
                ).toList(500)
                return ee.Algorithms.If(ee.Number(has_pre).gt(0), normalized, ee.List([]))

            masked_sentinel = ee.Image.constant([0] * p).rename(bl).updateMask(0).toFloat()
            pre_sentinel = masked_sentinel.set('system:time_start', war_start.advance(-1, 'day').millis())
            post_sentinel = masked_sentinel.set('system:time_start', inference_start.advance(1, 'day').millis())
            all_norm = ee.ImageCollection(grps.map(normalize_group_images).flatten()) \
                .merge(ee.ImageCollection([pre_sentinel, post_sentinel]))
            pre_norm = all_norm.filterDate(
                war_start.advance(ee.Number(pre_interval).multiply(-1), "month"), war_start)
            post_norm = all_norm.filterDate(
                inference_start, inference_start.advance(post_interval, "month"))

            pre_mean_raw = pre_norm.mean()
            post_mean_raw = post_norm.mean()
            if sensor == 's1' and lee_mode == 'composite' and bl == ['VV', 'VH']:
                _add_angle = lambda img: img.addBands(ee.Image.constant(0).rename('angle'))
                pre_mean = lee_filter(_add_angle(pre_mean_raw)).select(bl)
                post_mean = lee_filter(_add_angle(post_mean_raw)).select(bl)
            else:
                pre_mean = pre_mean_raw
                post_mean = post_mean_raw
            pre_n_l = pre_norm.select(bl[0]).count()
            post_n_l = post_norm.select(bl[0]).count()

            denom_pool = pre_n_l.add(post_n_l).subtract(2)

            def cross_cov(coll, mean_img, n, bi, bj):
                return coll.map(lambda img:
                    img.select(bi).subtract(mean_img.select(bi))
                    .multiply(img.select(bj).subtract(mean_img.select(bj)))
                    .rename('cov')
                ).mean().multiply(n).divide(n.subtract(1))

            cov_band_imgs = []
            for i in range(p):
                for j in range(p):
                    bi = bl[i]
                    bj = bl[j]
                    pre_ij = cross_cov(pre_norm, pre_mean_raw, pre_n_l, bi, bj)
                    post_ij = cross_cov(post_norm, post_mean_raw, post_n_l, bi, bj)
                    s_ij = (pre_ij.multiply(pre_n_l.subtract(1))
                            .add(post_ij.multiply(post_n_l.subtract(1)))
                            .divide(denom_pool))
                    cov_band_imgs.append(s_ij.rename(f's_{i}_{j}'))

            cov_flat = ee.Image.cat(cov_band_imgs).toArray()
            cov_array = cov_flat.arrayReshape(ee.Image(ee.Array([p, p])), 2)
            ridge = ee.Image(ee.Array([[1e-10 if i == j else 0.0 for j in range(p)]
                                        for i in range(p)]))
            s_inv = cov_array.add(ridge).matrixInverse()

            d_bands = [post_mean.select(b).subtract(pre_mean.select(b)).rename(f'd_{b}')
                       for b in bl]
            d_flat = ee.Image.cat(d_bands).toArray()
            d_col = d_flat.arrayReshape(ee.Image(ee.Array([p, 1])), 2)
            qf_arr = d_col.arrayTranspose().matrixMultiply(s_inv).matrixMultiply(d_col)
            quad_form_l = qf_arr.arrayProject([0]).arrayFlatten([['quad']])
            t2_l = pre_n_l.multiply(post_n_l).divide(pre_n_l.add(post_n_l)).multiply(quad_form_l)

            valid_mask_l = pre_n_l.gte(3).And(post_n_l.gte(2))

            # Latest-image z-test
            latest_post = all_norm.filterDate(
                inference_start, inference_start.advance(post_interval, 'month')
            ).sort('system:time_start', False).mosaic()
            z_per_band = [latest_post.select(b).abs() for b in bl]
            z_max_l = z_per_band[0]
            for zi in z_per_band[1:]:
                z_max_l = z_max_l.max(zi)

            return dict(
                t2=t2_l, quad_form=quad_form_l, pre_n=pre_n_l, post_n=post_n_l,
                valid_mask=valid_mask_l, p=p, z_max=z_max_l, all_norm=all_norm,
                make_group=make_group, bl=bl,
            )

        if sensor in ('s1', 's2'):
            r = _compute_for_sensor(band_list, groups, make_group_collection)
            t2 = r['t2']
            quad_form = r['quad_form']
            p = r['p']
            pre_n = r['pre_n']
            post_n = r['post_n']
            valid_mask = r['valid_mask']
            z_max = r['z_max'].updateMask(valid_mask).rename('Z_statistic')
        else:  # combined: independence assumption → block-diagonal cov → T²_combined = T²_S1 + T²_S2
            if method == 'cusum':
                raise ValueError("sensor='combined' does not support method='cusum'")
            r1 = _compute_for_sensor(s1_band_list, s1_groups, s1_make)
            r2 = _compute_for_sensor(s2_band_list, s2_groups, s2_make)
            t2 = r1['t2'].unmask(0).add(r2['t2'].unmask(0))
            # quad_form analogue: use t2 directly for max_change in mahalanobis branch
            quad_form = r1['quad_form'].unmask(0).add(r2['quad_form'].unmask(0))
            p = r1['p'] + r2['p']
            pre_n = r1['pre_n'].unmask(0).add(r2['pre_n'].unmask(0))
            post_n = r1['post_n'].unmask(0).add(r2['post_n'].unmask(0))
            # Need both sensors to have valid coverage in a pixel
            valid_mask = r1['valid_mask'].unmask(0).And(r2['valid_mask'].unmask(0))
            z_max = r1['z_max'].unmask(0).max(r2['z_max'].unmask(0)) \
                .updateMask(valid_mask).rename('Z_statistic')

        if method == 'cusum':
            # Page's CUSUM on per-pixel fused magnitude m_t = sqrt(sum_b x_b²)
            # over post-war z-normalized images. Output: max S_t per pixel.
            def normalize_group_post(group_id):
                coll = make_group_collection(group_id)
                pre = coll.filterDate(
                    war_start.advance(ee.Number(pre_interval).multiply(-1), "month"), war_start)
                has_pre = pre.select(band_list[0]).count().reduceRegion(
                    ee.Reducer.max(), aoi, 1000).values().get(0)
                pre_mean_g = pre.mean()
                pre_sd_g = pre.reduce(ee.Reducer.stdDev()).rename(band_list)
                post = coll.filterDate(inference_start, inference_start.advance(post_interval, "month"))
                normalized = post.map(lambda img:
                    img.select(band_list).subtract(pre_mean_g).divide(pre_sd_g.max(ee.Image.constant(1e-10)))
                    .copyProperties(img, ['system:time_start'])
                ).toList(500)
                return ee.Algorithms.If(ee.Number(has_pre).gt(0), normalized, ee.List([]))

            post_only = ee.ImageCollection(groups.map(normalize_group_post).flatten())
            post_sorted = post_only.sort('system:time_start')

            def to_magnitude(img):
                sq = ee.Image(img).select(band_list).pow(2)
                # sum across bands → 1-band magnitude
                total = ee.Image.constant(0)
                for b in band_list:
                    total = total.add(sq.select(b))
                return (total.sqrt().rename('m')
                        .copyProperties(img, ['system:time_start']))

            mag_ic = post_sorted.map(to_magnitude)
            first_mag = ee.Image(mag_ic.first())
            zero_img = first_mag.multiply(0).rename('m')
            cusum_k = ee.Image.constant(2.0)
            initial = ee.List([zero_img, zero_img])

            def step(img, prev):
                prev = ee.List(prev)
                s_prev = ee.Image(prev.get(0))
                max_s = ee.Image(prev.get(1))
                s_new = ee.Image(img).subtract(cusum_k).add(s_prev).max(ee.Image.constant(0))
                return ee.List([s_new, s_new.max(max_s)])

            final = ee.List(mag_ic.iterate(step, initial))
            max_change = ee.Image(final.get(1)).rename('max_change')
            p_value = max_change.multiply(-0.5).exp().max(ee.Image.constant(1e-10)).rename('p_value')
        elif method == 'mahalanobis':
            # Effect size: sqrt(Mahalanobis distance) — n-invariant
            max_change = quad_form.sqrt().rename('max_change')
            # Chi-squared(p) survival approximation (valid for large n).
            p_value = t2.multiply(-0.5).exp().max(ee.Image.constant(1e-10)).rename('p_value')
        else:
            # Hotelling: sqrt(T²) as test statistic
            max_change = t2.sqrt().rename('max_change')
            p_value = t2.multiply(-0.5).exp().max(ee.Image.constant(1e-10)).rename('p_value')

        max_change = max_change.updateMask(valid_mask)
        p_value = p_value.updateMask(valid_mask)
        n_pre = pre_n.rename('n_pre')
        n_post = post_n.rename('n_post')
        z_p = two_tailed_pvalue(z_max).updateMask(valid_mask).rename('Z_p_value')

    else:
        # Per-orbit test → combine across orbits
        if method == 'ztest':
            orbit_images = ee.ImageCollection(groups.map(map_orbit_ztest))
        elif method == 'mahalanobis_max':
            # JS app-style: per-orbit Mahalanobis on raw log SAR (no z-norm),
            # always with per-image Lee filter, MAX across orbits.
            def map_orbit_mahalanobis_raw(orbit):
                s1 = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT") \
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
                    .filter(ee.Filter.eq("instrumentMode", "IW")) \
                    .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
                    .filterBounds(aoi) \
                    .map(lee_filter) \
                    .select(['VV', 'VH']) \
                    .map(lambda image: image.log())
                pre = s1.filterDate(
                    war_start.advance(ee.Number(pre_interval).multiply(-1), 'month'), war_start)
                post = s1.filterDate(
                    inference_start, inference_start.advance(post_interval, 'month'))

                pre_n_o = pre.select('VV').count()
                post_n_o = post.select('VV').count()
                pre_mean_o = pre.mean()
                post_mean_o = post.mean()

                pre_sd_o = pre.reduce(ee.Reducer.stdDev())
                post_sd_o = post.reduce(ee.Reducer.stdDev())
                pre_var_vv = pre_sd_o.select('VV_stdDev').pow(2)
                pre_var_vh = pre_sd_o.select('VH_stdDev').pow(2)
                post_var_vv = post_sd_o.select('VV_stdDev').pow(2)
                post_var_vh = post_sd_o.select('VH_stdDev').pow(2)

                pre_cov_o = pre.map(lambda img:
                    img.select('VV').subtract(pre_mean_o.select('VV'))
                       .multiply(img.select('VH').subtract(pre_mean_o.select('VH')))
                       .rename('cov')
                ).mean().multiply(pre_n_o).divide(pre_n_o.subtract(1))
                post_cov_o = post.map(lambda img:
                    img.select('VV').subtract(post_mean_o.select('VV'))
                       .multiply(img.select('VH').subtract(post_mean_o.select('VH')))
                       .rename('cov')
                ).mean().multiply(post_n_o).divide(post_n_o.subtract(1))

                denom = pre_n_o.add(post_n_o).subtract(2)
                s11 = pre_var_vv.multiply(pre_n_o.subtract(1)) \
                    .add(post_var_vv.multiply(post_n_o.subtract(1))).divide(denom)
                s22 = pre_var_vh.multiply(pre_n_o.subtract(1)) \
                    .add(post_var_vh.multiply(post_n_o.subtract(1))).divide(denom)
                s12 = pre_cov_o.multiply(pre_n_o.subtract(1)) \
                    .add(post_cov_o.multiply(post_n_o.subtract(1))).divide(denom)
                det = s11.multiply(s22).subtract(s12.pow(2)).max(ee.Image.constant(1e-10))

                d_vv = post_mean_o.select('VV').subtract(pre_mean_o.select('VV'))
                d_vh = post_mean_o.select('VH').subtract(pre_mean_o.select('VH'))
                quad = d_vv.pow(2).multiply(s22) \
                    .subtract(d_vv.multiply(d_vh).multiply(s12).multiply(2)) \
                    .add(d_vh.pow(2).multiply(s11)) \
                    .divide(det)
                mahal = quad.sqrt().rename('mahal')

                # Hotelling T² → chi-squared(p) approx for p-value
                t2 = pre_n_o.multiply(post_n_o).divide(pre_n_o.add(post_n_o)).multiply(quad)
                p_v = t2.multiply(-0.5).exp().max(ee.Image.constant(1e-10)).rename('p_value')

                return mahal.addBands(p_v) \
                    .addBands(pre_n_o.toFloat().rename('n_pre')) \
                    .addBands(post_n_o.toFloat().rename('n_post'))

            orbit_images = ee.ImageCollection(groups.map(map_orbit_mahalanobis_raw))
        else:
            orbit_images = ee.ImageCollection(groups.map(map_orbit_ttest))

        if method == 'stouffer':
            # Stouffer's weighted Z-score: weight each orbit by sqrt(df).
            # Combined Z = sum(w*t)/sqrt(sum(w²)) is standard normal under H0.
            def add_stouffer_bands(img):
                df = img.select('df_VV').max(img.select('df_VH'))
                w = df.sqrt()
                return img.addBands(img.select('VV').multiply(w).rename('w_VV')) \
                          .addBands(img.select('VH').multiply(w).rename('w_VH')) \
                          .addBands(df.rename('w_sq'))

            orbit_images = orbit_images.map(add_stouffer_bands)
            sum_w_sq = orbit_images.select('w_sq').sum()
            z_vv = orbit_images.select('w_VV').sum().divide(sum_w_sq.sqrt()).rename('VV')
            z_vh = orbit_images.select('w_VH').sum().divide(sum_w_sq.sqrt()).rename('VH')

            max_change = z_vv.max(z_vh).rename('max_change')
            # Independent VV/VH tests → Bonferroni correction ×2
            p_value = two_tailed_pvalue(max_change).multiply(2) \
                .min(ee.Image.constant(1)).rename('p_value')
            n_pre = orbit_images.select('n_pre').sum()
            n_post = orbit_images.select('n_post').sum()

        elif method in ('max', 'ztest'):
            # max t-value (or z-value) across orbits, min p-value, Bonferroni
            t_max = orbit_images.select(['VV', 'VH']).max()
            p_min = orbit_images.select(['VV_pvalue', 'VH_pvalue']).min()
            n_pre = orbit_images.select('n_pre').max()
            n_post = orbit_images.select('n_post').max()
            image = t_max.addBands(p_min)

            max_change = image.select('VV').max(image.select('VH')).rename('max_change')
            p_value = image.select('VV_pvalue').min(image.select('VH_pvalue')).rename('p_value')
            n_orbits = groups.size()
            p_value = p_value.multiply(n_orbits).min(ee.Image.constant(1)).rename('p_value')

        elif method == 'mahalanobis_max':
            max_change = orbit_images.select('mahal').max().rename('max_change')
            p_value = orbit_images.select('p_value').min().rename('p_value')
            n_pre = orbit_images.select('n_pre').max()
            n_post = orbit_images.select('n_post').max()
            valid_mask_l = n_pre.gte(3).And(n_post.gte(2))
            max_change = max_change.updateMask(valid_mask_l)
            p_value = p_value.updateMask(valid_mask_l)
            # Z_statistic placeholders (zero-valued) for output schema parity with mahalanobis
            z_max = ee.Image.constant(0).toFloat().rename('Z_statistic').updateMask(valid_mask_l)
            z_p = ee.Image.constant(1).toFloat().rename('Z_p_value').updateMask(valid_mask_l)

        else:
            raise ValueError(f"method must be 'stouffer', 'max', 'ztest', 'hotelling', 'mahalanobis', or 'mahalanobis_max', got '{method}'")

    # Build a fully-masked empty image as fallback for areas with no S1 coverage
    empty_bands = ['T_statistic', 'damage', 'p_value', 'n_pre', 'n_post']
    empty_vals = [0, 0, 1, 0, 0]
    if method in ('hotelling', 'mahalanobis', 'cusum', 'mahalanobis_max'):
        empty_bands += ['Z_statistic', 'Z_p_value']
        empty_vals += [0, 1]
    empty = ee.Image.constant(empty_vals).rename(empty_bands) \
        .updateMask(ee.Image.constant(0)).toFloat()

    # Constrain to areas with valid raw data before smoothing
    raw_data_mask = max_change.mask()
    urban_mask = urban.gt(0.1)

    # Parse smoothing config
    if method == 'mahalanobis_max' and smoothing == 'default':
        # JS-app default: 20m gaussian focal median, no multi-scale convolutions
        smoothing = dict(focal_radius=20, kernels=[], weights=[1.0])
    if smoothing == 'default':
        smooth_cfg = dict(focal_radius=10, kernels=[50, 100, 150], weights=[0.25, 0.25, 0.25, 0.25])
    elif smoothing == 'focal_only':
        smooth_cfg = dict(focal_radius=10, kernels=[], weights=[1.0])
    elif isinstance(smoothing, dict):
        smooth_cfg = smoothing
    else:
        raise ValueError(f"smoothing must be 'default', 'focal_only', or a dict, got '{smoothing}'")

    # Urban mask ordering: before or after focal median
    if mask_before_smooth:
        max_change_input = max_change.updateMask(urban_mask).updateMask(raw_data_mask)
        t_smooth = max_change_input.focalMedian(smooth_cfg['focal_radius'], 'gaussian', 'meters')
        if clip:
            t_smooth = t_smooth.clip(aoi)
    else:
        t_smooth = max_change.focalMedian(smooth_cfg['focal_radius'], 'gaussian', 'meters')
        if clip:
            t_smooth = t_smooth.clip(aoi)
        t_smooth = t_smooth.updateMask(urban_mask).updateMask(raw_data_mask)

    # Multi-scale convolutions
    layers = [t_smooth]
    for radius in smooth_cfg.get('kernels', []):
        layers.append(t_smooth.convolve(ee.Kernel.circle(radius, 'meters', True)))

    # Weighted average across scales
    weights = smooth_cfg['weights']
    T_statistic = ee.Image.constant(0).toFloat()
    for layer, w in zip(layers, weights):
        T_statistic = T_statistic.add(layer.multiply(w))
    T_statistic = T_statistic.rename('T_statistic')

    # Re-apply raw data mask so T_statistic doesn't extend beyond where n_post is valid
    T_statistic = T_statistic.updateMask(raw_data_mask)
    damage = T_statistic.gt(threshold).rename('damage')

    # Mask p-values with urban mask
    p_value = p_value.updateMask(urban.gt(0.1))
    if clip:
        p_value = p_value.clip(aoi)

    image = T_statistic.addBands(damage).addBands(p_value).addBands(n_pre).addBands(n_post)
    if method in ('hotelling', 'mahalanobis', 'cusum', 'mahalanobis_max'):
        image = image.addBands(z_max).addBands(z_p)
    image = image.toFloat()

    # If no groups had coverage, return the empty fallback
    image = ee.Image(ee.Algorithms.If(groups.size().gt(0), image, empty))
    if clip:
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
        task_grid.start()

    if viz:
        Map = geemap.Map()
        Map.add_basemap('SATELLITE')
        Map.addLayer(image.select('T_statistic'), {'min': 3, 'max': 5, 'opacity': 0.5, 'palette': ["yellow", "red", "purple"]}, "T-test")
        Map.centerObject(aoi)
        return Map

    if footprints is not None:
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

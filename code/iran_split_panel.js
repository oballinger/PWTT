

// ======================================== CONFIG ==========================================

var WAR_START = ee.Date('2026-03-01');
var CLOUD_THRESH = 20;

var iran = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0')
  .filter(ee.Filter.eq('ADM0_CODE', 117))
  .geometry();
// Load and merge all feature collections in the iran_ttest asset folder
var folder = 'projects/ggmap-325812/assets/iran_hotelling/';

var assets = ee.data.listAssets(folder);
var ids = assets.assets.map(function(a) { return a.id; });

var damagePoints = ee.FeatureCollection(ids.map(function(id) {
  return ee.FeatureCollection(id);
})).flatten();

print('Total features:', damagePoints.size());
print('Sample:', damagePoints.limit(5));

// Visualize damaged buildings


//var damagePoints = ee.FeatureCollection('projects/ggmap-325812/assets/iran_damage_all')
//  .filter(ee.Filter.bounds(geometry).not());

// ======================================== CLOUD MASKING ==================================

function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var clear = scl.eq(4).or(scl.eq(5)).or(scl.eq(6))
    .or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(clear)
    .select('B4', 'B3', 'B2', 'B8')
    .divide(10000)
    .copyProperties(image, ['system:time_start']);
}

// ======================================== COMPOSITES =====================================

function addTimeBand(image) {
  return image.addBands(image.metadata('system:time_start').rename('time'));
}

// Pre-war: most recent clear pixel in the month before war start
var preComposite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(iran)
  .filterDate(WAR_START.advance(-1, 'month'), WAR_START)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESH))
  .map(maskS2Clouds)
  .map(addTimeBand)
  .qualityMosaic('time')
  .select('B4', 'B3', 'B2', 'B8')
  .clip(iran);

// Post-war: most recent clear pixel since war start
var postComposite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(iran)
  .filterDate(WAR_START, ee.Date(Date.now()))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESH))
  .map(maskS2Clouds)
  .map(addTimeBand)
  .qualityMosaic('time')
  .select('B4', 'B3', 'B2', 'B8')
  .clip(iran);

// ======================================== VIS PARAMS =====================================

var trueColor = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
// ======================================== DAMAGE LAYER ===================================

var DEFAULT_T_THRESH = 3.5;
var DEFAULT_P_THRESH = 0.01;
var DEFAULT_N_THRESH = 3;

var currentT = DEFAULT_T_THRESH;
var currentP = DEFAULT_P_THRESH;
var currentN = DEFAULT_N_THRESH;

function filterDamage() {
  return damagePoints
    .filter(ee.Filter.gte('T_statistic', currentT))
    .filter(ee.Filter.lte('p_value', currentP))
    .filter(ee.Filter.gte('n_post', currentN));
}

function makeDamageLayer() {
  return filterDamage().style({color: 'FF0000', pointSize: 3, width: 1});
}

// ======================================== LEE FILTER ==========================================

function leeFilter(image) {
  var KERNEL_SIZE = 2;
  var bandNames = image.bandNames().remove('angle');
  var enl = 5;
  var eta = ee.Image.constant(1.0 / Math.sqrt(enl));
  var oneImg = ee.Image.constant(1);
  var reducers = ee.Reducer.mean().combine({
    reducer2: ee.Reducer.variance(), sharedInputs: true
  });
  var stats = image.select(bandNames).reduceNeighborhood({
    reducer: reducers,
    kernel: ee.Kernel.square(KERNEL_SIZE / 2, 'pixels'),
    optimization: 'window'
  });
  var meanBand = bandNames.map(function(name) { return ee.String(name).cat('_mean'); });
  var varBand  = bandNames.map(function(name) { return ee.String(name).cat('_variance'); });
  var zBar = stats.select(meanBand);
  var varz = stats.select(varBand);
  var varx = varz.subtract(zBar.pow(2).multiply(eta.pow(2))).divide(oneImg.add(eta.pow(2)));
  var b = varx.divide(varz);
  var newB = b.where(b.lt(0), 0);
  var output = oneImg.subtract(newB).multiply(zBar.abs()).add(newB.multiply(image.select(bandNames)));
  return image.addBands(output.rename(bandNames), null, true);
}

// ======================================== S1 TIME SERIES CONFIG ==========================================

var PRE_INTERVAL = 12;
var COLLECTION_START = WAR_START.advance(-PRE_INTERVAL, 'month');
var s1_base = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// ======================================== SPLIT PANEL ====================================

ui.root.clear();

var leftMap = ui.Map();
var rightMap = ui.Map();

// Layer 0: S2 composites
leftMap.addLayer(preComposite, trueColor, 'Pre-war (1 month)');
rightMap.addLayer(postComposite, trueColor, 'Post-war');

// Layer 1: damage
var initDamage = makeDamageLayer();
leftMap.addLayer(initDamage, {}, 'Damaged buildings');
rightMap.addLayer(initDamage, {}, 'Damaged buildings');

// Labels
leftMap.add(ui.Label('Pre-war (Feb 2026)', {
  fontWeight: 'bold', fontSize: '16px', color: 'white',
  backgroundColor: '00000088', padding: '6px 12px',
  position: 'top-left'
}));
rightMap.add(ui.Label('Post-war (Mar 2026 —)', {
  fontWeight: 'bold', fontSize: '16px', color: 'white',
  backgroundColor: '00000088', padding: '6px 12px',
  position: 'top-right'
}));

leftMap.setOptions('HYBRID');
rightMap.setOptions('HYBRID');

var linker = ui.Map.Linker([leftMap, rightMap]);

var splitPanel = ui.SplitPanel({
  firstPanel: leftMap,
  secondPanel: rightMap,
  wipe: true,
  style: {stretch: 'both'}
});

ui.root.add(splitPanel);

// ======================================== CONTROLS =======================================

var countLabel = ui.Label('', {fontSize: '13px', color: 'white', backgroundColor: '00000000'});

function refreshLayers() {
  var layer = makeDamageLayer();
  leftMap.layers().set(1, ui.Map.Layer(layer, {}, 'Damaged buildings'));
  rightMap.layers().set(1, ui.Map.Layer(layer, {}, 'Damaged buildings'));

  var count = filterDamage().size();
  count.evaluate(function(val) {
    countLabel.setValue('Buildings shown: ' + (val ? val.toLocaleString() : '0'));
  });
}

// --- T-statistic slider ---
var tLabel = ui.Label('', {fontSize: '12px', color: 'white', backgroundColor: '00000000'});

var tSlider = ui.Slider({
  min: 2, max: 8, value: DEFAULT_T_THRESH, step: 0.1,
  style: {stretch: 'horizontal', padding: '0px 8px'},
  onChange: function(value) {
    currentT = value;
    tLabel.setValue('T > ' + value.toFixed(1));
    refreshLayers();
  }
});

// --- p-value slider ---
var pLabel = ui.Label('', {fontSize: '12px', color: 'white', backgroundColor: '00000000'});

var pSlider = ui.Slider({
  min: 0.001, max: 0.1, value: DEFAULT_P_THRESH, step: 0.001,
  style: {stretch: 'horizontal', padding: '0px 8px'},
  onChange: function(value) {
    currentP = value;
    pLabel.setValue('p < ' + value.toFixed(3));
    refreshLayers();
  }
});

// --- n_post slider ---
var nLabel = ui.Label('', {fontSize: '12px', color: 'white', backgroundColor: '00000000'});

var nSlider = ui.Slider({
  min: 1, max: 15, value: DEFAULT_N_THRESH, step: 1,
  style: {stretch: 'horizontal', padding: '0px 8px'},
  onChange: function(value) {
    currentN = value;
    nLabel.setValue('n_post ≥ ' + value);
    refreshLayers();
  }
});

// --- Threshold guide ---
var guidePanel = ui.Panel({
  widgets: [
    ui.Label('Threshold guide', {fontWeight: 'bold', fontSize: '12px', color: 'white', backgroundColor: '00000000', margin: '6px 0 2px 0'}),
    ui.Label('T > 2: Max sensitivity (39% precision, 98% recall)', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '1px 0'}),
    ui.Label('T > 3.3: Balanced (70% precision, 82% recall)', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '1px 0'}),
    ui.Label('T > 4: High confidence (78% precision, 63% recall)', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '1px 0'}),
    ui.Label('T > 5: Very high confidence (82% precision, 32% recall)', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '1px 0'}),
    ui.Label('p-value: statistical significance of change', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '4px 0 1px 0'}),
    ui.Label('n_post: # of post-event SAR images (higher = more reliable)', {fontSize: '11px', color: 'AAAAAA', backgroundColor: '00000000', margin: '1px 0'}),
  ]
});

// --- Assemble control panel ---
var controlPanel = ui.Panel({
  widgets: [
    ui.Label('Iran Damage Assessment', {
      fontWeight: 'bold', fontSize: '18px', color: 'white',
      backgroundColor: '00000000'
    }),
    ui.Label('Sentinel-2 Split View with PWTT Damage Detection', {
      fontSize: '12px', color: 'CCCCCC', backgroundColor: '00000000'
    }),

    // T-statistic
    ui.Label('T-statistic threshold:', {fontSize: '13px', color: 'white', backgroundColor: '00000000', margin: '8px 0 2px 0'}),
    tSlider,
    tLabel,

    // p-value
    ui.Label('p-value threshold:', {fontSize: '13px', color: 'white', backgroundColor: '00000000', margin: '6px 0 2px 0'}),
    pSlider,
    pLabel,

    // n_post
    ui.Label('Min post-event images (n_post):', {fontSize: '13px', color: 'white', backgroundColor: '00000000', margin: '6px 0 2px 0'}),
    nSlider,
    nLabel,

    countLabel,

    // CSV download button
    ui.Button({
      label: '⬇ Download CSV',
      style: {stretch: 'horizontal', margin: '8px 0 4px 0'},
      onClick: function() {
        var filtered = filterDamage();
        var url = filtered.getDownloadURL({format: 'csv'});
        print('Download CSV:', ui.Label(url, {color: '4488FF'}, url));
      }
    }),

    guidePanel
  ],
  style: {
    position: 'bottom-left', padding: '12px 16px',
    backgroundColor: '000000CC', width: '340px'
  }
});

leftMap.add(controlPanel);

// Initialize labels and count
tLabel.setValue('T > ' + DEFAULT_T_THRESH.toFixed(1));
pLabel.setValue('p < ' + DEFAULT_P_THRESH.toFixed(3));
nLabel.setValue('n_post ≥ ' + DEFAULT_N_THRESH);
refreshLayers();

// ======================================== CLICK → TIME SERIES ==========================================

var chartPanel = ui.Panel({style: {width: '600px', position: 'bottom-right'}});
rightMap.add(chartPanel);
leftMap.style().set('cursor', 'crosshair');
rightMap.style().set('cursor', 'crosshair');

function showTimeSeries(coords) {
  chartPanel.clear();
  chartPanel.add(ui.Label('Loading time series...', {color: 'gray'}));

  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var aoi = point.buffer(100);

  var localOrbits = s1_base.filterBounds(aoi)
    .filterDate(COLLECTION_START, ee.Date(Date.now()))
    .aggregate_array('relativeOrbitNumber_start').distinct();

  function makeLocalOrbit(orbit) {
    var s1 = s1_base
      .filter(ee.Filter.eq('relativeOrbitNumber_start', orbit))
      .filterDate(COLLECTION_START, ee.Date(Date.now()))
      .filterBounds(aoi)
      .map(leeFilter)
      .select(['VV', 'VH'])
      .map(function(img) { return img.log().copyProperties(img, ['system:time_start']); });
    return s1;
  }

  function normalizeLocalOrbit(orbit) {
    var s1 = makeLocalOrbit(orbit);
    var pre = s1.filterDate(COLLECTION_START, WAR_START);
    var preMean = pre.mean();
    var preSd = pre.reduce(ee.Reducer.stdDev()).rename(['VV', 'VH']);
    return s1.map(function(img) {
      return img.subtract(preMean)
        .divide(preSd.max(ee.Image.constant(1e-10)))
        .copyProperties(img, ['system:time_start']);
    }).toList(200);
  }

  var allNorm = ee.ImageCollection(localOrbits.map(normalizeLocalOrbit).flatten());

  // Add ±2.576 CI bands (99% CI for standard normal)
  var withCI = allNorm.map(function(img) {
    return img.select(['VV', 'VH'])
      .addBands(ee.Image.constant(2.576).rename('z_upper_99'))
      .addBands(ee.Image.constant(-2.576).rename('z_lower_99'))
      .copyProperties(img, ['system:time_start']);
  });

  var chart = ui.Chart.image.series({
    imageCollection: withCI,
    region: point,
    reducer: ee.Reducer.first(),
    scale: 10
  }).setOptions({
    title: 'Orbit-normalized z-scores (' + coords.lat.toFixed(3) + ', ' + coords.lon.toFixed(3) + ')',
    vAxis: {title: 'z-score (σ from pre-war mean)'},
    hAxis: {title: 'Date'},
    lineWidth: 0, pointSize: 3,
    series: {
      0: {color: '4488FF', pointSize: 3},
      1: {color: 'FF8800', pointSize: 3},
      2: {color: 'CC0000', lineWidth: 1, lineDashStyle: [4, 4], pointSize: 0, visibleInLegend: false},
      3: {color: 'CC0000', lineWidth: 1, lineDashStyle: [4, 4], pointSize: 0, visibleInLegend: false},
    },
    interpolateNulls: false
  }).setChartType('ScatterChart');

  chartPanel.clear();
  chartPanel.add(chart);
  chartPanel.add(ui.Label('--- 99% confidence interval', {fontSize: '11px', color: 'CC0000', margin: '0px 0px 0px 60px'}));
}

leftMap.onClick(showTimeSeries);
rightMap.onClick(showTimeSeries);

// Center on Tehran
leftMap.setCenter(51.37, 35.70, 12);
rightMap.setCenter(51.37, 35.70, 12);


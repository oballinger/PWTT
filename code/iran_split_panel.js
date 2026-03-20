// ======================================== CONFIG ==========================================

var WAR_START = ee.Date('2026-03-01');
var CLOUD_THRESH = 20;

var iran = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0')
  .filter(ee.Filter.eq('ADM0_CODE', 117))
  .geometry();

var damagePoints = ee.FeatureCollection('projects/ggmap-325812/assets/iran_damage_all');

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

var DEFAULT_T_THRESH = 3;
var DEFAULT_P_THRESH = 0.05;
var DEFAULT_N_THRESH = 3;

var currentT = DEFAULT_T_THRESH;
var currentP = DEFAULT_P_THRESH;
var currentN = DEFAULT_N_THRESH;

function filterDamage() {
  var filtered = damagePoints
    .filter(ee.Filter.gte('T_statistic', currentT))
    .filter(ee.Filter.lte('p_value', currentP))
    .filter(ee.Filter.gte('n_post', currentN));

  // Exclude points inside user-drawn geometries
  var drawLayer = leftMap.drawingTools().layers().get(0);
  var exclusion_geom = drawLayer.toGeometry();
  filtered = filtered.filter(ee.Filter.bounds(exclusion_geom).not());

  return filtered;
}

function makeDamageLayer() {
  return filterDamage().style({color: 'FF0000', pointSize: 3, width: 1});
}

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

// ======================================== DRAWING TOOLS ==================================

var drawingTools = leftMap.drawingTools();
drawingTools.setShown(true);
drawingTools.setLinked(true);
drawingTools.addLayer([], 'User drawings', {color: '00FFFF'});

// Also enable on right map so drawn geometries appear on both sides
rightMap.drawingTools().setShown(false); // hide duplicate toolbar
rightMap.drawingTools().setLinked(true);

// Refresh damage layer when user draws/edits/erases exclusion zones
drawingTools.onDraw(refreshLayers);
drawingTools.onEdit(refreshLayers);
drawingTools.onErase(refreshLayers);

// Center on Tehran
leftMap.setCenter(51.37, 35.70, 12);
rightMap.setCenter(51.37, 35.70, 12);

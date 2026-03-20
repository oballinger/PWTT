// ======================================== DATA ==========================================

var aoi = ee.Geometry.Rectangle([35.10, 33.05, 35.60, 33.35])

var war_start = ee.Date('2026-03-02')
var inference_start = ee.Date('2026-03-02')

var ms_buildings = ee.FeatureCollection('projects/sat-io/open-datasets/MSBuildings/Lebanon')

var title = "South Lebanon Change Detection"

var locations = [
  {name: 'Aalma El Chaeb',              lon: 35.1822, lat: 33.0956, zoom: 15},
  {name: 'Ramyeh',                      lon: 35.3129, lat: 33.1079, zoom: 15},
  {name: 'Ayta Al-Shab',                lon: 35.3301, lat: 33.0956, zoom: 14},
  {name: 'Bint Jbeil & Maroun el Ras',  lon: 35.4370, lat: 33.1155, zoom: 14},
  {name: 'Aitaroun',                    lon: 35.4689, lat: 33.1223, zoom: 15},
  {name: 'Jalal Deirah base',           lon: 35.4818, lat: 33.1013, zoom: 16},
  {name: 'Meiss el Jabal',              lon: 35.5174, lat: 33.1753, zoom: 14},
  {name: 'Houla',                       lon: 35.5203, lat: 33.2199, zoom: 15},
  {name: 'Taybeh to Kfar Kela',         lon: 35.5399, lat: 33.2819, zoom: 14}
]

// ======================================== SAR HELPERS ==========================================

function powerToDb(img){
  return ee.Image(10).multiply(img.log10())
}

function dbToPower(img){
  return ee.Image(10).pow(img.divide(10))
}

var leeFilter = function(image) {

  var KERNEL_SIZE=2
  var bandNames = image.bandNames().remove('angle')

  var enl = 5
  var eta = 1.0/Math.sqrt(enl)
  eta = ee.Image.constant(eta)

  var oneImg = ee.Image.constant(1)

  var reducers = ee.Reducer.mean().combine({
    reducer2: ee.Reducer.variance(),
    sharedInputs: true
  })

  var stats = image.select(bandNames).reduceNeighborhood({
    reducer: reducers,
    kernel: ee.Kernel.square(KERNEL_SIZE/2,'pixels'),
    optimization: 'window'
  })

  var meanBand = bandNames.map(function(b){ return ee.String(b).cat('_mean') })
  var varBand = bandNames.map(function(b){ return ee.String(b).cat('_variance') })

  var z_bar = stats.select(meanBand)
  var varz = stats.select(varBand)

  var varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2))))
    .divide(oneImg.add(eta.pow(2)))

  var b = varx.divide(varz)
  var new_b = b.where(b.lt(0),0)

  var output = oneImg.subtract(new_b)
    .multiply(z_bar.abs())
    .add(new_b.multiply(image.select(bandNames)))

  output = output.rename(bandNames)

  return image.addBands(output,null,true)
}

// ======================================== T TEST ==========================================

function ttest(s1, inference_start, war_start, pre_interval ,post_interval) {

  var inference_start = ee.Date(inference_start)

  var pre = s1.filterDate(
    war_start.advance(ee.Number(pre_interval).multiply(-1),'month'),
    war_start
  )

  var post = s1.filterDate(
    inference_start,
    inference_start.advance(post_interval,'month')
  )

  var pre_mean = pre.mean()
  var pre_sd = pre.reduce(ee.Reducer.stdDev())
  var pre_n = pre.count()

  var post_mean = post.mean()
  var post_sd = post.reduce(ee.Reducer.stdDev())
  var post_n = post.count()

  var pooled_sd = (pre_sd.pow(2)
    .multiply(pre_n.subtract(1))
    .add(post_sd.pow(2).multiply(post_n.subtract(1))))
    .divide(pre_n.add(post_n).subtract(2))
    .sqrt()

  var denom = pooled_sd.multiply(
    ee.Image(1).divide(pre_n)
    .add(ee.Image(1).divide(post_n)).sqrt()
  )

  var change = post_mean.subtract(pre_mean)
    .divide(denom)
    .abs()

  return change
}

// ======================================== S1 PIPELINE ==========================================

function filter_s1(aoi, inference_start, war_start, pre_interval, post_interval) {

  var start = ee.Date(war_start).advance(pre_interval * -1,'month')
  var end = ee.Date(inference_start).advance(post_interval,'month')

  var s1_base = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
    .filter(ee.Filter.eq("instrumentMode","IW"))
    .filterBounds(aoi)

  var orbits = s1_base
    .filterDate(inference_start,end)
    .aggregate_array('relativeOrbitNumber_start')
    .distinct()

  var image_col = ee.ImageCollection(orbits.map(function(orbit){

    var s1_orbit = s1_base
      .filter(ee.Filter.eq("relativeOrbitNumber_start",orbit))
      .map(leeFilter)
      .select(['VV','VH'])
      .map(function(img){ return img.log() })

    return ttest(s1_orbit,inference_start,war_start,pre_interval,post_interval)

  })).max()

  return image_col
}

function footprints() {

  var aoi = drawingTools.layers().get(0).getEeObject();
  drawingTools.layers().get(0).setShown(false);

  var cutoff = 3;

  var footprints = ms_buildings
    .filterBounds(aoi)
    .map(function(feat) {
      return feat.set('area', feat.geometry().area(10))
                 .set('geometry_type', feat.geometry().type());
    })
    .filter(ee.Filter.gt('area', 200))
    .filter(ee.Filter.equals('geometry_type', 'Polygon'));

  var mean = image.select('max_change').reduceRegions({
    collection: footprints,
    reducer: ee.Reducer.mean(),
    scale: 10
  });
  var damaged = mean.filter(ee.Filter.gt('mean', cutoff));

  var totalCount  = mean.size();
  var damagedCount = damaged.size();
  var proportion  = damagedCount.divide(totalCount).multiply(100).int();

  var sumLabel2  = ui.Label({ value: 'Calculating...' });
  var meanLabel2 = ui.Label({ value: 'Calculating...' });

  var errorMargin = 0.14;
  function makerange(val, max) {
    var low  = val - (val * errorMargin);
    var high = val + (val * errorMargin);
    if (high > max) {
      return Math.round(low).toString().concat(" — ", Math.round(max).toString());
    } else {
      return Math.round(low).toString().concat(" — ", Math.round(high).toString());
    }
  }

  damagedCount.evaluate(function(val) { sumLabel2.setValue(makerange(val, totalCount.getInfo())); });
  proportion.evaluate(function(val)   { meanLabel2.setValue(makerange(val, 100)); });

  var sumPanel  = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), widgets: [ui.Label("Estimated damaged buildings: "), sumLabel2]  });
  var meanPanel = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), widgets: [ui.Label("Proportion (%): "),               meanLabel2] });

  var statsPanel = ui.Panel([sumPanel, meanPanel]);
  mainPanel.widgets().set(5, statsPanel);

  var damageVisParams = { min: 3, max: 5, opacity: 0.8, palette: ["yellow", "red", "purple"] };
  var empty  = ee.Image().byte();
  var fills  = empty.paint({ featureCollection: damaged, color: 'mean' })
                    .paint({ featureCollection: damaged, color: 'black', width: 1 });

  Map.layers().set(1, ui.Map.Layer(fills, damageVisParams, 'Damaged Buildings'));
}

// ======================================== MAIN TTEST RUN ==========================================

function run_ttest(aoi, war_start, inference_start, pre_interval, post_interval){

  var urbanStart = war_start.advance(-pre_interval,'month')

  var urban = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate(urbanStart,war_start)
    .mean()
    .select('built')

  var image = filter_s1(aoi,inference_start,war_start,pre_interval,post_interval)

  image = image
    .updateMask(urban.gt(0.1))
    .unmask(0)
    .clip(aoi)

  var maxChange = image.select('VV')
    .max(image.select('VH'))
    .rename('max_change')

  image = maxChange.focalMedian(20,'gaussian','meters')

  var sizes = [50,100, 150]

  var convs = sizes.map(function(size){
    return image.unmask(0)
      .convolve(ee.Kernel.circle({
        radius:size,
        units:'meters',
        normalize:true
      }))
      .rename('k'+size)
  })

  image = image.rename('max_change').addBands(convs)

  var cutoff = 3

  image = image.updateMask(image.gt(cutoff))

  var reds = ["yellow","red","purple"]

  var damageVis = {
    min:cutoff,
    max:5,
    palette:reds,
    opacity:0.8
  }

  var layer = ui.Map.Layer(
    image.select('max_change'),
    damageVis,
    "Damage Probability"
  )

  Map.layers().set(0,layer)

  return image
}

// ======================================== MAP OPTIONS =====================================
Map.setOptions('Hybrid');
Map.setControlVisibility({all:false});
Map.setControlVisibility({layerList:true,mapTypeControl:true});

// ------------------------------------- USER INTERFACE -------------------------------------
function makeColorBarParams(palette) {
  return {
    bbox: [0, 0, 1, 0.1],
    dimensions: "100x10",
    format: "png",
    min: 0,
    max: 1,
    palette: palette.reverse(),
  };
}

var reds = ["yellow", "red","purple"];

var colorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(reds.reverse()),
  style: { stretch: "horizontal", margin: "0px 8px", maxHeight: "24px" },
});

var legendTitle = ui.Label({
  value: "Damage Probability",
  style: { fontWeight: "bold", textAlign: "center", stretch: "horizontal"},
});

var legendLabels = ui.Panel({
  widgets: [
    ui.Label('61%', { margin: "4px 8px" }),
    ui.Label(" ", { margin: "4px 8px", textAlign: "center", stretch: "horizontal" }),
    ui.Label('>83%', { margin: "4px 8px" }),
  ],
  layout: ui.Panel.Layout.flow("horizontal"),
});

var legendPanel = ui.Panel({
  widgets: [legendTitle, colorBar, legendLabels],
  style: {height: '90px'}
});

// ======================================== LOCATION DROPDOWN ================================

var locationNames = locations.map(function(loc){ return loc.name })

var locationSelect = ui.Select({
  items: locationNames,
  placeholder: 'Select a location...',
  onChange: function(selected) {
    var loc = locations.filter(function(l){ return l.name === selected })[0]
    Map.setCenter(loc.lon, loc.lat, loc.zoom)
  },
  style: { stretch: 'horizontal' }
})

// ======================================== DRAWING TOOLS ==================================

var drawButton = ui.Button({
  label: "🔺 Draw a Polygon",
  onClick: drawPolygon,
  style: { stretch: "horizontal" },
});

var mainPanel = ui.Panel({
  widgets: [
    ui.Label(title, { fontWeight: "bold", fontSize: "20px" }),
    ui.Label("Select a location to navigate, or draw a polygon for building-level estimates", { whiteSpace: "wrap" }),
    locationSelect,
    drawButton,
    ui.Label(),
    legendPanel
  ],
  style: { position: "top-left", maxWidth: "350px", maxHeight: '90%'},
  layout: ui.Panel.Layout.flow("vertical", true),
});

// ======================================== DRAWING TOOLS ==================================
var drawingTools = Map.drawingTools();
drawingTools.setShown(false);

while (drawingTools.layers().length() > 0) {
  var layer = drawingTools.layers().get(0);
  drawingTools.layers().remove(layer);
}

var dummyGeometry = ui.Map.GeometryLayer({
  geometries: null,
  name: "geometry",
  color: "23cba7",
}).setShown(false);

drawingTools.layers().add(dummyGeometry)

function clearGeometry() {
  var layers = drawingTools.layers();
  layers.get(0).geometries().remove(layers.get(0).geometries().get(0));
}

function drawPolygon() {
  clearGeometry();
  drawingTools.setShape("polygon");
  drawingTools.draw();
}

// ======================================== INITIALIZE MAP ==================================
function home(){
  Map.setOptions('Hybrid');
  Map.style().set("cursor", "crosshair");
  Map.setCenter(35.35, 33.15, 11);
  var image = run_ttest(aoi, war_start, inference_start, 12, 1).select(['max_change','k50','k100']);
  Map.add(mainPanel);
  return image;
}

var image = home();

drawingTools.onDraw(ui.util.debounce(footprints, 500));

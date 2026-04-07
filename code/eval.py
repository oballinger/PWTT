"""
Evaluate PWTT damage detection against ground truth damage annotations.

Computes precision, recall, F1, and AUC at 500m grid level by pulling
scores/labels via aggregate_array().getInfo() — no Drive exports needed.

Usage:
    python eval.py                          # both methods, all cities
    python eval.py --method stouffer        # new method only
    python eval.py --cities Gaza Bucha      # specific cities
"""

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from pwtt import detect_damage

ee.Initialize(project='ggmap-325812')

POST_INTERVAL = 1

# ========================= Geometries =========================

gaza = ee.Geometry.Polygon([[
    [34.21632370049618, 31.325772019150463], [34.23520645196103, 31.296146595009333],
    [34.2448194890704, 31.276782374243183], [34.25237258965634, 31.253305260668878],
    [34.263702240535245, 31.23129016304913], [34.282928314753995, 31.23393224569227],
    [34.34026321465634, 31.266805314667213], [34.37871536309384, 31.30172040609968],
    [34.36875900323056, 31.360958427544336], [34.37493881280087, 31.379426022431435],
    [34.40000137383603, 31.403457772374438], [34.42889376147439, 31.43129174150309],
    [34.461509423095485, 31.453845844503416], [34.498244957763454, 31.486056572632563],
    [34.51541109545877, 31.500694147975228], [34.551821787607736, 31.51650015575805],
    [34.572077830088205, 31.54459312391977], [34.54049213672883, 31.559514075294807],
    [34.49036701465852, 31.59607494108798], [34.376040537607736, 31.483421565782656],
]])

mariupol = ee.Geometry.MultiPolygon([
    [[[37.49898977590781, 47.093621017941295], [37.505083754789645, 47.092627635086494],
      [37.53847189260703, 47.090348627933196], [37.53924436880332, 47.08947206075292],
      [37.54173345876914, 47.089413622427955], [37.54139013601523, 47.08333568654904],
      [37.563019469511325, 47.08783576341764], [37.56585188223105, 47.088069523257474],
      [37.57194586111289, 47.10098316029895], [37.56387777639609, 47.10513122069421],
      [37.54439421011191, 47.111907638064054], [37.53641195608359, 47.11412731132893],
      [37.526798918974215, 47.116638935212016], [37.50568456960898, 47.11733983234765],
      [37.493839934599215, 47.0957245910547]]],
    [[[37.61364942503284, 47.120684751895496], [37.61296277952503, 47.10666593626707],
      [37.62669568968128, 47.094045844687194], [37.650213298323855, 47.091124100819194],
      [37.651243266585574, 47.13785276836463], [37.63030057859729, 47.13575085999574]]],
    [[[37.660684642317996, 47.13785276836463], [37.658453044417605, 47.08925410062196],
      [37.669439372542605, 47.08656586031157], [37.68179899168323, 47.11075513896823],
      [37.69621854734729, 47.11823172574337], [37.68763547849964, 47.13224749474436]]],
])

irpin = ee.Geometry.Polygon([[
    [30.200764375218444, 50.506975396521725], [30.219990449437194, 50.50217149897791],
    [30.22411032248407, 50.493654297737336], [30.20608587790399, 50.482732562241026],
    [30.22084875632196, 50.47421185599678], [30.247112946995788, 50.473447102121895],
    [30.26530905295282, 50.48360639398299], [30.270630555638366, 50.49572914209275],
    [30.25930090475946, 50.4991142188246], [30.268913941868835, 50.52149329763416],
    [30.280415254124694, 50.52836863986559], [30.29054327536493, 50.542007226490156],
    [30.281616883763366, 50.551170141492065], [30.266854005345397, 50.54702523382955],
    [30.245911317357116, 50.541789040145], [30.23355169821649, 50.538734325322444],
    [30.18788977194696, 50.51549019572313],
]])

lysychansk = ee.Geometry.Polygon([[
    [38.350529527606, 48.919850544415645], [38.35276112550639, 48.92312166966242],
    [38.362889146746625, 48.92808434712257], [38.37147221559428, 48.927971564475435],
    [38.38228688234233, 48.92424959420976], [38.382115220965375, 48.921880923237126],
    [38.378853654803265, 48.91939933791301], [38.397564744891156, 48.91353328245856],
    [38.412327623309125, 48.90712784280081], [38.42365727418803, 48.903291480770015],
    [38.42297062868022, 48.897987789476865], [38.42846379274272, 48.888846064562394],
    [38.432068681658734, 48.87575130037355], [38.42365727418803, 48.873831952053926],
    [38.4190224170103, 48.86829930075582], [38.43224034303569, 48.859716835314586],
    [38.44442830079936, 48.8642341058578], [38.445801591814984, 48.86931554788049],
    [38.44871983522319, 48.87100924722223], [38.458504533709515, 48.8703317743654],
    [38.45833287233256, 48.8641211790632], [38.46605763429545, 48.86208845318643],
    [38.47189412111186, 48.8563286148433], [38.48665699952983, 48.85486031474842],
    [38.4988449572935, 48.865476283780445], [38.488716936053265, 48.878155439538325],
    [38.48476872438334, 48.88718624422948], [38.48219380372905, 48.895538286774254],
    [38.45867619508647, 48.91302790830422], [38.44992146486186, 48.91810439392559],
    [38.442025041522015, 48.92250359751281], [38.43395695680522, 48.92475944905124],
    [38.41318593019389, 48.93772861798036], [38.4083794116392, 48.94460654888774],
    [38.40769276613139, 48.94911615297979], [38.38280186647319, 48.948777946809635],
    [38.36597905153178, 48.94922888786042], [38.3479546069517, 48.935698883487554],
    [38.34280476564311, 48.92295477597236],
]])

sievierodonetsk = ee.Geometry.Polygon([[
    [38.431526306655584, 48.9490517024942], [38.45607388355988, 48.93016499099437],
    [38.47701657154816, 48.93546526718773], [38.48062146046418, 48.92802216643069],
    [38.48954785206574, 48.92260830527719], [38.49401104786652, 48.92238271499188],
    [38.50997555592316, 48.93129275645934], [38.51272213795441, 48.93320989928789],
    [38.5204468999173, 48.93208217712799], [38.52508175709504, 48.92802216643069],
    [38.53469479420441, 48.93456313225493], [38.54224789479035, 48.93557803290803],
    [38.55254757740754, 48.94177975524331], [38.54774105885285, 48.951475449426056],
    [38.51598370411652, 48.95282818721421], [38.50551298386053, 48.953335604937294],
    [38.49950421192902, 48.96038029911607], [38.4819947514798, 48.96184550181579],
    [38.47426998951691, 48.967029719591125], [38.47426998951691, 48.970185066549114],
    [38.48903286793488, 48.973678253483385], [38.50534069874543, 48.98404368985209],
    [38.498645905044256, 48.98798649637177], [38.492466095473944, 48.98573350227508],
    [38.469463470962225, 48.96872010886518], [38.44834912159699, 48.96443767807034],
    [38.43410122730988, 48.9521518229049],
]])

adviika = ee.Geometry.Polygon([[
    [37.71134911488289, 48.14632572534626], [37.73212014149422, 48.126965201728886],
    [37.75718270252938, 48.11195290897451], [37.77967034291024, 48.11172367966609],
    [37.78585015248055, 48.12341307066121], [37.785678491103596, 48.14151491598732],
    [37.754192041111665, 48.14949688592137], [37.7397724854476, 48.17846495656275],
    [37.713508294773774, 48.187392524921215], [37.68690078134604, 48.175603227400934],
]])

bucha = ee.Geometry.Polygon([[
    [30.19015556842432, 50.52271662762792], [30.23186928302393, 50.53832071503045],
    [30.260708394352054, 50.54933867410509], [30.260193410221195, 50.559808803492906],
    [30.25332695514307, 50.569731466723795], [30.230667653385257, 50.56907728953237],
    [30.166637959781742, 50.56951340866923], [30.153076711002445, 50.55326524514815],
    [30.179855885807132, 50.54443000040324], [30.183289113346195, 50.53341089436236],
]])

mikolayiv = ee.Geometry.Polygon([[
    [31.88712390914021, 46.983117180756274], [31.914374754175412, 46.986923498769364],
    [31.923944139005116, 46.982764916045284], [31.887208604337147, 46.95066699863325],
    [31.89657047337942, 46.93215187979067], [31.94334187460082, 46.91655651359808],
    [31.97372593832152, 46.9215982625819], [31.981622361661366, 46.91561846144532],
    [31.99801757927003, 46.91438529350639], [31.99278035116332, 46.88570944515149],
    [31.948148393155506, 46.850031403518564], [31.93784871053832, 46.82184777664836],
    [31.958448075772694, 46.81761895768411], [32.057325028897694, 46.84768333251757],
    [32.09715046835082, 46.90682342390656], [32.096463822843006, 46.929804746174945],
    [32.071057939053944, 46.92464653181911], [32.104703568936756, 46.971521122318784],
    [32.05114521932738, 47.04596720493091], [31.977674149991444, 47.04690297286401],
    [31.883775376798084, 46.98955689571937],
]])

shchastia = ee.Geometry.Polygon([[
    [39.22028712520427, 48.74372791480332], [39.215266029928394, 48.73922807943761],
    [39.21479396114177, 48.73611475002484], [39.21337775478191, 48.73529393106154],
    [39.217712204549976, 48.734529708255245], [39.21912841090984, 48.73314275571219],
    [39.21861342677898, 48.732435112123056], [39.21977214107341, 48.73195390879336],
    [39.2205446172697, 48.73082164749767], [39.22264746913738, 48.7304536570866],
    [39.22380618343181, 48.73266155915523], [39.22633818874187, 48.73486936427006],
    [39.22865561733074, 48.73535053969619], [39.235436241720386, 48.733001227791185],
    [39.23852614650554, 48.730227200110406], [39.244920532797046, 48.73472784120943],
    [39.229213516805835, 48.74534096523399],
]])

kramatorsk = ee.Geometry.Polygon([[
    [37.459833722890885, 48.69924133417226], [37.464296918691666, 48.69040327520038],
    [37.50428209800125, 48.69952069754394], [37.50840197104812, 48.69691477899233],
    [37.52127657431961, 48.69634825710836], [37.53294954795242, 48.70156001796294],
    [37.53947268027664, 48.7005403680814], [37.548570733255154, 48.694761961916065],
    [37.566766839212185, 48.690342733175406], [37.577753167337185, 48.690002776436735],
    [37.5868512203157, 48.68354316230749], [37.60710726279617, 48.6960083409178],
    [37.597494225686795, 48.703372677861545], [37.60813723105789, 48.715719686207336],
    [37.62959490317703, 48.71118902587047], [37.63903627890945, 48.71832463115763],
    [37.63474474448562, 48.725685702836515], [37.60779390830398, 48.7468569320614],
    [37.59663591880203, 48.75715628384203], [37.59285936850906, 48.75885377650639],
    [37.59646425742508, 48.7647379736346], [37.59114275473953, 48.76598261930645],
    [37.59165773887039, 48.76847181810897], [37.589254479593045, 48.775259914987544],
    [37.577753167337185, 48.78136841752173], [37.56762514609695, 48.7826126509786],
    [37.56007204551101, 48.78057661653486], [37.56110201377273, 48.77752241000651],
    [37.56693850058914, 48.77424175896149], [37.56865511435867, 48.7723185190182],
    [37.565050225442654, 48.77073461903982], [37.56007204551101, 48.76915066909531],
    [37.56007204551101, 48.7647379736346], [37.56161699790359, 48.760438051239625],
    [37.565050225442654, 48.75998540641467], [37.56367693442703, 48.75444017626453],
    [37.55766878623367, 48.75025255603932], [37.55457888144851, 48.744140267666964],
    [37.55423555869461, 48.73859328852954], [37.55492220420242, 48.73531009414285],
    [37.55148897666336, 48.731800235434136], [37.551317315286404, 48.72806366497459],
    [37.55011568564773, 48.73184997561205], [37.53621111411453, 48.740567551582686],
    [37.52299318808914, 48.739775107143494], [37.50771532554031, 48.73954869215189],
    [37.508573632425076, 48.728453108210445], [37.50222216147781, 48.722791152581834],
    [37.50325212973953, 48.71746833305215], [37.51166353721023, 48.712144950246156],
]])

makariv = ee.Geometry.Polygon([[
    [29.786413298674812, 50.45756755548351], [29.781778441497078, 50.44674679321896],
    [29.79345141512989, 50.445981595133716], [29.80546771151661, 50.44313932248508],
    [29.815595732756844, 50.43953157675058], [29.840486632415047, 50.44620022442068],
    [29.842203246184578, 50.46193887867705], [29.840829955168953, 50.478327742711926],
    [29.816282378264656, 50.48302483610008],
]])

antonivka = ee.Geometry.Polygon([[
    [32.64432403692804, 46.67562964464677], [32.65453788885675, 46.670034726644126],
    [32.64955970892511, 46.666441894983556], [32.6653525556048, 46.65772386818471],
    [32.68947097906671, 46.66155290443205], [32.72646400580011, 46.67504073318559],
    [32.755560609193665, 46.672626129096095], [32.756333085389954, 46.67698411665171],
    [32.746548386903626, 46.68458030869577], [32.72706482061945, 46.68340267445656],
    [32.7147910321673, 46.68958496782213], [32.70105812201105, 46.68893733217283],
    [32.65479538092218, 46.68322602710655], [32.65080425390802, 46.68119454106666],
    [32.64591190466486, 46.67724911807299],
]])

trostianets = ee.Geometry.Polygon([[
    [34.93685638372351, 50.46238001217981], [34.94423782293249, 50.462489289064216],
    [34.95179092351843, 50.451341746231925], [34.958142394465696, 50.44937426008369],
    [34.95213424627234, 50.44183147191478], [34.95608245794226, 50.428710715743456],
    [34.973935241145384, 50.4226958204528], [34.98320495550085, 50.42783586936315],
    [34.971875304621946, 50.44478314090004], [35.006894225520384, 50.44893702982901],
    [35.039338225764524, 50.47133987818742], [35.03024017278601, 50.47647464574971],
    [34.980115050715696, 50.471995411483796], [34.99522125188757, 50.47669313412591],
    [34.992646331233274, 50.48575951117261], [34.977883452815306, 50.497444925048256],
    [34.98725471907606, 50.501485003146826], [34.97849998885145, 50.50978245803542],
    [34.95515404158583, 50.50639814591929], [34.93455467635145, 50.49264005801303],
    [34.93318138533583, 50.47865948406755], [34.917560200033094, 50.47603766596767],
]])

kharkiv_missing = ee.Geometry.Polygon([[
    [36.23027017844288, 50.077197041728745], [36.25979593527882, 50.08738617789176],
    [36.271554739600106, 50.09619665730235], [36.272584707861824, 50.101206894842214],
    [36.26202753317921, 50.10203270787136], [36.25653436911671, 50.09735291247107],
    [36.23095682395069, 50.08832236824121], [36.22657945883839, 50.082099112698955],
]])


# ========================= Metrics =========================

def run_evaluation(labels, scores, area, score_type='t'):
    """
    Compute area-weighted precision, recall, F1 (optimal PR-curve threshold),
    and AUC-ROC.

    score_type: 't' for T_statistic (higher = more damage),
                'p' for p_value (lower = more damage, scores are inverted internally)

    Returns: dict with precision, recall, f1, auc, threshold
    """
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    area = np.array(area, dtype=float)

    # Remove NaN rows
    valid = ~(np.isnan(labels) | np.isnan(scores) | np.isnan(area))
    labels, scores, area = labels[valid], scores[valid], area[valid]

    if score_type == 'p':
        # Invert p-values: lower p = more damage, so use -log10(p) as score
        scores = -np.log10(np.clip(scores, 1e-10, 1.0))

    # AUC from ROC curve (area-weighted)
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=area)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold via precision-recall curve (maximize F1, area-weighted)
    prec_curve, rec_curve, thresholds = precision_recall_curve(
        labels, scores, sample_weight=area
    )
    with np.errstate(invalid='ignore'):
        f1_curve = (2 * prec_curve * rec_curve) / (prec_curve + rec_curve)
    f1_curve = np.nan_to_num(f1_curve)
    best_idx = np.argmax(f1_curve)
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0

    y_pred = (scores > threshold).astype(int)

    return dict(
        precision=precision_score(labels, y_pred, sample_weight=area),
        recall=recall_score(labels, y_pred, sample_weight=area),
        f1=f1_score(labels, y_pred, sample_weight=area),
        auc=roc_auc,
        threshold=threshold,
    )



# ========================= Evaluation =========================

def run_eval(name, pre_interval, post_interval, inference_start,
             ground_truth, footprints, war_start, bounds, method='stouffer', **kwargs):
    """Run PWTT and evaluate against ground truth damage annotations."""

    inference_date = (
        inference_start if isinstance(inference_start, ee.Date)
        else ee.Date(inference_start)
    )
    war_date = ee.Date(war_start)

    # Run PWTT
    image = detect_damage(
        aoi=bounds,
        inference_start=inference_date,
        war_start=war_date,
        pre_interval=pre_interval,
        post_interval=post_interval,
        method=method,
        **kwargs,
    )

    # Ground truth points
    points = ee.FeatureCollection(ground_truth) if isinstance(ground_truth, str) else ground_truth

    # Building footprints, labeled by spatial join with ground truth
    fp = ee.FeatureCollection(footprints) if isinstance(footprints, str) else footprints
    fp = fp.filterBounds(bounds) \
        .map(lambda f: f.set('area', f.geometry().area(10))) \
        .filter(ee.Filter.gt('area', 50))

    spatial_filter = ee.Filter.intersects(
        leftField='.geo', rightField='.geo', maxError=10
    )
    join = ee.Join.saveAll(matchesKey='damage_pts', outer=True)

    def count_pts(feat):
        n = ee.List(feat.get('damage_pts')).size()
        return ee.Algorithms.If(
            ee.Number(n).gt(0),
            feat.set('class', 1),
            feat.set('class', 0),
        )

    buffered_pts = points.map(lambda f: f.buffer(10))
    labeled_fp = join.apply(fp, buffered_pts, spatial_filter).map(count_pts)

    # Reduce PWTT image to footprints
    fp_sample = image.reduceRegions(
        collection=labeled_fp, reducer=ee.Reducer.mean(),
        scale=10, tileScale=8,
    )

    # Filter to footprints that have valid T_statistic and p_value (non-null)
    fp_sample = fp_sample.filter(ee.Filter.notNull(['T_statistic', 'p_value']))

    # Select only needed properties and drop geometry to minimize payload
    fp_sample = fp_sample.select(['class', 'T_statistic', 'p_value', 'area'], retainGeometry=False)

    # Pull data — 1 getInfo call per page (not 4), parse features locally
    labels, t_scores, p_scores, areas = [], [], [], []
    page_size = 5000
    total = fp_sample.size().getInfo()
    offset = 0
    while offset < total:
        page = fp_sample.toList(page_size, offset).getInfo()
        for f in page:
            p = f['properties']
            labels.append(p['class'])
            t_scores.append(p['T_statistic'])
            p_scores.append(p['p_value'])
            areas.append(p['area'])
        offset += page_size
        if offset < total:
            # Incremental metrics on data so far
            inc_t = run_evaluation(labels, t_scores, areas, score_type='t')
            print(f"    ... {offset:,}/{total:,}  AUC={inc_t['auc']:.3f}  F1={inc_t['f1']:.3f}  t*={inc_t['threshold']:.2f}")

    metrics_t = run_evaluation(labels, t_scores, areas, score_type='t')
    metrics_p = run_evaluation(labels, p_scores, areas, score_type='p')
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos

    print(f"  {name:<18s} [T] P={metrics_t['precision']:.3f}  R={metrics_t['recall']:.3f}  "
          f"F1={metrics_t['f1']:.3f}  AUC={metrics_t['auc']:.3f}  t*={metrics_t['threshold']:.2f}")
    print(f"  {'':<18s} [p] P={metrics_p['precision']:.3f}  R={metrics_p['recall']:.3f}  "
          f"F1={metrics_p['f1']:.3f}  AUC={metrics_p['auc']:.3f}  -log10(p)*={metrics_p['threshold']:.2f}")
    print(f"  {'':<18s} (n={len(labels):,}, pos={n_pos:,}, neg={n_neg:,})")

    return dict(name=name, method=method, n=len(labels),
                n_pos=n_pos, n_neg=n_neg,
                **{f't_{k}': v for k, v in metrics_t.items()},
                **{f'p_{k}': v for k, v in metrics_p.items()})


# ========================= City Configs =========================

UA_BUILDINGS = 'projects/sat-io/open-datasets/MSBuildings/Ukraine'

CITIES = [
    dict(name='Gaza', inference_start='2025-04-04',
         ground_truth='projects/ggmap-325812/assets/UNOSAT_GAZA_20250404',
         footprints='projects/ggmap-325812/assets/Gaza_OSM',
         war_start='2023-10-10', bounds=gaza),

    dict(name='Irpin',
         inference_start=ee.Date(ee.FeatureCollection('users/ollielballinger/irpin')
                                 .aggregate_array('SensorDate').distinct().sort().get(-1)),
         ground_truth='users/ollielballinger/irpin',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=irpin),

    dict(name='Mariupol', inference_start='2022-05-12',
         ground_truth='projects/ggmap-325812/assets/mariupol_pts',
         footprints=ee.FeatureCollection(UA_BUILDINGS).filterBounds(mariupol),
         war_start='2022-02-22', bounds=mariupol),

    dict(name='Rubizhne', inference_start='2022-07-09',
         ground_truth='projects/ggmap-325812/assets/Rubizhne_CDA_20220709',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Rubizhne_CDA_20220709_bounds').geometry()),

    dict(name='Chernihiv', inference_start='2022-03-22',
         ground_truth='projects/ggmap-325812/assets/Chernihiv_20220322',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Chernihiv_20220322_bounds').geometry()),

    dict(name='Hostomel', inference_start='2022-03-31',
         ground_truth='projects/ggmap-325812/assets/Hostomel_20220331',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Hostomel_20220331_bounds').geometry()),

    dict(name='Kharkiv', inference_start='2022-06-15',
         ground_truth='projects/ggmap-325812/assets/Kharkiv_CDA_20220615',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Kharkiv_CDA_20220615_bounds')
               .geometry().union(kharkiv_missing)),

    dict(name='Lysychansk',
         inference_start=ee.Date(ee.FeatureCollection('projects/ggmap-325812/assets/Lysychansk_CDA_20220921')
                                 .aggregate_array('SensorDate').distinct().sort().get(-1)),
         ground_truth='projects/ggmap-325812/assets/Lysychansk_CDA_20220921',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=lysychansk),

    dict(name='Sievierodonetsk', inference_start='2022-07-27',
         ground_truth='projects/ggmap-325812/assets/Sievierodonetsk_CDA_20220725',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=sievierodonetsk),

    dict(name='Avdiivka', inference_start='2022-09-20',
         ground_truth='projects/ggmap-325812/assets/Avdiivka_CDA_20220920',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=adviika),

    dict(name='Bucha', inference_start='2022-03-31',
         ground_truth='projects/ggmap-325812/assets/Bucha_20220331_CDA',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=bucha),

    dict(name='Mykolaiv', inference_start='2022-07-21',
         ground_truth='projects/ggmap-325812/assets/Mykolaiv_CDA_20220721',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=mikolayiv),

    dict(name='Shchastia', inference_start='2022-07-07',
         ground_truth='projects/ggmap-325812/assets/Shchastia_CDA_20220707',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=shchastia),

    dict(name='Sumy', inference_start='2022-07-08',
         ground_truth='projects/ggmap-325812/assets/Sumy_CDA_20220708',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Sumy_bounds')
               .filter(ee.Filter.eq('Settlement', 'Sumy')).geometry()),

    dict(name='Trostianets', inference_start='2022-03-25',
         ground_truth='projects/ggmap-325812/assets/Trostianets_Okhtyrka_CDA_20220325',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=trostianets),

    dict(name='Kramatorsk', inference_start='2022-07-24',
         ground_truth='projects/ggmap-325812/assets/Kramatorsk_CDA_20220724',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=kramatorsk),

    dict(name='Okhtyrka', inference_start='2022-03-25',
         ground_truth='projects/ggmap-325812/assets/Okhtyrka_CDA_20220325',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Trostianets_Okhtyrka_bounds').geometry()),

    dict(name='Makariv', inference_start='2022-03-16',
         ground_truth='projects/ggmap-325812/assets/Makariv_CDA_20220316',
         footprints=UA_BUILDINGS, war_start='2022-02-22', bounds=makariv),

    dict(name='Antonivka', inference_start='2022-10-14',
         ground_truth='projects/ggmap-325812/assets/Antonivka_CDA_20221014',
         footprints='projects/ggmap-325812/assets/Antonivka_Kherson_buildings',
         war_start='2022-02-22', bounds=antonivka),

    dict(name='Kremenchuk', inference_start='2022-06-29',
         ground_truth='projects/ggmap-325812/assets/Kremenchuk_CDA_20220629',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Kremenchuk_bounds').geometry()),

    dict(name='Melitopol', inference_start='2022-08-02',
         ground_truth='projects/ggmap-325812/assets/Melitopol_CDA_20220802',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Melitopol_bounds').geometry()),

    dict(name='Kherson', inference_start='2022-10-14',
         ground_truth='projects/ggmap-325812/assets/Kherson_CDA_20221014',
         footprints='projects/ggmap-325812/assets/Antonivka_Kherson_buildings',
         war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Kherson_bounds').geometry()),

    dict(name='Moschun', inference_start='2022-03-31',
         ground_truth='projects/ggmap-325812/assets/Moschun_CDA_20220331',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Moschun_CDA_20220331').bounds()),

    dict(name='Volnovakha', inference_start='2022-04-26',
         ground_truth='projects/ggmap-325812/assets/Volnovakha_CDA_20220426',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Volnovakha_CDA_20220426').bounds()),

    dict(name='Borodyanka', inference_start='2022-05-02',
         ground_truth='projects/ggmap-325812/assets/Borodyanka_2May2022_CDA',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Borodyanka_2May2022_CDA').bounds()),

    dict(name='Vorzel', inference_start='2022-03-31',
         ground_truth='projects/ggmap-325812/assets/Vorzel_20220331_CDA',
         footprints=UA_BUILDINGS, war_start='2022-02-22',
         bounds=ee.FeatureCollection('projects/ggmap-325812/assets/Vorzel_20220331_CDA').bounds()),
]


# ========================= Run =========================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate PWTT against ground truth')
    parser.add_argument('--method', choices=['stouffer', 'max', 'ztest', 'hotelling', 'mahalanobis', 'both', 'all'], default='both',
                        help='Orbit combination method (default: both)')
    parser.add_argument('--cities', nargs='*', default=None,
                        help='Specific cities to run (default: all)')
    parser.add_argument('--ttest-type', choices=['welch', 'pooled'], default='pooled',
                        help='T-test type: welch (unequal variance) or pooled (default: welch)')
    args = parser.parse_args()

    # Build kwargs for detect_damage
    detect_kwargs = dict(
        ttest_type=args.ttest_type,
    )

    if args.method == 'both':
        methods = ['stouffer', 'max']
    elif args.method == 'all':
        methods = ['stouffer', 'max', 'ztest', 'hotelling', 'mahalanobis']
    else:
        methods = [args.method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  method={method}  ttest_type={args.ttest_type}")
        print(f"{'='*60}\n")

        results = []
        for city in CITIES:
            if args.cities and 'all' not in args.cities and city['name'] not in args.cities:
                continue
            try:
                res = run_eval(
                    name=city['name'],
                    pre_interval=12,
                    post_interval=POST_INTERVAL,
                    inference_start=city['inference_start'],
                    ground_truth=city['ground_truth'],
                    footprints=city['footprints'],
                    war_start=city['war_start'],
                    bounds=city['bounds'],
                    method=method,
                    **detect_kwargs,
                )
                results.append(res)
            except Exception as e:
                print(f"  {city['name']:<18s} FAILED: {e}")
                continue

        # Summary table
        if results:
            df = pd.DataFrame(results)
            weights = df['n'].values
            avg_t = {col: np.average(df[f't_{col}'], weights=weights)
                     for col in ['precision', 'recall', 'f1', 'auc']}
            avg_p = {col: np.average(df[f'p_{col}'], weights=weights)
                     for col in ['precision', 'recall', 'f1', 'auc']}
            print(f"\n  {'Weighted avg':<18s} [T] P={avg_t['precision']:.3f}  R={avg_t['recall']:.3f}  "
                  f"F1={avg_t['f1']:.3f}  AUC={avg_t['auc']:.3f}")
            print(f"  {'':<18s} [p] P={avg_p['precision']:.3f}  R={avg_p['recall']:.3f}  "
                  f"F1={avg_p['f1']:.3f}  AUC={avg_p['auc']:.3f}")
            print(f"  {'':<18s} (n={df['n'].sum():,})")

    print(f"\nDone.")

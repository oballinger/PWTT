"""
Profile Hotelling T_statistic distributions across Iran to understand
coverage-dependent false positives and find optimal thresholds.

Uses the per-district iran_hotelling assets which have full metadata
(T_statistic, p_value, Z_statistic, ADM1_NAME, ADM2_NAME, lat/lon).

Usage:
    python code/profile_coverage.py
"""

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

ee.Initialize(project='ggmap-325812')

# ========================= Load data =========================

print("Loading iran_hotelling assets...")
assets = ee.data.listAssets('projects/ggmap-325812/assets/iran_hotelling')
ids = [a['id'] for a in assets['assets']]
print(f"  Found {len(ids)} district assets")

# Merge all into one collection
fc = ee.FeatureCollection(
    [ee.FeatureCollection(aid) for aid in ids]
).flatten()

total = fc.size().getInfo()
print(f"  Total buildings: {total:,}")

# ========================= Pull data (paginated) =========================

props = ['T_statistic', 'p_value', 'Z_statistic', 'Z_p_value',
         'area', 'damage', 'ADM1_NAME', 'ADM2_NAME', 'latitude', 'longitude']
fc_select = fc.select(props, retainGeometry=False)

print("Pulling data...")
rows = []
page_size = 5000
offset = 0
while offset < total:
    page = fc_select.toList(page_size, offset).getInfo()
    for f in page:
        rows.append(f['properties'])
    offset += page_size
    if offset < total and offset % 10000 == 0:
        print(f"  ... {offset:,}/{total:,}")

df = pd.DataFrame(rows)
print(f"  Pulled {len(df):,} buildings")

# ========================= Compute n_post proxy =========================

# We don't have n_post directly, but we can get S1 overpass count per location.
# For now, use ADM1_NAME as a proxy — areas with more coverage will have
# systematically higher Z_statistic even without damage.

# ========================= Province-level summary =========================

print("\n" + "="*80)
print("Province-level summary (sorted by mean T_statistic)")
print("="*80)

prov = df.groupby('ADM1_NAME').agg(
    n_buildings=('T_statistic', 'count'),
    mean_T=('T_statistic', 'mean'),
    median_T=('T_statistic', 'median'),
    std_T=('T_statistic', 'std'),
    mean_Z=('Z_statistic', 'mean'),
    pct_above_3=('T_statistic', lambda x: (x > 3.0).mean() * 100),
    pct_above_4=('T_statistic', lambda x: (x > 4.0).mean() * 100),
    mean_p=('p_value', 'mean'),
).sort_values('mean_T', ascending=False)

print(prov.to_string(float_format='%.3f'))

# ========================= Coverage bias analysis =========================

# Z_statistic is from a single post-war image (no accumulation bias),
# while T_statistic uses all post-war images. If coverage drives FPs,
# T_statistic should be inflated relative to Z_statistic in high-coverage areas.

print("\n" + "="*80)
print("Coverage bias: T_statistic vs Z_statistic ratio by province")
print("="*80)

prov['T_Z_ratio'] = prov['mean_T'] / prov['mean_Z'].clip(lower=0.01)
print(prov[['n_buildings', 'mean_T', 'mean_Z', 'T_Z_ratio', 'pct_above_3']].sort_values('T_Z_ratio', ascending=False).to_string(float_format='%.3f'))

# ========================= Plots =========================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. T_statistic distribution overall
ax = axes[0, 0]
ax.hist(df['T_statistic'].clip(upper=15), bins=100, color='steelblue', edgecolor='none', alpha=0.8)
ax.axvline(3.3, color='red', linestyle='--', label='T=3.3 (default)')
ax.axvline(4.0, color='orange', linestyle='--', label='T=4.0')
ax.set_xlabel('T_statistic')
ax.set_ylabel('Count')
ax.set_title(f'T_statistic distribution (all {len(df):,} buildings)')
ax.legend()

# 2. Scatter: mean_T vs mean_Z per province
ax = axes[0, 1]
ax.scatter(prov['mean_Z'], prov['mean_T'], s=prov['n_buildings'] / 10, alpha=0.7, c='steelblue')
for name, row in prov.iterrows():
    if row['mean_T'] > prov['mean_T'].quantile(0.8) or row['pct_above_3'] > 50:
        ax.annotate(name, (row['mean_Z'], row['mean_T']), fontsize=7, alpha=0.8)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='T=Z line')
ax.set_xlabel('Mean Z_statistic (single image)')
ax.set_ylabel('Mean T_statistic (multi-image)')
ax.set_title('Province: T vs Z (size = n_buildings)')
ax.legend()

# 3. % above threshold by province (bar chart, top 20)
ax = axes[1, 0]
top = prov.nlargest(20, 'pct_above_3')
ax.barh(range(len(top)), top['pct_above_3'], color='steelblue', alpha=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=8)
ax.set_xlabel('% buildings with T > 3.0')
ax.set_title('Top 20 provinces by false positive rate')
ax.invert_yaxis()

# 4. T_statistic vs Z_statistic scatter (sample of buildings)
ax = axes[1, 1]
sample = df.sample(min(5000, len(df)), random_state=42)
ax.scatter(sample['Z_statistic'].clip(upper=15), sample['T_statistic'].clip(upper=15),
           s=1, alpha=0.3, c='steelblue')
ax.plot([0, 15], [0, 15], 'k--', alpha=0.3)
ax.axhline(3.3, color='red', linestyle='--', alpha=0.5, label='T=3.3')
ax.set_xlabel('Z_statistic')
ax.set_ylabel('T_statistic')
ax.set_title('Per-building: T vs Z (5k sample)')
ax.legend()

plt.tight_layout()
plt.savefig('coverage_profile.png', dpi=150)
plt.show()
print("\nSaved coverage_profile.png")

# ========================= Threshold analysis =========================

print("\n" + "="*80)
print("Threshold sensitivity")
print("="*80)

for t in [2.0, 2.5, 3.0, 3.3, 4.0, 5.0, 6.0]:
    n_above = (df['T_statistic'] > t).sum()
    pct = 100 * n_above / len(df)
    print(f"  T > {t:.1f}: {n_above:,} buildings ({pct:.1f}%)")

# ========================= Correlation analysis =========================

print("\n" + "="*80)
print("Correlation: T_statistic with Z_statistic")
print("="*80)

corr = df[['T_statistic', 'Z_statistic']].corr()
print(f"  Pearson r = {corr.iloc[0, 1]:.4f}")

# Buildings where T is high but Z is low (coverage-inflated FPs)
suspect_fp = df[(df['T_statistic'] > 3.3) & (df['Z_statistic'] < 2.0)]
print(f"\n  Suspect FPs (T>3.3 but Z<2.0): {len(suspect_fp):,} "
      f"({100*len(suspect_fp)/len(df):.1f}% of all buildings)")
if len(suspect_fp) > 0:
    print("  Top provinces with suspect FPs:")
    print(suspect_fp['ADM1_NAME'].value_counts().head(10).to_string())

print("\nDone.")

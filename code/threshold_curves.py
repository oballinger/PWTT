"""
Plot precision, recall, and F1 as a function of T-statistic threshold.

Generates interactive Plotly charts for one_month and z_test datasets,
saved as HTML files that can be embedded in docs/readme.

Usage (standalone):
    python threshold_curves.py

Usage (from replication notebook):
    %run threshold_curves.py
    plot_threshold_curves(fp, title='My Title')
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics_at_thresholds(y_true, scores, weights=None,
                                  t_min=1.5, t_max=8, n_points=200):
    """
    Compute precision, recall, and F1 at evenly spaced T-statistic thresholds.
    """
    thresholds = np.linspace(t_min, t_max, n_points)
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            rows.append({'threshold': t, 'precision': np.nan,
                         'recall': np.nan, 'f1': np.nan})
            continue
        p = precision_score(y_true, y_pred, sample_weight=weights,
                            zero_division=0)
        r = recall_score(y_true, y_pred, sample_weight=weights,
                         zero_division=0)
        f = f1_score(y_true, y_pred, sample_weight=weights,
                     zero_division=0)
        rows.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f})
    return pd.DataFrame(rows)


def plot_threshold_curves(df, col='T', title='Precision, Recall & F1 vs Threshold',
                          t_min=1.5, t_max=6, n_points=200,
                          save_path=None):
    """
    Interactive Plotly plot of precision, recall, and F1 vs T-statistic threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: `col`, 'class', 'area'.
    col : str
        Column with T-statistic scores.
    title : str
        Plot title.
    t_min, t_max : float
        Range of thresholds to evaluate.
    n_points : int
        Number of threshold steps.
    save_path : str or None
        If provided, save as HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    metrics = compute_metrics_at_thresholds(
        df['class'].values, df[col].values, df['area'].values,
        t_min, t_max, n_points)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metrics['threshold'], y=metrics['precision'],
        mode='lines', name='Precision',
        line=dict(color='#6fa8dc', width=2.5),
        hovertemplate='T = %{x:.2f}<br>Precision = %{y:.3f}<extra></extra>'))

    fig.add_trace(go.Scatter(
        x=metrics['threshold'], y=metrics['recall'],
        mode='lines', name='Recall',
        line=dict(color='#e06666', width=2.5),
        hovertemplate='T = %{x:.2f}<br>Recall = %{y:.3f}<extra></extra>'))

    fig.add_trace(go.Scatter(
        x=metrics['threshold'], y=metrics['f1'],
        mode='lines', name='F1',
        line=dict(color='#ff7c43', width=3),
        hovertemplate='T = %{x:.2f}<br>F1 = %{y:.3f}<extra></extra>'))

    # Mark optimal F1
    best_idx = metrics['f1'].idxmax()
    if not np.isnan(metrics.loc[best_idx, 'f1']):
        best_t = metrics.loc[best_idx, 'threshold']
        best_f1 = metrics.loc[best_idx, 'f1']

        fig.add_trace(go.Scatter(
            x=[best_t], y=[best_f1],
            mode='markers+text', name=f'Best F1 = {best_f1:.2f} (T = {best_t:.1f})',
            marker=dict(color='#ff7c43', size=12, symbol='circle'),
            text=[f'F1 = {best_f1:.2f}'], textposition='top right',
            textfont=dict(color='#cccccc'),
            hovertemplate=f'Best F1 = {best_f1:.3f}<br>T = {best_t:.2f}<extra></extra>'))

    # Mark default threshold
    fig.add_vline(x=3.3, line_dash='dot', line_color='#aaaaaa', opacity=0.5,
                  annotation_text='T = 3.3 (default)', annotation_position='top left',
                  annotation_font_size=10, annotation_font_color='#aaaaaa')

    fig.update_layout(
        title=title,
        xaxis_title='T-statistic threshold',
        yaxis_title='Score',
        xaxis=dict(range=[t_min, t_max]),
        yaxis=dict(range=[0, 1]),
        template='plotly_dark',
        legend=dict(x=0.75, y=0.65),
        width=800, height=500,
        margin=dict(l=60, r=30, t=60, b=60),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
        print(f'Saved to {save_path}')

    return fig


def load_footprints(data_dir, countries=None):
    """Load footprint CSVs from a directory, derive loc/country from filenames."""
    loc_to_country = {
        'Gaza': 'Palestine', 'Aleppo': 'Syria', 'Raqqa': 'Syria',
        'Mosul': 'Iraq',
    }
    fp = pd.DataFrame()
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and '_footprints' in file:
            path = os.path.join(data_dir, file)
            tdf = pd.read_csv(path, low_memory=False)
            if 'class' in tdf.columns and 'area' in tdf.columns:
                loc = file.split('_')[0]
                tdf['loc'] = loc
                tdf['country'] = loc_to_country.get(loc, 'Ukraine')
                fp = pd.concat([fp, tdf], ignore_index=True)

    if len(fp) == 0:
        return fp

    # Compute T if not present
    if 'T' not in fp.columns:
        t_cols = [c for c in ['max_change', 'k50', 'k100', 'k150']
                  if c in fp.columns]
        if t_cols:
            fp['T'] = fp[t_cols].mean(axis=1)

    if countries:
        fp = fp[fp['country'].isin(countries)]

    return fp


if __name__ == '__main__':
    os.makedirs('../figs', exist_ok=True)

    for data_dir, label in [('../data/one_month', 'one_month'),
                             ('../data/z_test', 'z_test')]:
        if not os.path.isdir(data_dir):
            print(f'Skipping {data_dir} (not found)')
            continue

        fp = load_footprints(data_dir, countries=['Palestine', 'Ukraine'])
        if len(fp) == 0:
            print(f'No data in {data_dir}')
            continue

        print(f'{label}: {len(fp)} buildings loaded')
        fig = plot_threshold_curves(
            fp, col='T',
            title=f'Precision, Recall & F1 vs Threshold ({label})',
            save_path=f'../figs/threshold_curves_{label}.html')
        fig.show()

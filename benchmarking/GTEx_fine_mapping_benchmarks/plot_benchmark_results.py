#!/usr/bin/env python3
"""
Generate ROC and PR curves from consolidated benchmark scores.

Usage:
    python plot_benchmark_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Configuration
PREDICTORS = ['Otari', 'SpliceAI', 'MTSplice', 'Pangolin']
PREDICTOR_COLORS = {
    'Otari': '#0173B2',
    'SpliceAI': '#DE8F05',
    'MTSplice': '#029E73',
    'Pangolin': '#CC78BC'
}
TISSUES = [
    'Brain_Cortex', 'Liver', 'Whole_Blood',
    'Lung', 'Heart_Atrial_Appendage', 'Colon_Sigmoid'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'axes.linewidth': 1.8,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def main():
    setup_style()

    # Load data
    scores_file = os.path.join(SCRIPT_DIR, 'results', 'consolidated_scores.csv')
    print(f"Loading: {scores_file}")
    df = pd.read_csv(scores_file)

    # Create figure
    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                          left=0.06, right=0.98, top=0.97, bottom=0.03)

    # Panel A: ROC Curves
    for idx, tissue in enumerate(TISSUES):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        tissue_df = df[df['tissue'] == tissue]
        y_true = tissue_df['label'].values

        for predictor in PREDICTORS:
            y_scores = tissue_df[predictor].values
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auroc = roc_auc_score(y_true, y_scores)
            ax.plot(fpr, tpr, label=f'{predictor} ({auroc:.3f})',
                   color=PREDICTOR_COLORS[predictor], linewidth=3, alpha=0.9)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, zorder=0)
        ax.set_xlabel('False Positive Rate', fontweight='semibold')
        ax.set_ylabel('True Positive Rate', fontweight='semibold')
        ax.set_title(tissue.replace('_', ' '), fontweight='bold', pad=10)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        if idx == 0:
            ax.text(-0.18, 1.12, 'a', transform=ax.transAxes,
                   fontsize=24, fontweight='bold', va='top')

    # Panel B: PR Curves
    for idx, tissue in enumerate(TISSUES):
        ax = fig.add_subplot(gs[idx // 3 + 2, idx % 3])
        tissue_df = df[df['tissue'] == tissue]
        y_true = tissue_df['label'].values
        baseline = y_true.mean()

        for predictor in PREDICTORS:
            y_scores = tissue_df[predictor].values
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            auprc = average_precision_score(y_true, y_scores)
            ax.plot(recall, precision, label=f'{predictor} ({auprc:.3f})',
                   color=PREDICTOR_COLORS[predictor], linewidth=3, alpha=0.9)

        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
        ax.set_xlabel('Recall', fontweight='semibold')
        ax.set_ylabel('Precision', fontweight='semibold')
        ax.set_title(tissue.replace('_', ' '), fontweight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.95)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        if idx == 0:
            ax.text(-0.18, 1.12, 'b', transform=ax.transAxes,
                   fontsize=24, fontweight='bold', va='top')

    # Save
    output_file = os.path.join(SCRIPT_DIR, 'results', 'roc_pr_curves_all_tissues.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    main()

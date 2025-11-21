#!/usr/bin/env python3
"""
Consolidate Prediction Scores for sQTL Benchmark

This script consolidates prediction scores from all tools into a single file
that is ready for plotting and analysis.

Output formats:
1. consolidated_scores.csv - Per-variant scores with labels (for custom analysis)
2. consolidated_metrics.csv - Pre-computed metrics by tissue/predictor (for tables)
3. curve_data.csv - ROC/PR curve points for each predictor/tissue (for plotting)

Usage:
    python consolidate_scores.py --data-dir data/predictions --susie-dir data/susie
"""

import os
import sys
import argparse
import json
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Configuration
TISSUES = [
    'Brain_Cortex',
    'Liver',
    'Whole_Blood',
    'Lung',
    'Heart_Atrial_Appendage',
    'Colon_Sigmoid'
]

MMSPLICE_TISSUE_NAMES = {
    'Brain_Cortex': 'Cortex - Brain',
    'Liver': 'Liver',
    'Whole_Blood': 'Whole Blood',
    'Lung': 'Lung',
    'Heart_Atrial_Appendage': 'Atrial Appendage - Heart',
    'Colon_Sigmoid': 'Sigmoid - Colon'
}

OTARI_TISSUE_NAMES = {
    'Brain_Cortex': ['Cerebral_Cortex'],
    'Liver': ['liver'],
    'Whole_Blood': ['blood'],
    'Lung': ['lung'],
    'Heart_Atrial_Appendage': ['heart'],
    'Colon_Sigmoid': ['colon']
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'predictions')
DEFAULT_SUSIE_DIR = os.path.join(SCRIPT_DIR, 'data', 'susie')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')


def get_spliceai_scores(vcf_file: str) -> Dict[str, float]:
    """Parse SpliceAI VCF and extract max delta scores."""
    variant2score = {}
    with open(vcf_file) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            sp = line.strip().split('\t')
            if len(sp) < 8:
                continue
            chrom, pos, ref, alt, info = sp[0], sp[1], sp[3], sp[4], sp[7]
            for field in info.split(';'):
                if field.startswith('SpliceAI='):
                    spliceai_info = field.split('=')[1]
                    parts = spliceai_info.split('|')
                    if len(parts) >= 6:
                        dsmax = max(float(parts[2]), float(parts[3]),
                                   float(parts[4]), float(parts[5]))
                        variant_id = f'{chrom}:{pos}:{ref}:{alt}'
                        variant2score[variant_id] = dsmax
    return variant2score


def get_mmsplice_tissue_scores(tsv_file: str, tissue_name: str) -> Dict[str, float]:
    """Parse MMSplice/MTSplice tissue-specific predictions."""
    variant2score = {}
    with open(tsv_file) as fp:
        header = fp.readline().strip().split('\t')
        if tissue_name not in header:
            tissue_indices = [idx for idx, col in enumerate(header) if tissue_name in col]
        else:
            tissue_indices = [header.index(tissue_name)]
        if not tissue_indices:
            return variant2score
        for line in fp:
            sp = line.strip().split('\t')
            variant_id = sp[0].replace('>', ':')
            tissue_score = np.mean([abs(float(sp[idx])) for idx in tissue_indices])
            variant2score[variant_id] = tissue_score
    return variant2score


def get_pangolin_scores(vcf_file: str) -> Dict[str, float]:
    """Parse Pangolin VCF and extract splice impact scores."""
    variant2score = {}
    with open(vcf_file) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            sp = line.strip().split('\t')
            chrom, pos, ref, alt, info = sp[0], sp[1], sp[3], sp[4], sp[7]
            for field in info.split(';'):
                if field.startswith('Pangolin='):
                    pangolin_info = field.split('=')[1]
                    parts = pangolin_info.split('|')
                    if len(parts) >= 2:
                        scores = parts[1].split(':')
                        if len(scores) == 2:
                            score = float(scores[1])
                            variant_id = f'{chrom}:{pos}:{ref}:{alt}'
                            variant2score[variant_id] = score
    return variant2score


def get_otari_tissue_scores(tsv_file: str, tissues: List[str],
                           score_type: str = 'max_effect',
                           aggregation: str = 'sum') -> Dict[str, float]:
    """Parse Otari tissue-specific predictions."""
    variant2all_scores = defaultdict(list)
    with open(tsv_file) as fp:
        header = fp.readline().split('\t')
        tissue_indices = [header.index(t) for t in tissues if t in header]
        if not tissue_indices:
            return {}
        for line in fp:
            sp = line.strip().split('\t')
            variant_id = sp[0]
            tissue_values = [float(sp[idx]) for idx in tissue_indices]
            if score_type == 'max_effect':
                score = np.max(np.abs(tissue_values))
            else:
                score = np.mean(np.abs(tissue_values))
            variant_id = ':'.join(variant_id.split('_')[:-1])
            variant_id = 'chr' + variant_id
            variant2all_scores[variant_id].append(score)

    variant2score = {}
    for variant_id, scores in variant2all_scores.items():
        if aggregation == 'max':
            variant2score[variant_id] = max(scores)
        elif aggregation == 'mean':
            variant2score[variant_id] = np.mean(scores)
        else:
            variant2score[variant_id] = np.sum(scores)
    return variant2score


def get_positive_negative_variants(tissue_name: str, susie_dir: str) -> Tuple[Set[str], Set[str]]:
    """Load positive (PIP >= 0.9) and negative (PIP <= 0.1) variants."""
    positive_set, negative_set = set(), set()
    for _file in os.listdir(susie_dir):
        if _file.endswith('.parquet') and tissue_name == _file.split('.')[0]:
            df = pd.read_parquet(os.path.join(susie_dir, _file))
            print(f'  Loaded {_file}: {len(df)} rows')
            df_positive = df[df['pip'] >= 0.9]
            for variant_id in df_positive['variant_id']:
                chrom, pos, ref, alt, _ = variant_id.split('_')
                positive_set.add(f'{chrom}:{pos}:{ref}:{alt}')
            df_negative = df[df['pip'] <= 0.1]
            for variant_id in df_negative['variant_id']:
                chrom, pos, ref, alt, _ = variant_id.split('_')
                negative_set.add(f'{chrom}:{pos}:{ref}:{alt}')
    return positive_set, negative_set


def consolidate_scores(data_dir: str, susie_dir: str, output_dir: str):
    """
    Consolidate all prediction scores into ready-to-plot formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading prediction scores...")
    spliceai_scores = get_spliceai_scores(os.path.join(data_dir, 'sQTL_variants.spliceai.vcf'))
    pangolin_scores = get_pangolin_scores(os.path.join(data_dir, 'sQTL_variants.pangolin.vcf'))
    print(f"  SpliceAI: {len(spliceai_scores)} variants")
    print(f"  Pangolin: {len(pangolin_scores)} variants")

    # Storage for consolidated outputs
    all_variant_scores = []  # Per-variant scores
    all_metrics = []         # Per-tissue/predictor metrics
    all_curve_data = []      # ROC/PR curve points

    for tissue in TISSUES:
        print(f"\nProcessing tissue: {tissue}")

        positive_set, negative_set = get_positive_negative_variants(tissue, susie_dir)
        print(f"  Positive: {len(positive_set)}, Negative: {len(negative_set)}")

        # Get tissue-specific scores
        mmsplice_tissue = MMSPLICE_TISSUE_NAMES[tissue]
        mmsplice_scores = get_mmsplice_tissue_scores(
            os.path.join(data_dir, 'sQTL_variants.mmsplice.tsv'), mmsplice_tissue)

        otari_tissues = OTARI_TISSUE_NAMES[tissue]
        otari_scores = get_otari_tissue_scores(
            os.path.join(data_dir, 'sQTL_variants.Otari.tsv'),
            tissues=otari_tissues, score_type='max_effect', aggregation='sum')

        # Find common variants
        common_variants = (set(spliceai_scores.keys()) & set(mmsplice_scores.keys()) &
                         set(pangolin_scores.keys()) & set(otari_scores.keys()))
        positive_common = positive_set & common_variants
        negative_common = negative_set & common_variants

        print(f"  Common variants: {len(positive_common)} pos, {len(negative_common)} neg")

        # Collect per-variant scores
        for variant_id in positive_common:
            all_variant_scores.append({
                'variant_id': variant_id,
                'tissue': tissue,
                'label': 1,
                'Otari': otari_scores[variant_id],
                'SpliceAI': spliceai_scores[variant_id],
                'MTSplice': mmsplice_scores[variant_id],
                'Pangolin': pangolin_scores[variant_id]
            })

        for variant_id in negative_common:
            all_variant_scores.append({
                'variant_id': variant_id,
                'tissue': tissue,
                'label': 0,
                'Otari': otari_scores[variant_id],
                'SpliceAI': spliceai_scores[variant_id],
                'MTSplice': mmsplice_scores[variant_id],
                'Pangolin': pangolin_scores[variant_id]
            })

        # Compute metrics and curves for each predictor
        y_true = [1] * len(positive_common) + [0] * len(negative_common)

        for predictor in ['Otari', 'SpliceAI', 'MTSplice', 'Pangolin']:
            if predictor == 'Otari':
                scores = [otari_scores[v] for v in positive_common] + [otari_scores[v] for v in negative_common]
            elif predictor == 'SpliceAI':
                scores = [spliceai_scores[v] for v in positive_common] + [spliceai_scores[v] for v in negative_common]
            elif predictor == 'MTSplice':
                scores = [mmsplice_scores[v] for v in positive_common] + [mmsplice_scores[v] for v in negative_common]
            else:
                scores = [pangolin_scores[v] for v in positive_common] + [pangolin_scores[v] for v in negative_common]

            # Compute metrics
            auroc = roc_auc_score(y_true, scores)
            auprc = average_precision_score(y_true, scores)

            all_metrics.append({
                'tissue': tissue,
                'predictor': predictor,
                'auroc': auroc,
                'auprc': auprc,
                'n_positive': len(positive_common),
                'n_negative': len(negative_common),
                'n_total': len(y_true)
            })

            # Compute curve data
            fpr, tpr, _ = roc_curve(y_true, scores)
            precision, recall, _ = precision_recall_curve(y_true, scores)

            # Subsample curve points for manageable file size
            for i in range(0, len(fpr), max(1, len(fpr) // 100)):
                all_curve_data.append({
                    'tissue': tissue,
                    'predictor': predictor,
                    'curve_type': 'ROC',
                    'x': fpr[i],
                    'y': tpr[i]
                })

            for i in range(0, len(recall), max(1, len(recall) // 100)):
                all_curve_data.append({
                    'tissue': tissue,
                    'predictor': predictor,
                    'curve_type': 'PR',
                    'x': recall[i],
                    'y': precision[i]
                })

    # Save consolidated files
    print("\nSaving consolidated files...")

    # 1. Per-variant scores
    df_scores = pd.DataFrame(all_variant_scores)
    scores_file = os.path.join(output_dir, 'consolidated_scores.csv')
    df_scores.to_csv(scores_file, index=False)
    print(f"  {scores_file}: {len(df_scores)} rows")

    # 2. Metrics summary
    df_metrics = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(output_dir, 'consolidated_metrics.csv')
    df_metrics.to_csv(metrics_file, index=False)
    print(f"  {metrics_file}: {len(df_metrics)} rows")

    # 3. Curve data for plotting
    df_curves = pd.DataFrame(all_curve_data)
    curves_file = os.path.join(output_dir, 'curve_data.csv')
    df_curves.to_csv(curves_file, index=False)
    print(f"  {curves_file}: {len(df_curves)} rows")

    # Print summary table
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    pivot_auroc = df_metrics.pivot(index='tissue', columns='predictor', values='auroc')
    pivot_auprc = df_metrics.pivot(index='tissue', columns='predictor', values='auprc')
    print("\nAUROC:")
    print(pivot_auroc.round(4).to_string())
    print("\nAUPRC:")
    print(pivot_auprc.round(4).to_string())

    return df_scores, df_metrics, df_curves


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate prediction scores for plotting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                       help='Directory containing prediction files')
    parser.add_argument('--susie-dir', default=DEFAULT_SUSIE_DIR,
                       help='Directory containing SuSiE parquet files')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='Output directory')

    args = parser.parse_args()

    print("Consolidating Prediction Scores")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"SuSiE directory: {args.susie_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    consolidate_scores(args.data_dir, args.susie_dir, args.output_dir)

    print("\n" + "=" * 50)
    print("Output files ready for plotting:")
    print("  - consolidated_scores.csv: Per-variant scores (for custom analysis)")
    print("  - consolidated_metrics.csv: AUROC/AUPRC by tissue/predictor (for tables)")
    print("  - curve_data.csv: ROC/PR curve points (for plotting)")


if __name__ == '__main__':
    main()

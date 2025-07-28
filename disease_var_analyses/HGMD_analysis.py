import os

import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pickle as rick

from utils.utils import draw_lines_and_stars, get_star_labels


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def analyze_HGMD_variants(col='max_effect'):
    pathogenic_all = '../resources/variant_effects/HGMD_DFP_pathogenic_variants/variant_effects_comprehensive.tsv'
    benign_all = '../resources/variant_effects/HGMD_DFP_variants_neutral/variant_effects_comprehensive.tsv'

    pathogenic_all = pd.read_csv(pathogenic_all, sep='\t').drop_duplicates(keep='first')
    benign_all = pd.read_csv(benign_all, sep='\t').drop_duplicates(keep='first')

    # scale scores to background distribution
    background_path = '../resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()
    pathogenic_all[TISSUE_NAMES] = (pathogenic_all[TISSUE_NAMES] - mean_global) / std_global
    benign_all[TISSUE_NAMES] = (benign_all[TISSUE_NAMES] - mean_global) / std_global
 
    # add max effect and mean effect (magnitude of effect)
    pathogenic_all['max_effect'] = pathogenic_all[TISSUE_NAMES].abs().max(axis=1)
    pathogenic_all['mean_effect'] = pathogenic_all[TISSUE_NAMES].abs().mean(axis=1)
    benign_all['max_effect'] = benign_all[TISSUE_NAMES].abs().max(axis=1)
    benign_all['mean_effect'] = benign_all[TISSUE_NAMES].abs().mean(axis=1)
    pathogenic_all[TISSUE_NAMES] = pathogenic_all[TISSUE_NAMES].abs()
    benign_all[TISSUE_NAMES] = benign_all[TISSUE_NAMES].abs()

    # get principal transcript for each variant
    with open('../resources/gene2_principal_transcript.pkl', 'rb') as f:
        gene2principal = rick.load(f)
    all_principal_transcripts = []
    for k, v in gene2principal.items():
        all_principal_transcripts.extend(v)
    principal_copy_path = pathogenic_all.copy()
    benign_copy_path = benign_all.copy()
    principal_copy_path['principal'] = principal_copy_path['transcript_id'].apply(lambda x: 1 if x in all_principal_transcripts else 0)
    benign_copy_path['principal'] = benign_copy_path['transcript_id'].apply(lambda x: 1 if x in all_principal_transcripts else 0)
    principal_copy_path = principal_copy_path[principal_copy_path['principal'] == 1]
    benign_copy_path = benign_copy_path[benign_copy_path['principal'] == 1]

    # groupby variant_id and get the row with the maximum 'max_effect'
    # in case there are multiple principal transcripts per gene
    principal_copy_path = principal_copy_path.loc[principal_copy_path.groupby('variant_id')['max_effect'].idxmax()].reset_index(drop=True)
    benign_copy_path = benign_copy_path.loc[benign_copy_path.groupby('variant_id')['max_effect'].idxmax()].reset_index(drop=True)

    # compute mean and ste for principal transcripts
    pathogenic_principal_mean = principal_copy_path[col].mean()
    pathogenic_principal_ste = principal_copy_path[col].sem() 
    benign_principal_mean = benign_copy_path[col].mean()
    benign_principal_ste = benign_copy_path[col].sem() 

    # transcript-level analysis
    # groupby 'variant_id' and get the row with the maximum effect (top transcript)
    pathogenic_all = pathogenic_all.loc[pathogenic_all.groupby('variant_id')['max_effect'].idxmax()].reset_index(drop=True)
    benign_all = benign_all.loc[benign_all.groupby('variant_id')['max_effect'].idxmax()].reset_index(drop=True)

    # compute mean and standard error for top-ranked transcripts
    pathogenic_mean = pathogenic_all[col].mean()
    pathogenic_ste = pathogenic_all[col].sem() 
    benign_mean = benign_all[col].mean()
    benign_ste = benign_all[col].sem() 

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x_positions = [0,1,2]
    ax.errorbar(x=x_positions[0], y=benign_mean-benign_mean, yerr=benign_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[0], y=benign_mean-benign_mean, s=170, color='steelblue', zorder=2)
    ax.errorbar(x=x_positions[1], y=pathogenic_principal_mean-benign_mean, yerr=pathogenic_principal_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[1], y=pathogenic_principal_mean-benign_mean, s=170, color='orange', zorder=2)
    ax.errorbar(x=x_positions[2], y=pathogenic_mean-benign_mean, yerr=pathogenic_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[2], y=pathogenic_mean-benign_mean, s=170, color='mediumvioletred', zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Neutral\n(isoform)', 'Pathogenic\n(gene)', 'Pathogenic\n(isoform)'], fontsize=13.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel('')
    ax.set_ylabel('abs(effect size)', fontsize=14)

    # compute p values
    ttest_max = ttest_ind(pathogenic_all[col], benign_all[col], equal_var=False, alternative='greater').pvalue
    ttest_principal = ttest_ind(principal_copy_path[col], benign_all[col], equal_var=False, alternative='greater').pvalue
    ttest_patho_principal_vs_max = ttest_ind(pathogenic_all[col], principal_copy_path[col], equal_var=False, alternative='greater').pvalue
    
    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }
    pvalues = [ttest_max, ttest_principal, ttest_patho_principal_vs_max]
    corrected_pvals = multipletests(pvalues, method='fdr_bh')[1]
    star_labels = get_star_labels(corrected_pvals, custom_thresholds)
    pairs = [(0, 2), (0, 1), (1, 2)] 
    offsets = [0.14, 0.04, 0.027]
    y_positions = [pathogenic_mean-benign_mean + pathogenic_ste + offsets[0], 
                    pathogenic_principal_mean-benign_mean + pathogenic_principal_ste + offsets[1], 
                    pathogenic_mean-benign_mean + pathogenic_ste + offsets[2]]
    draw_lines_and_stars(ax, pairs, y_positions, star_labels)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.title(f'HGMD DFP', fontsize=17)

    os.makedirs('figures', exist_ok=True)
    
    plt.savefig(f'figures/HGMD_DFP_Otari_predicted_variant_effects.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    analyze_HGMD_variants()

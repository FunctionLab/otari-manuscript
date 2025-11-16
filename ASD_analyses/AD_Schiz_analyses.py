import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']

BRAIN_TISSUES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                 'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                 'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 'Temporal_Lobe', 
                 'Thalamus']
    
CONTROL_PATH = '../resources/variant_effects/DNVs_siblings_satterstrom/variant_effects_comprehensive.tsv'


def analyze_variants(disease, col='max_effect'):
    """
    Analyze variants for Alzheimer's disease or schizophrenia.

    Args:
        disease: Disease name, either 'AD' or 'schizophrenia'
        col: Column score name to analyze, default is 'max_effect'
    """
    if disease == 'AD':
        pathogenic_path = '../resources/variant_effects/alzheimers_disease/variant_effects_comprehensive.tsv'
    elif disease == 'schizophrenia':
        pathogenic_path = '../resources/variant_effects/schizophrenia/variant_effects_comprehensive.tsv'
    else:
        raise ValueError(f"Invalid disease: {disease}. Must be 'AD' or 'schizophrenia'.")

    pathogenic = pd.read_csv(pathogenic_path, sep='\t').drop_duplicates(keep='first')
    control = pd.read_csv(CONTROL_PATH, sep='\t').drop_duplicates(keep='first')

    # Scale scores to background
    background_path = '../resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()
    pathogenic[TISSUE_NAMES] = (pathogenic[TISSUE_NAMES] - mean_global) / std_global
    control[TISSUE_NAMES] = (control[TISSUE_NAMES] - mean_global) / std_global

    # Compute max and mean effect
    pathogenic['max_effect'] = pathogenic[BRAIN_TISSUES].abs().max(axis=1) 
    pathogenic['mean_effect'] = pathogenic[BRAIN_TISSUES].abs().mean(axis=1)
    control['max_effect'] = control[BRAIN_TISSUES].abs().max(axis=1)
    control['mean_effect'] = control[BRAIN_TISSUES].abs().mean(axis=1)
    pathogenic[TISSUE_NAMES] = pathogenic[TISSUE_NAMES].abs()
    control[TISSUE_NAMES] = control[TISSUE_NAMES].abs()

    # Compute mean and standard error for pathogenic and benign
    pathogenic_mean = pathogenic[col].mean()
    pathogenic_ste = pathogenic[col].sem() 
    control_mean = control[col].mean()
    control_ste = control[col].sem()
    
    # Normalize to control mean
    pathogenic_mean = pathogenic_mean - control_mean
    control_mean = control_mean - control_mean

    pval = ttest_ind(pathogenic[col], control[col], equal_var=False, alternative='greater').pvalue

    # Plot
    _, ax = plt.subplots(figsize=(3.9, 4.2))
    x_positions = [1, 2]
    ax.errorbar(x=x_positions[0], y=control_mean, yerr=control_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[0], y=control_mean, s=170, color='gray', zorder=2)
    ax.errorbar(x=x_positions[1], y=pathogenic_mean, yerr=pathogenic_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[1], y=pathogenic_mean, s=170, color='orange', zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Control', f'{disease}'], fontsize=15.5)
    ax.set_xlabel('')
    ax.set_ylabel('abs(effect size)', fontsize=14.5)
    y_pos = pathogenic_mean+pathogenic_ste
    ax.text(1.5, y_pos+0.001, f'p={pval:.2e}', fontsize=13, ha='center')
    ax.set_xlim(0.5, 2.5)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.title(f'{disease} variants', fontsize=17)

    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{disease}_predicted_effects.png', dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    for disease in ['AD', 'schizophrenia']:
        analyze_variants(disease)

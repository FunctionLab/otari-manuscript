import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches


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
    
NON_BRAIN_TISSUES = ['bladder', 'blood', 'colon', 'heart', 'kidney', 'liver', 'lung',
                    'pancreas', 'prostate', 'skeletal_muscle', 'small_intestine', 
                    'spleen', 'stomach', 'thyroid']


def analyze_ASD_variants(col='max_effect'):
    probands = '../resources/variant_effects/DNVs_probands_satterstrom/variant_effects_comprehensive.tsv'
    siblings = '../resources/variant_effects/DNVs_siblings_satterstrom/variant_effects_comprehensive.tsv'
        
    probands = pd.read_csv(probands, sep='\t').drop_duplicates(keep='first')
    siblings = pd.read_csv(siblings, sep='\t').drop_duplicates(keep='first')

    # scale scores to background distribution
    background_path = '../resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()
    probands[TISSUE_NAMES] = (probands[TISSUE_NAMES] - mean_global) / std_global
    siblings[TISSUE_NAMES] = (siblings[TISSUE_NAMES] - mean_global) / std_global

    # add max effect and mean effect (abs value)
    probands['max_effect'] = probands[BRAIN_TISSUES].abs().max(axis=1) 
    probands['mean_effect'] = probands[BRAIN_TISSUES].abs().mean(axis=1)
    siblings['max_effect'] = siblings[BRAIN_TISSUES].abs().max(axis=1)
    siblings['mean_effect'] = siblings[BRAIN_TISSUES].abs().mean(axis=1)
    probands[TISSUE_NAMES] = probands[TISSUE_NAMES].abs()
    siblings[TISSUE_NAMES] = siblings[TISSUE_NAMES].abs()

    # Compute mean and standard error
    proband_mean = probands[col].mean()
    proband_ste = probands[col].sem() 
    sibling_mean = siblings[col].mean()
    sibling_ste = siblings[col].sem() 
    
    # normalize to sibling mean
    proband_mean = proband_mean - sibling_mean
    sibling_mean = sibling_mean - sibling_mean
    
    pval = ttest_ind(probands[col], siblings[col], alternative='greater').pvalue

    fig, ax = plt.subplots(figsize=(3.9, 4.2))
    x_positions = [1, 2]
    ax.errorbar(x=x_positions[0], y=sibling_mean, yerr=sibling_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[0], y=sibling_mean, s=170, color='gray', zorder=2)
    ax.errorbar(x=x_positions[1], y=proband_mean, yerr=proband_ste, capsize=3, color='black', zorder=1)
    ax.scatter(x=x_positions[1], y=proband_mean, s=170, color='mediumvioletred', zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Siblings', 'Probands'], fontsize=15.5)
    ax.set_xlabel('')
    ax.set_ylabel('abs(effect size)', fontsize=14.5)
    y_pos = proband_mean + proband_ste
    ax.text(1.5, y_pos+0.001, f'p={pval:.2e}', fontsize=13, ha='center')
    ax.set_xlim(0.5, 2.5)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    os.makedirs('figures', exist_ok=True)
  
    plt.title('SPARK (Satterstrom genes)', fontsize=17)
    plt.savefig('figures/SPARK_DNVs_satterstrom_predicted_effects.png', dpi=600, bbox_inches='tight')
    plt.close()

    # tissue-specific analysis
    effect_sizes = []
    for tissue in tissue_names:
        if tissue == 'testis' or tissue == 'ovary':
            continue
        effect_size = np.log2(probands[tissue].mean() / siblings[tissue].mean())
        effect_sizes.append(effect_size)

    tissue_names = [t for t in tissue_names if t not in ['testis', 'ovary']]
    tissue_to_effect = {tissue: effect for tissue, effect in zip(tissue_names, effect_sizes)}
    tissue_to_effect = dict(sorted(tissue_to_effect.items(), key=lambda x: x[1], reverse=False))

    # 'red' for brain tissues, 'dimgray' for non-brain tissues
    colors = ['darkcyan' if tissue in BRAIN_TISSUES else 'dimgray' for tissue in tissue_to_effect.keys()]

    # plot bubble plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(tissue_to_effect.values(), range(len(tissue_names)), s=230, c=colors, alpha=0.7, edgecolor=colors, linewidth=2)
    ax.set_yticks(range(len(tissue_names)))
    ax.tick_params(axis='x', labelsize=14)
    tissue_labels = [x.replace('_', ' ').capitalize() for x in tissue_to_effect.keys()]
    ax.set_yticklabels(tissue_labels, fontsize=14)
    ax.set_ylabel('')
    ax.set_xlabel('log2 fc (probands v. siblings)', fontsize=18)
    brain_patch = mpatches.Patch(color='darkcyan', label='Brain/CNS')
    non_brain_patch = mpatches.Patch(color='dimgray', label='Non-brain')
    ax.legend(handles=[brain_patch, non_brain_patch], fontsize=19, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig('figures/SPARK_DNVs_tissue_specific_effects.png', dpi=600, bbox_inches='tight')
    plt.close()

    # compare brain vs non-brain tissues
    brain_effect_sizes = [tissue_to_effect[tissue] for tissue in BRAIN_TISSUES]
    non_brain_effect_sizes = [tissue_to_effect[tissue] for tissue in NON_BRAIN_TISSUES]
    brain_mean = np.mean(brain_effect_sizes)-np.mean(non_brain_effect_sizes)
    brain_ste = np.std(brain_effect_sizes) / np.sqrt(len(brain_effect_sizes))
    non_brain_mean = np.mean(non_brain_effect_sizes)-np.mean(non_brain_effect_sizes)
    non_brain_ste = np.std(non_brain_effect_sizes) / np.sqrt(len(non_brain_effect_sizes))
    ttest = ttest_ind(brain_effect_sizes, non_brain_effect_sizes, alternative='greater').pvalue

    fig, ax = plt.subplots(figsize=(1.7, 2.5))
    x_positions = [2, 1]
    ax.errorbar(x=x_positions[0], y=brain_mean, yerr=brain_ste, capsize=2, color='black', zorder=1)
    ax.scatter(x=x_positions[0], y=brain_mean, s=80, color='darkcyan', zorder=2)
    ax.errorbar(x=x_positions[1], y=non_brain_mean, yerr=non_brain_ste, capsize=2, color='black', zorder=1)
    ax.scatter(x=x_positions[1], y=non_brain_mean, s=80, color='dimgray', zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([])
    y_pos = brain_mean + brain_ste
    ax.text(1.5, y_pos+0.0015, f'p={ttest:.2e}', fontsize=14, ha='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig('figures/SPARK_DNVs_brain_vs_nonbrain.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    analyze_ASD_variants()

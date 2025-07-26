import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def get_H19_variant_effects(gene_id = 'ENSG00000130600'):
    variant_list = ['11_1995678_C_T_hg38', '11_1997623_G_A_hg38', '11_1999845_C_T_hg38'] # H19 cancer polymorphisms

    t1_effects = [] 
    t2_effects = [] 
    for variant in variant_list:
        clinvar_df = pd.read_csv('../resources/variant_effects/clinvar_H19_variant/interpretability_analysis.tsv', sep='\t')
        vep = pd.read_csv('../resources/variant_effects/clinvar_H19_variant/variant_effects_comprehensive.tsv', sep='\t')

        clinvar_df = clinvar_df[clinvar_df['variant_id'] == variant].reset_index()
        vep = vep[vep['variant_id'] == variant].reset_index()

        clinvar_df = clinvar_df[clinvar_df['gene_id'] == gene_id].reset_index()
        transcripts = clinvar_df['transcript_id'].values.tolist()
        vep = vep[vep['transcript_id'].isin(transcripts)]

        # scale scores to background distribution
        background_path = '../resources/background_distribution.tsv'
        background = pd.read_csv(background_path, sep='\t')
        mean_global = background[TISSUE_NAMES].values.flatten().mean()
        std_global = background[TISSUE_NAMES].values.flatten().std()
        vep[TISSUE_NAMES] = (vep[TISSUE_NAMES] - mean_global) / std_global
 
        # get average across all tissues
        t1_effects.append(vep.loc[vep['transcript_id'] == 'ENST00000691195', TISSUE_NAMES].values)
        t2_effects.append(vep.loc[vep['transcript_id'] == 'ENST00000710492', TISSUE_NAMES].values)
            
    avg_t1 = [x.flatten() for x in t1_effects if x.size > 0]
    avg_t2 = [x.flatten() for x in t2_effects if x.size > 0]

    data = []
    positions = []
    colors = []

    for i, variant in enumerate(variant_list):
        pos = i * 2 + 1
        if i < len(avg_t1):
            data.append(avg_t1[i])
            positions.append(pos)
            colors.append("darkcyan")  # color for t1
        if i < len(avg_t2):
            data.append(avg_t2[i])
            positions.append(pos + 0.5)
            colors.append("dimgray")  # color for t2

    # plot
    fig, ax = plt.subplots(1,1,figsize=(5.8, 4.5))
    box = plt.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                        showfliers=False, 
                        whiskerprops = dict(color = "black", linewidth=2), 
                        capprops = dict(color = "black", linewidth=2),
                        medianprops=dict(color='white', linewidth=1), 
                        boxprops=dict(edgecolor='white', linewidth=0.5))
    for i, d in enumerate(data):
        x = np.random.normal(positions[i], 0.04, size=len(d))  # add jitter
        ax.scatter(x, d, alpha=0.8, color=colors[i], s=23, zorder=3, edgecolor='white', linewidth=0.5)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    plt.xticks([i * 2 + 1.25 for i in range(len(variant_list))], variant_list)
    ax.set_xticklabels(['rs217727', 'rs2839698', 'rs2107425'], fontsize=13.5)
    plt.yticks(fontsize=12)
    plt.legend([box["boxes"][0], box["boxes"][1]], [t1, t2], frameon=False, fontsize=12, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.09))
    plt.ylabel("effect size", fontsize=14)
    plt.title("H19 lncRNA", fontsize=17, pad=17)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    os.makedirs('figures', exist_ok=True)
    
    plt.savefig('figures/H19_variant_effects_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    get_H19_variant_effects()

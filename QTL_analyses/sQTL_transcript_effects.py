import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as rick
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests   

from utils.utils import read_gtf


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def sQTL_effects_analysis():
    tissues = ['Brain_Cortex', 'Liver', 'Whole_Blood', 'Lung', 'Heart_Atrial_Appendage', 'Colon_Sigmoid']
    variant_files = [f'../resources/GTEx_fine_mapped_sQTLs/{tissue}.v10.sQTLs.SuSiE_summary.tsv' 
                    for tissue in tissues]
    effect_files = [f'../resources/variant_effects/GTEx_fine_mapped_sQTLs/{tissue.split("_")[0]}/variant_effects_comprehensive.tsv'
                    for tissue in tissues]

    sqtl_variants = {tissue: pd.read_csv(file, sep='\t') for tissue, file in zip(tissues, variant_files)}
    sqtl_variant_effects = {tissue: pd.read_csv(file, sep='\t') for tissue, file in zip(tissues, effect_files)}

    for tissue, effects in sqtl_variant_effects.items():
        effects['variant_id'] = effects['variant_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        effects['variant_id'] = 'chr' + effects['variant_id']

    brain_sqtl_variants = sqtl_variants['Brain_Cortex']
    liver_sqtl_variants = sqtl_variants['Liver']
    blood_sqtl_variants = sqtl_variants['Whole_Blood']
    lung_sqtl_variants = sqtl_variants['Lung']
    heart_sqtl_variants = sqtl_variants['Heart_Atrial_Appendage']
    colon_sqtl_variants = sqtl_variants['Colon_Sigmoid']

    brain_sqtl_variant_effects = sqtl_variant_effects['Brain_Cortex']
    liver_sqtl_variant_effects = sqtl_variant_effects['Liver']
    blood_sqtl_variant_effects = sqtl_variant_effects['Whole_Blood']
    lung_sqtl_variant_effects = sqtl_variant_effects['Lung']
    heart_sqtl_variant_effects = sqtl_variant_effects['Heart_Atrial_Appendage']
    colon_sqtl_variant_effects = sqtl_variant_effects['Colon_Sigmoid']

    gtf_reader = read_gtf()
    gene_to_transcript = '../resources/gene2transcripts.pkl'
    with open(gene_to_transcript, 'rb') as file:
        gene2transcripts = rick.load(file)
        
    selected_tissues = ['Brain CTX', 'Liver', 'Blood', 'Lung', 'Heart', 'Colon']
    for tissue in selected_tissues:
        if tissue == 'Brain CTX':
            sqtl_variants = brain_sqtl_variants
            sqtl_variant_effects = brain_sqtl_variant_effects
        elif tissue == 'Liver':
            sqtl_variants = liver_sqtl_variants
            sqtl_variant_effects = liver_sqtl_variant_effects
        elif tissue == 'Blood':
            sqtl_variants = blood_sqtl_variants
            sqtl_variant_effects = blood_sqtl_variant_effects
        elif tissue == 'Lung':
            sqtl_variants = lung_sqtl_variants
            sqtl_variant_effects = lung_sqtl_variant_effects
        elif tissue == 'Heart':
            sqtl_variants = heart_sqtl_variants
            sqtl_variant_effects = heart_sqtl_variant_effects
        elif tissue == 'Colon':
            sqtl_variants = colon_sqtl_variants
            sqtl_variant_effects = colon_sqtl_variant_effects

        # drop duplucates in sqtl_variant_effects
        sqtl_variant_effects = sqtl_variant_effects.drop_duplicates(
            subset=['variant_id', 'transcript_id'], 
            keep='first'
            )

        # scale to background distribution
        background_path = '../resources/background_distribution.tsv'
        background = pd.read_csv(background_path, sep='\t')
        mean_global = background[TISSUE_NAMES].values.flatten().mean()
        std_global = background[TISSUE_NAMES].values.flatten().std()
        sqtl_variant_effects[TISSUE_NAMES] = (sqtl_variant_effects[TISSUE_NAMES] - mean_global) / std_global

        var_effects_utilized = {} # overlapping transcripts
        var_effect_not_utilized = {} # non-overlapping transcripts
        
        for i, r in sqtl_variants.iterrows(): # get affected region
            print(f'Processing {i+1}/{len(sqtl_variants)}')
            
            regions = r['phenotype_id'].split(':')[1:3]
            regions = [int(x) for x in regions]
            gene_id = r['gene']
            variant = r['variant_id']

            transcript_ids = [] 
            transcript_ids.extend(gene2transcripts[gene_id])

            try:
                gene = gtf_reader.get_gene(gene_id)
            except ValueError:
                print(f'Gene {gene_id} not found in gtf')
                continue
            
            exons_rolling = []
            for tid in transcript_ids:
                exons_rolling += gene[tid].exons
            
            # get all exons that overlap/border spliced region
            overlapping_exons = []
            for exon in exons_rolling:
                overlap = max(0, min(exon[1], regions[1]) - max(exon[0], regions[0]))
                if overlap > 0:
                    overlapping_exons.append(exon)
                if exon[0] == regions[1]:
                    overlapping_exons.append(exon)
                if exon[1] == regions[0]:
                    overlapping_exons.append(exon)
            
            transcripts_with_exon = []
            transcripts_without_exon = []
            for transcript in transcript_ids:
                tid_exons = gene[transcript].exons
                overlapping_exon = [exon for exon in overlapping_exons if exon in tid_exons]
                
                if len(overlapping_exon) > 0:
                    transcripts_with_exon.append(transcript)
                else:
                    transcripts_without_exon.append(transcript)
                
            # fetch variant effects
            for transcript in transcripts_with_exon:
                sqtl_effect = sqtl_variant_effects.loc[sqtl_variant_effects['variant_id'] == variant]
                sqtl_effect = sqtl_effect.loc[sqtl_effect['transcript_id'] == transcript]
                if len(sqtl_effect) == 0:
                    continue
                tissue_effects = [abs(sqtl_effect[tissue].values[0]) for tissue in TISSUE_NAMES]
                var_effects_utilized[f'{variant}_{transcript}'] = tissue_effects
            
            for transcript in transcripts_without_exon:
                sqtl_effect = sqtl_variant_effects.loc[sqtl_variant_effects['variant_id'] == variant]
                sqtl_effect = sqtl_effect.loc[sqtl_effect['transcript_id'] == transcript]
                if len(sqtl_effect) == 0:
                    continue
                tissue_effects = [abs(sqtl_effect[tissue].values[0]) for tissue in TISSUE_NAMES]
                var_effect_not_utilized[f'{variant}_{transcript}'] = tissue_effects
        
        if tissue == 'Brain CTX':
            brain_var_effects_utilized = var_effects_utilized
            brain_var_effect_not_utilized = var_effect_not_utilized
        elif tissue == 'Liver':
            liver_var_effects_utilized = var_effects_utilized
            liver_var_effects_not_utilized = var_effect_not_utilized
        elif tissue == 'Blood':
            blood_var_effects_utilized = var_effects_utilized
            blood_var_effect_not_utilized = var_effect_not_utilized
        elif tissue == 'Lung':
            lung_var_effects_utilized = var_effects_utilized
            lung_var_effect_not_utilized = var_effect_not_utilized
        elif tissue == 'Heart':
            heart_var_effects_utilized = var_effects_utilized
            heart_var_effect_not_utilized = var_effect_not_utilized
        elif tissue == 'Colon':
            colon_var_effects_utilized = var_effects_utilized
            colon_var_effect_not_utilized = var_effect_not_utilized

    brain_idx = TISSUE_NAMES.index('Cerebral_Cortex')
    liver_idx = TISSUE_NAMES.index('liver')
    blood_idx = TISSUE_NAMES.index('blood')
    lung_idx = TISSUE_NAMES.index('lung')
    heart_idx = TISSUE_NAMES.index('heart')
    colon_idx = TISSUE_NAMES.index('colon')

    # get tissue-specific effects
    brain_avg_effects_utilized = [effects[brain_idx] for effects in brain_var_effects_utilized.values()]
    brain_avg_effects_non_utilized = [effects[brain_idx] for effects in brain_var_effect_not_utilized.values()]
    liver_avg_effects_utilized = [effects[liver_idx] for effects in liver_var_effects_utilized.values()]
    liver_avg_effects_non_utilized = [effects[liver_idx] for effects in liver_var_effects_not_utilized.values()]
    blood_avg_effects_utilized = [effects[blood_idx] for effects in blood_var_effects_utilized.values()]
    blood_avg_effects_non_utilized = [effects[blood_idx] for effects in blood_var_effect_not_utilized.values()]
    lung_avg_effects_utilized = [effects[lung_idx] for effects in lung_var_effects_utilized.values()]
    lung_avg_effects_non_utilized = [effects[lung_idx] for effects in lung_var_effect_not_utilized.values()]
    heart_avg_effects_utilized = [effects[heart_idx] for effects in heart_var_effects_utilized.values()]
    heart_avg_effects_non_utilized = [effects[heart_idx] for effects in heart_var_effect_not_utilized.values()]
    colon_avg_effects_utilized = [effects[colon_idx] for effects in colon_var_effects_utilized.values()]
    colon_avg_effects_non_utilized = [effects[colon_idx] for effects in colon_var_effect_not_utilized.values()]

    # calculate metrics
    def calculate_metrics(utilized, non_utilized):
        mean_utilized = np.mean(utilized)
        mean_non_utilized = np.mean(non_utilized)
        ste_utilized = np.std(utilized) / np.sqrt(len(utilized))
        ste_non_utilized = np.std(non_utilized) / np.sqrt(len(non_utilized))
        p_val = ttest_ind(utilized, non_utilized, alternative='greater').pvalue
        return [mean_utilized, mean_non_utilized], [ste_utilized, ste_non_utilized], p_val

    tissue_data = [
        (brain_avg_effects_utilized, brain_avg_effects_non_utilized),
        (liver_avg_effects_utilized, liver_avg_effects_non_utilized),
        (blood_avg_effects_utilized, blood_avg_effects_non_utilized),
        (lung_avg_effects_utilized, lung_avg_effects_non_utilized),
        (heart_avg_effects_utilized, heart_avg_effects_non_utilized),
        (colon_avg_effects_utilized, colon_avg_effects_non_utilized),
    ]

    means, ste, p_vals = zip(*[calculate_metrics(utilized, non_utilized) for utilized, non_utilized in tissue_data])
    p_vals = multipletests(p_vals, method='fdr_bh')[1]  # adjust p-values

    # plot 
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.1))
    x = np.arange(len(selected_tissues))
    bar_width = 0.1
    colors = ['mediumvioletred', 'steelblue']
    labels = ['Overlapping transcripts', 'Non-overlapping transcripts']
    for i, tissue in enumerate(selected_tissues):
        ax.errorbar(x[i] - bar_width, means[i][0], yerr=ste[i][0], color='black', capsize=3, zorder=1)
        ax.scatter(x[i] - bar_width, means[i][0], color=colors[0], s=140, zorder=2)
        ax.errorbar(x[i] + bar_width, means[i][1], yerr=ste[i][1], color='black', capsize=3, zorder=1)
        ax.scatter(x[i] + bar_width, means[i][1], color=colors[1], s=140, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_tissues, fontsize=15)
    ax.set_xlim(-0.5, len(selected_tissues)-0.5)
    ax.set_ylabel('abs(effect size)', fontsize=15)
    ax.set_title('sQTLs', fontsize=17, pad=33)
    ax.tick_params(axis='y', labelsize=14)
    for i, p in enumerate(p_vals):
        if p < 0.05:
            ax.text(i, 1.1, f'q={p:.2e}', fontsize=12, ha='center')
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', markersize=11.5) for color in colors]
    ax.legend(markers, labels, loc='upper center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.18))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    os.makedirs('figures', exist_ok=True)
  
    plt.tight_layout()
    plt.savefig(f'figures/Otari_sQTL_predicted_effects.png', dpi=600, bbox_inches='tight')
    plt.close()


if __main__ == "__name__":
    sQTL_effects_analysis()

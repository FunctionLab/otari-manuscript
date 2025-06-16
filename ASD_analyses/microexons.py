import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import pickle as rick

from utils.utils import read_gtf, draw_lines_and_stars, get_star_labels


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


def analyze_microexons():
    # load variants from brain-expressed genes
    probands_vep = '../resources/combined_proband_variants_brain_expressed_genes.tsv'
    siblings_vep = '../resources/combined_sibling_variants_brain_expressed_genes.tsv'   
    probands_interpret = '../resources/combined_proband_interpretability_brain_expressed_genes.tsv'
    siblings_interpret = '../resources/combined_sibling_interpretability_brain_expressed_genes.tsv'
    
    probands_interpret = pd.read_csv(probands_interpret, sep='\t').drop_duplicates(keep='first')
    siblings_interpret = pd.read_csv(siblings_interpret, sep='\t').drop_duplicates(keep='first')
        
    probands_vep = pd.read_csv(probands_vep, sep='\t').drop_duplicates(keep='first')
    siblings_vep = pd.read_csv(siblings_vep, sep='\t').drop_duplicates(keep='first')

    # merge to get transcript_id to gene_id mapping for probands and siblings
    merged_probands = pd.merge(
        probands_vep, 
        probands_interpret[['variant_id', 'transcript_id', 'gene_id']], 
        on=['variant_id', 'transcript_id'], 
        how='inner')
    merged_siblings = pd.merge(
        siblings_vep, 
        siblings_interpret[['variant_id', 'transcript_id', 'gene_id']], 
        on=['variant_id', 'transcript_id'], 
        how='inner')
    probands_tid_to_gid = dict(zip(merged_probands['transcript_id'], merged_probands['gene_id']))
    siblings_tid_to_gid = dict(zip(merged_siblings['transcript_id'], merged_siblings['gene_id']))

    # scale scores
    background_path = 'resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()
    probands_vep[TISSUE_NAMES] = (probands_vep[TISSUE_NAMES] - mean_global) / std_global
    siblings_vep[TISSUE_NAMES] = (siblings_vep[TISSUE_NAMES] - mean_global) / std_global

    # add max effect and mean effect
    probands_vep['max_effect'] = probands_vep[BRAIN_TISSUES].abs().max(axis=1) 
    probands_vep['mean_effect'] = probands_vep[BRAIN_TISSUES].abs().mean(axis=1)
    siblings_vep['max_effect'] = siblings_vep[BRAIN_TISSUES].abs().max(axis=1)
    siblings_vep['mean_effect'] = siblings_vep[BRAIN_TISSUES].abs().mean(axis=1)
    probands_vep[TISSUE_NAMES] = probands_vep[TISSUE_NAMES].abs()
    siblings_vep[TISSUE_NAMES] = siblings_vep[TISSUE_NAMES].abs()

    # gtf reader to get exon lengths
    gtf_reader = read_gtf()

    # iterate through proband variants
    WINDOW = 600
    microexons = []
    non_microexons = []
    impacted_genes = []
    for i, row in probands_vep.iterrows():
        variant_pos = int(row['variant_id'].split('_')[1])
        transcript_id = row['transcript_id']
        gene_id = probands_tid_to_gid[transcript_id]
        gene = gtf_reader.get_gene(gene_id)
        transcripts = gene.transcripts
        if transcript_id not in transcripts:
            continue
        exons = transcripts[transcript_id].exons
        # check if variant is near any microexon in the transcript
        is_near_microexon = False
        for exon in exons:
            exon_length = np.abs(exon[1] - exon[0])
            if 3 <= exon_length <= 27:  # if microexon (break if found)
                if np.abs(variant_pos - exon[0]) <= WINDOW or np.abs(variant_pos - exon[1]) <= WINDOW:
                    microexons.append(row['max_effect'])
                    is_near_microexon = True
                    impacted_genes.append(gene_id)
                    break
        if not is_near_microexon:
            # check if within window nt of any long exon (break if found)
            for exon in exons:
                exon_length = np.abs(exon[1] - exon[0])
                if exon_length > 27:
                    if np.abs(variant_pos - exon[0]) <= WINDOW or np.abs(variant_pos - exon[1]) <= WINDOW:
                        non_microexons.append(row['max_effect'])
                        break 
    print(f'Proband microexons: {len(microexons)}, non-microexons: {len(non_microexons)}')
    microexon_mean = np.mean(microexons)
    non_microexon_mean = np.mean(non_microexons)
    microexon_ste = np.std(microexons) / np.sqrt(len(microexons))
    non_microexon_ste = np.std(non_microexons) / np.sqrt(len(non_microexons))

    microexons_siblings = []
    non_microexons_siblings = []
    for i, row in siblings_vep.iterrows():
        variant_pos = int(row['variant_id'].split('_')[1])
        transcript_id = row['transcript_id']
        gene_id = siblings_tid_to_gid[transcript_id]
        gene = gtf_reader.get_gene(gene_id)
        transcripts = gene.transcripts
        if transcript_id not in transcripts:
            continue
        exons = transcripts[transcript_id].exons
        # check if variant is near any microexon in the transcript (break if found)
        is_near_microexon = False
        for exon in exons:
            exon_length = np.abs(exon[1] - exon[0])
            if 3 <= exon_length <= 27:  # if microexon
                if np.abs(variant_pos - exon[0]) <= WINDOW or np.abs(variant_pos - exon[1]) <= WINDOW:
                    microexons_siblings.append(row['max_effect'])
                    is_near_microexon = True
                    break
        if not is_near_microexon:
            # check if within window nt of any long exon (break if found)
            for exon in exons:
                exon_length = np.abs(exon[1] - exon[0])
                if exon_length > 27:
                    if np.abs(variant_pos - exon[0]) <= WINDOW or np.abs(variant_pos - exon[1]) <= WINDOW:
                        non_microexons_siblings.append(row['max_effect'])
                        break
    print(f'Sibling microexons: {len(microexons_siblings)}, Sibling non-microexons: {len(non_microexons_siblings)}')
    microexon_mean_sibs = np.mean(microexons_siblings)
    non_microexon_mean_sibs = np.mean(non_microexons_siblings)
    microexon_ste_sibs = np.std(microexons_siblings) / np.sqrt(len(microexons_siblings))
    non_microexon_ste_sibs = np.std(non_microexons_siblings) / np.sqrt(len(non_microexons_siblings))

    # get pvals
    pvals = [ttest_ind(microexons, microexons_siblings, alternative='greater', equal_var=False)[1], 
             ttest_ind(microexons, non_microexons, alternative='greater', equal_var=False)[1],
             ttest_ind(microexons_siblings, non_microexons_siblings, alternative='greater', equal_var=False)[1]]
    pvals = multipletests(pvals, method='fdr_bh')[1]

    # plot
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    ax.errorbar([0], [microexon_mean], yerr=[microexon_ste], color='black', capsize=3)
    ax.errorbar([1], [microexon_mean_sibs], yerr=[microexon_ste_sibs], color='black', capsize=3)
    ax.errorbar([2], [non_microexon_mean], yerr=[non_microexon_ste], color='black', capsize=3)
    ax.errorbar([3], [non_microexon_mean_sibs], yerr=[non_microexon_ste_sibs], color='black', capsize=3)
    ax.scatter(x=[0, 2], y=[microexon_mean, non_microexon_mean], s=130, color='mediumvioletred', label='Probands', zorder=2)
    ax.scatter(x=[1, 3], y=[microexon_mean_sibs, non_microexon_mean_sibs], s=130, color='dimgray', label='Siblings', zorder=2)
    labels = ['Microexons', 'Long exons']
    ax.set_xticks([0.5, 2.5])
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([0, 1.41])
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel('abs(effect size)', fontsize=15)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }
    star_labels = get_star_labels(pvals, custom_thresholds)
    pairs = [(0, 1), (0, 2), (2, 3)]
    y_positions = [1.76, 1.87, 1.93]
    draw_lines_and_stars(ax, pairs, y_positions, star_labels)
    plt.tight_layout()
    plt.savefig('figures/microexons_predicted_effects.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    analyze_microexons()

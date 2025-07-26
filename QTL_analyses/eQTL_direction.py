import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def eqtl_direction_analysis():
    output_path = '../resources/GTEx_fine_mapped_eQTLs/'

    tissues = ['Kidney', 'Brain_Cortex', 'Whole_Blood', 'Lung', 'Heart', 'Colon', 
               'Brain_Hippocampus', 'Pancreas', 'Brain_Cerebellum', 'Liver'] # GTEx v10 tissues
    matched_tissues = ['kidney', 'Cerebral_Cortex', 'blood', 'lung', 'heart', 'colon', 
                       'Hippocampus', 'pancreas', 'Cerebellum', 'liver'] # matched tissue models

    # scale to background distribution
    background_path = '../resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()

    master_df = [] # combine into one dataframe
    for tissue, matched_tissue in zip(tissues, matched_tissues):
        if tissue == 'Heart':
            eqtl = pd.read_csv(f'{output_path}Heart_Atrial_Appendage.v10.eQTLs.SuSiE_summary.tsv', sep='\t')
        elif tissue == 'Colon':
            eqtl = pd.read_csv(f'{output_path}Colon_Sigmoid.v10.eQTLs.SuSiE_summary.tsv', sep='\t')
        elif tissue == 'Kidney':
            eqtl = pd.read_csv(f'{output_path}Kidney_Cortex.v10.eQTLs.SuSiE_summary.tsv', sep='\t')
        else:
            eqtl = pd.read_csv(f'{output_path}{tissue}.v10.eQTLs.SuSiE_summary.tsv', sep='\t')
        
        output = pd.read_csv(
            f'{output_path}{tissue}/variant_effects_comprehensive.tsv', 
            sep='\t', 
            index_col=None
            ).drop_duplicates(keep='first')
        interpretability = pd.read_csv(
            f'{output_path}{tissue}/interpretability_analysis.tsv', 
            sep='\t', 
            index_col=None
            ).drop_duplicates(keep='first')
        output['variant_id'] = output['variant_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        output['variant_id'] = 'chr' + output['variant_id']

        eqtl = pd.merge(eqtl, output, on='variant_id', how='inner')
        eqtl = eqtl.dropna(subset=[matched_tissue, 'afc'])
        transcript2gene = dict(zip(interpretability['transcript_id'], interpretability['gene_id'])) 
        eqtl['gene_id'] = eqtl['transcript_id'].map(transcript2gene)
        eqtl = eqtl.dropna(subset=['gene_id'])

        eqtl = eqtl.rename(columns={matched_tissue: 'selected_tissue'})
        eqtl['tissue_name'] = tissue

        # scale to background distribution
        eqtl['selected_tissue'] = (eqtl['selected_tissue'] - mean_global) / std_global

        master_df.append(eqtl[['variant_id', 'gene_id', 'afc', 'selected_tissue', 'tissue_name']])
    
    master_df = pd.concat(master_df, ignore_index=True)

    # compute direction accuracy at a variety of thresholds
    thresholds = np.arange(0, 85, 1) # Otari score thresholds
    accuracies = []
    standard_errors = []
    for thresh in thresholds:
        master_df_copy = master_df.copy()
        
        # sum across isoforms for each variant/gene/tissue combination 
        master_df_copy = master_df_copy.groupby(
            ['variant_id', 'gene_id', 'tissue_name']
            ).agg({'afc': 'first', 'selected_tissue': 'sum'}).reset_index()
        master_df_copy = master_df_copy[
            (master_df_copy['selected_tissue'] >= thresh) | 
            (master_df_copy['selected_tissue'] <= -thresh)]

        # binarize direction
        master_df_copy['afc_sign'] = \
            master_df_copy['afc'].apply(lambda x: 1 if x > 0 else -1)
        master_df_copy['matched_tissue_sign'] = \
            master_df_copy['selected_tissue'].apply(lambda x: 1 if x > 0 else -1)

        # compute accuracy of 'afc_sign' and 'matched_tissue_sign' match
        accuracy = (master_df_copy['afc_sign'] == master_df_copy['matched_tissue_sign']).mean()
        accuracies.append(accuracy)

        # bootstrapping to estimate deviation
        bootstrap_accuracies = []
        n_bootstrap = 1000
        for _ in range(n_bootstrap):
            bootstrap_sample = master_df_copy.sample(frac=1, replace=True)
            bootstrap_accuracy = (bootstrap_sample['afc_sign'] == bootstrap_sample['matched_tissue_sign']).mean()
            bootstrap_accuracies.append(bootstrap_accuracy)
        
        std_dev = np.std(bootstrap_accuracies)
        standard_errors.append(std_dev)

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    accuracies = np.array(accuracies)
    std_devs = np.array(standard_errors)
    ax.plot(
        thresholds, 
        accuracies, 
        linewidth=2, 
        label='GTEx v10', 
        color='steelblue'
        )
    ax.fill_between(
        thresholds, 
        accuracies - std_devs, 
        accuracies + std_devs, 
        alpha=0.2, 
        color='steelblue', 
        label='±1 Std. Dev.'
        )
    ax.set_xlabel('Otari-predicted sum of effects', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('eQTL direction prediction', fontsize=17)
    ax.legend(fontsize=15, frameon=False, loc='lower right')
    ax.tick_params(axis='x', labelsize=13.5)
    ax.tick_params(axis='y', labelsize=13.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    os.makedirs('figures', exist_ok=True)
  
    plt.tight_layout()
    plt.savefig('figures/eQTL_direction_prediction_accuracy_across_tissues.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    eqtl_direction_analysis()

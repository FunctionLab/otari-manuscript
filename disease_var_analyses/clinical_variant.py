import os
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as rick
import matplotlib.lines as mlines


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def get_feature_names():
    """
    Construct a vector of embedding feature names.
    """
    convsplice_cols = ['5ss_splicing', '3ss_splicing']
    seqweaver_cols = pd.read_csv('../resources/model_weights/seqweaver.colnames', header=None)[0].tolist()
    sei_cols = pd.read_csv('../resources/model_weights/histone_features.csv', index_col=0)['Cell Line'].tolist()

    ss_embedding = convsplice_cols + 8*seqweaver_cols + sei_cols
    total_embedding = 2*ss_embedding
    
    return total_embedding


def PTEN_clinical_variant_analysis(transcript_subset=False):
    # load variant and data
    pten_variant = '10_87925512_G_C_hg38'
    pten_variant_name = 'chr10/87925512/G/C'
    file_name = 'PTEN_variant'
    clinvar_df = pd.read_csv(f'../resources/variant_effects/{file_name}/interpretability_analysis.tsv', sep='\t')
    vep = pd.read_csv(f'../resources/variant_effects/{file_name}/variant_effects_comprehensive.tsv', sep='\t')
    with open(f'../resources/variant_effects/{file_name}/variant_to_most_affected_node_embedding.pkl', 'rb') as f:
        variant_to_node_embed = rick.load(f)
    with open(f'../resources/variant_effects/{file_name}/node_to_l2.pkl', 'rb') as f:
        tid_node_to_l2 = rick.load(f)
    
    # scale scores to background distribution
    background_path = '../resources/background_distribution.tsv'
    background = pd.read_csv(background_path, sep='\t')
    mean_global = background[TISSUE_NAMES].values.flatten().mean()
    std_global = background[TISSUE_NAMES].values.flatten().std()
    vep[TISSUE_NAMES] = (vep[TISSUE_NAMES] - mean_global) / std_global

    # OPTIONAL: subset to four transcripts
    if transcript_subset:
        keep_transcripts = ['ENST00000371953', 'ENST00000693560', 'ENST00000688308', 'ENST00000700021'] # first variant
        vep = vep[vep['transcript_id'].isin(keep_transcripts)].reset_index()
        clinvar_df = clinvar_df[clinvar_df['transcript_id'].isin(keep_transcripts)].reset_index()

    # get top features affected
    feature_names = get_feature_names()
    transcripts_to_node_value = defaultdict(list)
    for i, row in clinvar_df.iterrows():
        # convert to list
        top_features = row['top_features'][1:-1]
        top_features = top_features.split(', ')
        top_features = [int(x) for x in top_features]
        tid = row['transcript_id'] 
        for feature in top_features:
            feature_name = feature_names[feature]
            transcripts_to_node_value[feature_name].append(variant_to_node_embed[pten_variant][tid][feature])
    
    # take the mean of the values and sort in descending order
    transcripts_to_node_value = {k: np.mean(v) for k, v in transcripts_to_node_value.items()}
    transcripts_to_node_value = dict(sorted(transcripts_to_node_value.items(), key=lambda item: item[1], reverse=True))
    
    # plot top features affected
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(
        [x.replace('.hg19', '') for x in transcripts_to_node_value.keys()], 
        list(transcripts_to_node_value.values()), 
        color='brown', 
        alpha=0.7, 
        height=0.3
        )
    ax.invert_yaxis()
    ax.set_xlabel('Mean effect')
    ax.tick_params(axis='y', labelsize=8)
    ax.axvline(x=0, color='dimgray', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    
    plt.savefig(f'figures/{file_name}_features_affected.png', dpi=600, bbox_inches='tight')
    plt.close()

    # generate colors for plots
    if transcript_subset:
        colors = ['steelblue', (1.0, 0.4980392156862745, 0.054901960784313725), 'mediumvioletred', 'plum']
        transcript_to_color = dict(zip(keep_transcripts, colors))
    else:
        colors = sns.color_palette('tab10', n_colors=vep.shape[0])
        transcript_ids = list(set(vep['transcript_id']))
        transcript_to_color = dict(zip(transcript_ids, colors))

    # order plot by tissue-specific effects
    vep['total_abs_effect'] = vep[TISSUE_NAMES].abs().sum(axis=1)
    max_effect_transcript = vep.loc[vep['total_abs_effect'].idxmax()]
    max_effect_transcript = max_effect_transcript[TISSUE_NAMES].sort_values(ascending=True)
    tissue_order = list(max_effect_transcript.index)
    
    # plot tissue-specific variant effects for each transcript
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5))
    for i, row in vep.iterrows():
        for j, tissue in enumerate(tissue_order):
            ax.scatter(j, row[tissue], color=transcript_to_color[row['transcript_id']], s=80, alpha=0.95)
    tissue_names_labels = [x.replace('_', ' ').capitalize() for x in tissue_order]
    plt.xticks(np.arange(len(tissue_names_labels)))
    ax.set_xticklabels(tissue_names_labels, rotation=90, fontsize=10)
    ax.set_ylabel('effect size', fontsize=13.5)
    ax.set_title(f'Variant effects for {pten_variant_name}', fontsize=16, pad=10)
    ax.axhline(y=0, color='dimgray', linestyle='--', linewidth=1.5)
    plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    if transcript_subset:
        transcript_ids = keep_transcripts
    else:
        transcript_ids = list(set(vep['transcript_id']))
    colors = [transcript_to_color[x] for x in transcript_ids]
    handles = [
        mlines.Line2D([], [], marker='o', linestyle='-', color=colors[i], markersize=8, label=transcript_ids[i])
        for i in range(len(transcript_ids))
    ]
    ax.legend(handles=handles, title='Transcript ID', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(f'figures/{file_name}_variant_effects.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    # plot L2 distances
    principal =  'ENST00000371953'
    principal_node_to_l2 = list(tid_node_to_l2[principal])
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 1.7))
    x_pos = [0, 2.25, 4.62, 5.05, 5.2, 6.65, 7.08, 7.3, 7.7] # custom x positions for each exon
    ax.scatter(x_pos, principal_node_to_l2, color='steelblue', alpha=0.9, s=60, zorder=2)
    ax.axhline(y=0, color='dimgray', linestyle='--', linewidth=1.5, zorder=1)
    ax.set_ylim(min(principal_node_to_l2) - 5, max(principal_node_to_l2) + 5)
    ax.set_xticks(x_pos)
    labels = [x+1 for x in np.arange(len(principal_node_to_l2))]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.2, 8.2)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel('L2 distance', fontsize=12)
    ax.set_xlabel('Exon number', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(f'figures/{file_name}_node_embedding_scores.png', dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    PTEN_clinical_variant_analysis(transcript_subset=False)
    PTEN_clinical_variant_analysis(transcript_subset=True)

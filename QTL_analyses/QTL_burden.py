import os
import defaultdict
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


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


def get_feature_to_category():
    """
    Map features to regulatory category (sei, seqweaver, convsplice).
    """
    convsplice_cols = ['5ss_splicing', '3ss_splicing']
    seqweaver_cols = pd.read_csv('../resources/model_weights/seqweaver.colnames', header=None)[0].tolist()
    sei_cols = pd.read_csv('../resources/model_weights/histone_features.csv', index_col=0)['Cell Line'].tolist()

    ss_embedding = convsplice_cols + 8*seqweaver_cols + sei_cols
    total_embedding = 2*ss_embedding

    convsplice = ['convsplice' for _ in range(len(convsplice_cols))]
    seqweaver = ['seqweaver' for _ in range(8*len(seqweaver_cols))]
    sei = ['sei' for _ in range(len(sei_cols))]
    ss_embedding = convsplice + seqweaver + sei
    total_embedding = ss_embedding + ss_embedding

    feature_to_category = {i: total_embedding[i] for i in range(len(total_embedding))}
    feature_names = get_feature_names()
    feature_name_to_category = {feature_names[i]: total_embedding[i] for i in range(len(total_embedding))}
    feature_category_counts = {
        'convsplice': len(convsplice_cols), 
        'seqweaver': len(seqweaver_cols), 
        'sei': len(sei_cols)
        }

    return feature_name_to_category, feature_category_counts


def get_category_counts(dataset, feature_names, feature_name_to_category, tissue, top_feature_thresh=750):
    brain_eqtls_feature_to_effect = defaultdict(list)
    
    # get top features for each variant, weigh by variant effects
    for i, row in dataset.iterrows():
        top_features = row['top_local_features'][1:-1]
        top_features = top_features.split(', ')
        top_features = [int(x) for x in top_features]        
        for feature in top_features:
            feature_name = feature_names[feature]
            brain_eqtls_feature_to_effect[feature_name].append(np.abs(row[tissue])) # absolute magnitude
    
    # sum effect for each feature
    brain_eqtls_feature_to_sum_effect = {feature: np.sum(effect) for feature, effect in brain_eqtls_feature_to_effect.items()}
    # normalize by number of variant/transcript combinations in dataset 
    brain_eqtls_feature_to_sum_effect = {feature: effect/len(dataset) for feature, effect in brain_eqtls_feature_to_sum_effect.items()}    
    # get top features in dataset by sum effect
    brain_eqtls_top_features = sorted(
        brain_eqtls_feature_to_sum_effect, 
        key=brain_eqtls_feature_to_sum_effect.get, 
        reverse=True)[:top_feature_thresh]
    # assign to regulatory categories
    top_feature_categories = [feature_name_to_category[feature] for feature in brain_eqtls_top_features]
    category_counts = Counter(top_feature_categories)
    cat, counts = zip(*category_counts.items())
    cat_to_counts = {cat: count for cat, count in zip(cat, counts)}
    cat_to_counts = {cat: cat_to_counts[cat] for cat in ['sei', 'seqweaver']}

    return cat_to_counts


def QTL_burden_heatmap():
    # eQTL paths
    brain_cortex = '../resources/GTEx_fine_mapped_eQTLs/Brain_Cortex'
    liver = '../resources/GTEx_fine_mapped_eQTLs/Liver'
    whole_blood = '../resources/GTEx_fine_mapped_eQTLs/Whole_Blood'
    lung = '../resources/GTEx_fine_mapped_eQTLs/Lung'
    heart = '../resources/GTEx_fine_mapped_eQTLs/Heart'
    colon = '../resources/GTEx_fine_mapped_eQTLs/Colon'

    # sQTL paths
    brain_cortex_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Brain_Cortex'
    liver_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Liver'
    whole_blood_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Whole_Blood'
    lung_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Lung'
    heart_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Heart'
    colon_sqtl = '../resources/GTEx_fine_mapped_sQTLs/Colon'

    # get feature names
    feature_names = get_feature_names()
    feature_name_to_category, norm_counts = get_feature_to_category()

    datasets = []

    def get_heatmap_counts(tissue_data_path, antithesis_path, tissue):
        eqtls_interpretability = pd.read_csv(f'{tissue_data_path}/interpretability_analysis.tsv', sep='\t').drop_duplicates(keep='first')
        eqtls_comprehensive = pd.read_csv(f'{tissue_data_path}/variant_effects_comprehensive.tsv', sep='\t').drop_duplicates(keep='first')

        sqtls_interpretability = pd.read_csv(f'{antithesis_path}/interpretability_analysis.tsv', sep='\t')
        sqtl_variants = set(sqtls_interpretability['variant_id'])
        eqtls_interpretability = eqtls_interpretability[~eqtls_interpretability['variant_id'].isin(sqtl_variants)]
        eqtls_comprehensive = eqtls_comprehensive[~eqtls_comprehensive['variant_id'].isin(sqtl_variants)]

        eqtls_interpretability = pd.merge(eqtls_interpretability, eqtls_comprehensive, on=['variant_id', 'transcript_id'], how='inner')
        feature_to_count = get_category_counts(eqtls_interpretability, feature_names, feature_name_to_category, tissue=tissue)

        return feature_to_count
    
    feature_to_count_list = []
    for tissue_data_path, antithesis_path, tissue in [
        (brain_cortex, brain_cortex_sqtl, 'Cerebral_Cortex'),
        (liver, liver_sqtl, 'liver'),
        (whole_blood, whole_blood_sqtl, 'blood'),
        (lung, lung_sqtl, 'lung'),
        (heart, heart_sqtl, 'heart'),
        (colon, colon_sqtl, 'colon'),
        (brain_cortex_sqtl, brain_cortex, 'Cerebral_Cortex'),
        (liver_sqtl, liver, 'liver'),
        (whole_blood_sqtl, whole_blood, 'blood'),
        (lung_sqtl, lung, 'lung'),
        (heart_sqtl, heart, 'heart'),
        (colon_sqtl, colon, 'colon')
        ]:
        if tissue_data_path and antithesis_path:
            feature_to_count, datasets = get_heatmap_counts(tissue_data_path, antithesis_path, tissue)
            feature_to_count_list.append(feature_to_count)
    
    # Unpack feature counts for each tissue
    brain_feature_to_count = feature_to_count_list[0]
    liver_feature_to_count = feature_to_count_list[1]
    blood_feature_to_count = feature_to_count_list[2]
    lung_feature_to_count = feature_to_count_list[3]
    heart_feature_to_count = feature_to_count_list[4]
    colon_feature_to_count = feature_to_count_list[5]
    brain_sqtl_feature_to_count = feature_to_count_list[6]
    liver_sqtl_feature_to_count = feature_to_count_list[7]
    blood_sqtl_feature_to_count = feature_to_count_list[8]
    lung_sqtl_feature_to_count = feature_to_count_list[9]
    heart_sqtl_feature_to_count = feature_to_count_list[10]
    colon_sqtl_feature_to_count = feature_to_count_list[11]

    # Extract counts for seqweaver and sei
    seqweaver_counts = [brain_feature_to_count.get('seqweaver', 0), liver_feature_to_count.get('seqweaver', 0),
                        blood_feature_to_count.get('seqweaver', 0), lung_feature_to_count.get('seqweaver', 0),
                        heart_feature_to_count.get('seqweaver', 0), colon_feature_to_count.get('seqweaver', 0),
                        brain_sqtl_feature_to_count.get('seqweaver', 0), liver_sqtl_feature_to_count.get('seqweaver', 0),
                        blood_sqtl_feature_to_count.get('seqweaver', 0), lung_sqtl_feature_to_count.get('seqweaver', 0),
                        heart_sqtl_feature_to_count.get('seqweaver', 0), colon_sqtl_feature_to_count.get('seqweaver', 0)]

    sei_counts = [brain_feature_to_count.get('sei', 0), liver_feature_to_count.get('sei', 0),
                  blood_feature_to_count.get('sei', 0), lung_feature_to_count.get('sei', 0),
                  heart_feature_to_count.get('sei', 0), colon_feature_to_count.get('sei', 0),
                  brain_sqtl_feature_to_count.get('sei', 0), liver_sqtl_feature_to_count.get('sei', 0),
                  blood_sqtl_feature_to_count.get('sei', 0), lung_sqtl_feature_to_count.get('sei', 0),
                  heart_sqtl_feature_to_count.get('sei', 0), colon_sqtl_feature_to_count.get('sei', 0)]

    # Calculate min-max scores for seqweaver and sei counts
    seqweaver_min = np.min(seqweaver_counts)
    seqweaver_max = np.max(seqweaver_counts)
    seqweaver_normalized = (np.array(seqweaver_counts) - seqweaver_min) / (seqweaver_max - seqweaver_min)

    sei_min = np.min(sei_counts)
    sei_max = np.max(sei_counts)
    sei_normalized = (np.array(sei_counts) - sei_min) / (sei_max - sei_min)
    
    scores = np.array([sei_normalized, seqweaver_normalized]).T

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(3.2, 4))
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["dimgray", "white", "orange"])

    sns.heatmap(scores, 
                cmap=custom_cmap, 
                yticklabels=datasets, 
                xticklabels=["Chromatin", "RBPs"], 
                cbar_kws={'label': 'Feature importance score'}, 
                ax=ax
                )
    ax.figure.axes[-1].yaxis.label.set_size(14)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)

    os.makedirs('figures', exist_ok=True)
    
    plt.tight_layout()
    fig.savefig('figures/Otari_estimated_QTL_burden_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    QTL_burden_heatmap()

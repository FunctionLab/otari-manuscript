import gzip
from collections import defaultdict

import pandas as pd
import h5py
import numpy as np
import pickle as rick
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
from pytorch_lightning import seed_everything

from utils.utils import binarize, compute_tissue_cutoffs, draw_lines_and_stars, get_star_labels
from models.otari import IsoAbundanceModel
from data import IsoAbundanceDataset


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def convert_edges(edges):
    """
    Convert edge connections to construct an undirected gene graph.
    
    'edges' has columns: 'Node1', 'Node2', each column represents node ID.
    ID is assumed to be 0-based integer.
    """
    edges = edges.astype(int)
    edge_index = torch.tensor(edges.values, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    return edge_index


def process_novel_espresso_isoforms():
    """
    Process novel isoforms into transcript graphs.
    """
    # load novel isoform abundance data
    novel_abundance = '../resources/novel_isoforms_abundance.tsv'
    novel_abundance_df = pd.read_csv(novel_abundance, sep='\t', header=0, index_col=False)
    novel_abundance_df = novel_abundance_df.set_index('transcript_ID')
    novel_abundance_df = novel_abundance_df.apply(pd.to_numeric, errors='coerce')
    
    # read in node embeddings
    node_embedding_path = '../resources/embeddings/espresso_ConvSplice.h5'
    embedding_data = h5py.File(node_embedding_path, 'r')
    transcript_ids = list(embedding_data.keys())

    all_graphs = []
    for transcript_id in transcript_ids:
        # get node embeddings
        node_embeddings = np.array(embedding_data[transcript_id]['reference'])
        # get abundances
        row = novel_abundance_df[novel_abundance_df.index == transcript_id]
        abundances = list(row.loc[transcript_id])
        abundances = [float(abundance) for abundance in abundances]
        abundances = [np.log2(abundance + 0.01) for abundance in abundances] # normalize + pseudocount
                
        edges = [] # list of tuples
        target = abundances

        # get edges
        for j in range(node_embeddings.shape[0]-1):
            seg1 = j
            seg2 = j+1
            edges.append((seg1, seg2))

        # create data object with x, edge_index, and y
        x = torch.tensor(node_embeddings, dtype=torch.float)
        df = pd.DataFrame(edges, columns=['Node1', 'Node2'])
        df = df.drop_duplicates()
        edge_idx = convert_edges(df)

        graph_data = Data(x=x, edge_index=edge_idx, y=torch.tensor(target, dtype=torch.float), transcript_id=transcript_id)
        
        all_graphs.append(graph_data)

    torch.save(all_graphs, '../resources/novel_isoforms_all_transcript_graphs.pt')

    print('Number of graphs:', len(all_graphs))
    print('Example graph:', all_graphs[0])


def evaluate_novel_isoforms(model):
    """
    Evaluate Otari's performance on novel (unannotated) isoforms.
    """    
    dataset = IsoAbundanceDataset(root='../resources', file_name='novel_isoforms_all_transcript_graphs.pt')
    print("ESPRESSO novel isoform data loaded successfully.")

    cutoffs = compute_tissue_cutoffs()

    predicted_abundances = defaultdict(list)
    true_abundances = defaultdict(list)
    all_predicted = defaultdict(list)
    correlations_true = defaultdict(list)
    correlations_pred = defaultdict(list)

    with torch.no_grad():
        for x in dataset:
            pred, y = model(x, mode='eval')
            pred = pred.squeeze(0).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            
            for i in range(len(TISSUE_NAMES)):
                correlations_true[TISSUE_NAMES[i]].append(y[i])
                correlations_pred[TISSUE_NAMES[i]].append(pred[i])
                tissue = TISSUE_NAMES[i]
                all_predicted[tissue].append(pred[i])
                y_tissue = binarize(y[i], tissue, cutoffs)
                if np.isnan(y_tissue):
                    continue

                x_tissue = pred[i]
                
                predicted_abundances[tissue].append(x_tissue)
                true_abundances[tissue].append(y_tissue)
        
    tissue_to_auroc = {}
    tissue_to_auprc = {}
    tissue_to_pearsonr = {}
    
    for i, tissue in enumerate(true_abundances.keys()):
        sorted_predicted = sorted(all_predicted[tissue])
        # convert predicted_abundances[tissue] to percentiles
        predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in predicted_abundances[tissue]]
        tissue_to_auroc[tissue] = roc_auc_score(true_abundances[tissue], predicted_ranked)
        tissue_to_auprc[tissue] = average_precision_score(true_abundances[tissue], predicted_ranked)

        # compute correlation
        tissue_to_pearsonr[tissue] = pearsonr(correlations_true[tissue], correlations_pred[tissue])[0]

    return tissue_to_auroc, tissue_to_auprc, tissue_to_pearsonr


def plot(tissue_to_pearsonr):
    # load chr8 low abundance canonical isoforms data
    with open('../resources/isomodel_low_abundance_predicted_corr_espresso_pclncrna.pkl', 'rb') as f:
        canonical_low_pred = rick.load(f)
    with open('../resources/isomodel_low_abundance_true_corr_espresso_pclncrna.pkl', 'rb') as f:
        canonical_low_true = rick.load(f)
    
    canonical_low_tissue_to_pearson = {}
    for tissue in TISSUE_NAMES:
        canonical_low_tissue_to_pearson[tissue] = pearsonr(canonical_low_true[tissue], canonical_low_pred[tissue])[0]

    # violinplot 
    novel_pearson_values = list(tissue_to_pearsonr.values())
    canonical_low_pearson_values = list(canonical_low_tissue_to_pearson.values())
    t_stat, p_value = ttest_ind(novel_pearson_values, canonical_low_pearson_values)

    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    fig, axs = plt.subplots(1, 1, figsize=(4, 6))
    sns.violinplot(
        data=[list(tissue_to_pearsonr.values()), 
              list(canonical_low_tissue_to_pearson.values())], 
              ax=axs, 
              palette=['plum', 'steelblue'])
    axs.set_xticklabels(['Novel', 'Canonical\n(low abundance)'], fontsize=13)
    axs.set_ylabel('Pearson r', fontsize=17)
    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }
    pvalues = [p_value]
    star_labels = get_star_labels(pvalues, custom_thresholds)
    pairs = [(0, 1)] 
    y_positions = [0.52] 
    draw_lines_and_stars(axs, pairs, y_positions, star_labels)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    plt.suptitle('Evaluation on novel isoforms\n(ESPRESSO)', fontsize=16, weight='bold', y=0.94)
    plt.tight_layout()
    plt.savefig('figures/novel_isoforms_low_abundance_canonical_pearson_violinplot.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    seed_everything(42, workers=True)

    # load model
    model = IsoAbundanceModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with gzip.open('../resources/otari.pth.gz', 'rb') as f:
        state_dict = torch.load(f, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    process_novel_espresso_isoforms()
    _, _, tissue_to_pearsonr = evaluate_novel_isoforms(model)
    plot(tissue_to_pearsonr)

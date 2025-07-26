import os
import gzip
from collections import defaultdict

import torch
import pickle as rick
from pytorch_lightning import seed_everything
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from models.otari import IsoAbundanceModel
from utils.utils import binarize, compute_tissue_cutoffs
from data import IsoAbundanceDataset


ESPRESSO_TISSUES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']

GTEX_TISSUES = ['Brain - Frontal Cortex (BA9)', 'Brain - Cerebellar Hemisphere', 
                'Brain - Putamen (basal ganglia)', 'Lung', 'Heart - Left Ventricle', 
                'Muscle - Skeletal', 'Brain - Anterior cingulate cortex (BA24)', 
                'Heart - Atrial Appendage', 'Brain - Caudate (basal ganglia)', 
                'Adipose - Subcutaneous', 'Liver', 'Breast - Mammary Tissue', 
                'Pancreas']

CTX_TISSUES = ['AdultCTX', 'FetalCTX']


def test_on_GTEx(model, dataset):    
    # compute cutoffs for binarization
    cutoffs = compute_tissue_cutoffs(dataset='gtex')

    # defaultdicts to store results
    predicted_abundances = defaultdict(list)
    true_abundances = defaultdict(list)
    all_predicted = defaultdict(list)
    predicted = defaultdict(list) # for computing corr
    ground_truth = defaultdict(list) # for computing corr


    tissue_pairs = [('Brain - Cerebellar Hemisphere', 'Cerebellum'), 
                    ('Brain - Frontal Cortex (BA9)', 'Frontal_Lobe'),
                    ('Brain - Caudate (basal ganglia)', 'Caudate_Nucleus'),
                    ('Lung', 'lung'), 
                    ('Heart - Atrial Appendage', 'heart'),
                    ('Muscle - Skeletal', 'skeletal_muscle'), 
                    ('Liver', 'liver'),
                    ('Pancreas', 'pancreas')]
    
    gtex_tissue_to_idx = {tissue: i for i, tissue in enumerate(GTEX_TISSUES)}
    espresso_tissue_to_idx = {tissue: i for i, tissue in enumerate(ESPRESSO_TISSUES)}

    tissue_pairs_idx = [(gtex_tissue_to_idx[t1], espresso_tissue_to_idx[t2]) for t1, t2 in tissue_pairs]

    with open('../resources/protein_coding_lncrna_transcripts_gtex.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)

    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            
            pred = model(x)
            pred = pred.squeeze(0)
            pred = pred.cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()
            
            for (i, j) in tissue_pairs_idx:
                tissue = GTEX_TISSUES[i] # gtex tissue

                predicted[tissue].append(pred[j]) # predicted espresso
                ground_truth[tissue].append(y[i]) # true gtex

                all_predicted[tissue].append(pred[j]) # predicted espresso
                y_tissue = binarize(y[i], tissue, cutoffs) # true gtex
                if np.isnan(y_tissue):
                    continue

                x_tissue = pred[j] # predicted espresso
                predicted_abundances[tissue].append(x_tissue)
                true_abundances[tissue].append(y_tissue)
    
    tissue_to_auroc = {}
    tissue_to_auprc = {}
    tissue_to_pearsonr = {}
    for i, tissue in enumerate(true_abundances.keys()):
        sorted_predicted = sorted(all_predicted[tissue])
        predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in predicted_abundances[tissue]]
        
        tissue_to_auroc[tissue] = roc_auc_score(true_abundances[tissue], predicted_ranked)
        tissue_to_auprc[tissue] = average_precision_score(true_abundances[tissue], predicted_ranked)
        tissue_to_pearsonr[tissue] = pearsonr(predicted[tissue], ground_truth[tissue])[0]
    
    return tissue_to_auroc, tissue_to_auprc, tissue_to_pearsonr


def test_on_CTX(model, dataset):
    # compute cutoffs for binarization
    cutoffs = compute_tissue_cutoffs(dataset='ctx')
    cutoffs = {k.replace('FL.', ''): v for k, v in cutoffs.items()} # remove 'FL.' from keys

    # defaultdicts to store results
    predicted_abundances = defaultdict(list)
    true_abundances = defaultdict(list)
    all_predicted = defaultdict(list)
    predicted = defaultdict(list) # for computing corr
    ground_truth = defaultdict(list) # for computing corr

    tissue_pairs = [('AdultCTX', 'Brain'),
                    ('FetalCTX', 'Fetal_Brain')]
    
    ctx_tissue_to_idx = {tissue: i for i, tissue in enumerate(CTX_TISSUES)}
    espresso_tissue_to_idx = {tissue: i for i, tissue in enumerate(ESPRESSO_TISSUES)}

    tissue_pairs_idx = [(ctx_tissue_to_idx[t1], espresso_tissue_to_idx[t2]) for t1, t2 in tissue_pairs]

    with open('../resources/protein_coding_lncrna_transcripts_ctx.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)

    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            
            pred = model(x)
            pred = pred.squeeze(0)
            pred = pred.cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()
            
            for (i, j) in tissue_pairs_idx:
                tissue = CTX_TISSUES[i] # gtex tissue

                predicted[tissue].append(pred[j]) # predicted espresso
                ground_truth[tissue].append(y[i]) # true gtex

                all_predicted[tissue].append(pred[j]) # predicted espresso
                y_tissue = binarize(y[i], tissue, cutoffs) # true gtex
                if np.isnan(y_tissue):
                    continue

                x_tissue = pred[j] # predicted espresso
                predicted_abundances[tissue].append(x_tissue)
                true_abundances[tissue].append(y_tissue)
    
    tissue_to_auroc = {}
    tissue_to_auprc = {}
    tissue_to_pearsonr = {}
    for i, tissue in enumerate(true_abundances.keys()):
        sorted_predicted = sorted(all_predicted[tissue])
        predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in predicted_abundances[tissue]]
        
        tissue_to_auroc[tissue] = roc_auc_score(true_abundances[tissue], predicted_ranked)
        tissue_to_auprc[tissue] = average_precision_score(true_abundances[tissue], predicted_ranked)
        tissue_to_pearsonr[tissue] = pearsonr(predicted[tissue], ground_truth[tissue])[0]
    
    return tissue_to_auroc, tissue_to_auprc, tissue_to_pearsonr


def plot_metrics(metrics, output):
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 4.5))
    sns.scatterplot(
        x=list(metrics.values()), 
        y=list(metrics.keys()), 
        marker='o', 
        color='steelblue', 
        edgecolor='steelblue', 
        linewidth=1.5, 
        s=200, 
        alpha=0.8, 
        ax=axs
        )
    tissue_labels = [x.replace('_', ' ').capitalize() for x in metrics.keys()]
    axs.set_yticklabels(tissue_labels, fontsize=13)
    axs.set_xlabel(f'{output}', fontsize=16)
    axs.tick_params(axis='x', labelsize=13.5)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.set_xlim([0, 1])
    axs.set_title(f'Validation\n({output})', fontsize=17)
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/Otari_validation_GTEX_{output}.png', dpi=600, bbox_inches='tight')
    plt.show()
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

    # load data
    gtex_dataset = IsoAbundanceDataset(root='../resources', file_name='all_transcript_graphs_directed_GTEx.pt')
    gtex_dataset = [d for d in gtex_dataset if d.chrom in [8]] # subset to holdout chrom for testing

    ctx_dataset = IsoAbundanceDataset(root='../resources', file_name='all_transcript_graphs_directed_CTX.pt')
    ctx_dataset = [d for d in ctx_dataset if d.chrom in [8]] # subset to holdout chrom for testing

    # predict on GTEx validation test set
    auroc, auprc, pearsonr = test_on_GTEx(model, gtex_dataset)
    plot_metrics(auroc, 'auroc_gtex')
    plot_metrics(auprc, 'auprc_gtex')
    plot_metrics(pearsonr, 'pearsonr_gtex')

    # predict on CTX validation test set
    auroc_ctx, auprc_ctx, pearsonr_ctx = test_on_CTX(model, ctx_dataset)
    plot_metrics(auroc_ctx, 'auroc_ctx')
    plot_metrics(auprc_ctx, 'auprc_ctx')
    plot_metrics(pearsonr_ctx, 'pearsonr_ctx')

from collections import defaultdict
import gzip
import os

import torch
import pickle as rick
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score

from models.otari import IsoAbundanceModel
from utils.utils import load_config, binarize, compute_tissue_cutoffs
from data import IsoAbundanceDataset


TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def test_on_ESPRESSO(model, dataset):    
    # compute cutoffs for binarization
    cutoffs = compute_tissue_cutoffs()

    # defaultdicts to store results
    predicted_abundances = defaultdict(list)
    true_abundances = defaultdict(list)
    all_predicted = defaultdict(list)
    predicted = defaultdict(list) # for computing corr
    ground_truth = defaultdict(list) # for computing corr

    with open('resources/protein_coding_lncrna_transcripts_espresso.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)

    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            
            pred = model(x)
            pred = pred.squeeze(0)
            pred = pred.cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()
            
            for i in range(len(TISSUE_NAMES)):
                tissue = TISSUE_NAMES[i]

                predicted[tissue].append(pred[i])
                ground_truth[tissue].append(y[i])

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
        predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in predicted_abundances[tissue]]
        
        tissue_to_auroc[tissue] = roc_auc_score(true_abundances[tissue], predicted_ranked)
        tissue_to_auprc[tissue] = average_precision_score(true_abundances[tissue], predicted_ranked)
        tissue_to_pearsonr[tissue] = pearsonr(predicted[tissue], ground_truth[tissue])[0]
    
    return tissue_to_auroc, tissue_to_auprc, tissue_to_pearsonr


def plot_ESPRESSO(metrics, output):
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 7.5))
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
    axs.set_xlabel('AUROC', fontsize=16)
    axs.tick_params(axis='x', labelsize=13.5)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_linewidth(2)
    axs.set_xlim([0, 1])
    axs.set_title('Evaluation on holdout set\n(Gao et al)', fontsize=17)

    plt.tight_layout()
    plt.savefig(f'figures/Otari_evaluation_chrom8_{output}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def transcript_complexity_analysis(model, dataset):
    cutoffs = compute_tissue_cutoffs()

    all_predicted = defaultdict(list) # tissue to all predicted values
    num_exons_to_predicted = defaultdict(lambda: defaultdict(list))
    num_exons_to_true = defaultdict(lambda: defaultdict(list))

    with open('resources/protein_coding_lncrna_transcripts_espresso.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)

    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            pred = model(x)
            pred = pred.squeeze(0).cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()
            
            for i in range(len(TISSUE_NAMES)):
                tissue = TISSUE_NAMES[i]
                all_predicted[tissue].append(pred[i])
                y_tissue = binarize(y[i], tissue, cutoffs)
                if np.isnan(y_tissue):
                    continue

                x_tissue = pred[i]
                num_exons = x.x.shape[0]
                if num_exons > 10: # threshold
                    num_exons = 10
                num_exons_to_predicted[num_exons][tissue].append(x_tissue)
                num_exons_to_true[num_exons][tissue].append(y_tissue)
    
    exon_num_to_tissueroc = defaultdict(list)
    for i, tissue in enumerate(all_predicted.keys()):
        sorted_predicted = sorted(all_predicted[tissue])
        for num_exons in num_exons_to_predicted.keys():
            predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in num_exons_to_predicted[num_exons][tissue]]
            exon_num_to_tissueroc[num_exons].append(roc_auc_score(num_exons_to_true[num_exons][tissue], predicted_ranked))

    # plot
    fig, ax = plt.subplots(figsize=(4.9, 6))
    exon_num_to_tissueroc = dict(sorted(exon_num_to_tissueroc.items())) # sort by exon number
    sns.boxplot(
        data=list(exon_num_to_tissueroc.values()), 
        ax=ax, 
        showfliers=False, 
        whiskerprops=dict(color="black", linewidth=2), 
        capprops=dict(color="black", linewidth=2), 
        medianprops=dict(color='white', linewidth=2), 
        boxprops=dict(edgecolor='white', linewidth=0.5), 
        color='dimgray', 
        orient='h'
        )
    sns.swarmplot(
        data=list(exon_num_to_tissueroc.values()), 
        ax=ax, 
        color='steelblue', 
        size=4, 
        alpha=0.75, 
        edgecolor='white', 
        linewidth=0.5, 
        orient='h'
        )
    labels = [f'{i}' for i in range(1, len(exon_num_to_tissueroc.keys())+1)]
    labels[-1] = str(labels[-1]) + '+'
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.set_ylabel('Exon count', fontsize=18.5)
    ax.set_xlabel('AUROC', fontsize=17.5)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig('figures/Otari_transcript_complexity_vs_AUROC.png', dpi=600, bbox_inches='tight')
    plt.close()


def compute_per_gene_correlation(model, dataset):
    with open('resources/transcript2gene.pkl', 'rb') as f:
        transcript2gene = rick.load(f)

    with open('resources/protein_coding_lncrna_transcripts_espresso.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)

    gene_to_iso_exp = defaultdict(lambda: defaultdict(list))
    pred_gene_to_iso_exp = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            pred = model(x)
            pred = pred.squeeze(0).cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()

            if x.transcript_id not in transcript2gene:
                continue
            gene = transcript2gene[x.transcript_id]
            
            for i, tissue in enumerate(TISSUE_NAMES):
                gene_to_iso_exp[tissue][gene].append(y)
                pred_gene_to_iso_exp[tissue][gene].append(pred[i])
            
    tissue_to_pearsonr = defaultdict(list) # list of pearsonr per gene
    for t in TISSUE_NAMES:
        for gene in gene_to_iso_exp.keys():        
            if len(gene_to_iso_exp[t][gene]) < 2: # must have at least 2 values
                continue

            y = gene_to_iso_exp[t][gene]
            pred = pred_gene_to_iso_exp[t][gene]
            tissue_to_pearsonr[t].append(pearsonr(pred, y)[0])

    means = []
    ste = []
    for tissue, values in tissue_to_pearsonr.items():
        means.append(np.mean(values))
        ste.append(np.std(values) / np.sqrt(len(values)))

    # plot per gene correlation distributions
    fig, ax = plt.subplots(figsize=(4.5, 7))
    sns.scatterplot(x = means, 
                    y = TISSUE_NAMES, 
                    ax=ax, 
                    color='dimgray', 
                    s=100, 
                    edgecolor='dimgray', 
                    linewidth=1.5
                    )
    ax.errorbar(means, 
                TISSUE_NAMES, 
                xerr=ste, 
                fmt='none', 
                color='dimgray', 
                linewidth=2, 
                capsize=0
                )
    ax.set_xlabel('Pearson r', fontsize=16)
    tissue_labels = [x.replace('_', ' ').capitalize() for x in TISSUE_NAMES]
    ax.set_yticklabels(tissue_labels, fontsize=12)
    ax.set_title('Pearson r per gene', fontsize=17)
    ax.tick_params(axis='x', labelsize=12.5)
    ax.set_xlim([0,0.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig('figures/Otari_pearsonr_per_gene.png', dpi=600, bbox_inches='tight')
    plt.close()


def isoform_vs_global_AUROC(model, dataset):
    cutoffs = compute_tissue_cutoffs()

    with open('resources/protein_coding_lncrna_transcripts_espresso.pkl', 'rb') as f:
        protein_coding_lncrna = rick.load(f)
    
    with open('resources/transcript2gene.pkl', 'rb') as f:
        transcript2gene = rick.load(f)

    true_observed = defaultdict(list)
    predicted = defaultdict(list)
    all_predicted = defaultdict(list)
    # keep track of observed abundances for isoforms of each gene
    tissue_to_gene_to_abundance = defaultdict(lambda: defaultdict(list))
    tissue_to_gene_to_predicted = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for x in dataset:
            if x.transcript_id not in protein_coding_lncrna:
                continue
            pred = model(x)
            pred = pred.squeeze(0).cpu().detach().numpy()
            y = x.y.cpu().detach().numpy()

            if x.transcript_id not in transcript2gene:
                continue
            gene = transcript2gene[x.transcript_id]

            for i, tissue in enumerate(TISSUE_NAMES):
                all_predicted[tissue].append(pred[i])
                y_tissue = binarize(y[i], tissue, cutoffs)
                if np.isnan(y_tissue):
                    continue # ignore transcripts not high or low expressed in tissue

                x_tissue = pred[i]
                true_observed[tissue].append(y_tissue)
                predicted[tissue].append(x_tissue)
                tissue_to_gene_to_abundance[tissue][gene].append(y_tissue)
                tissue_to_gene_to_predicted[tissue][gene].append(x_tissue)
    
    # compute max abundance per gene in each tissue
    tissue_to_gene_to_max_abundance = defaultdict(lambda: defaultdict(float))
    for t in TISSUE_NAMES:
        for gene in tissue_to_gene_to_abundance[t].keys():
            max_abundance = max(tissue_to_gene_to_abundance[t][gene])
            tissue_to_gene_to_max_abundance[t][gene] = max_abundance

    tissue_to_auroc_isoform = {}
    tissue_to_auroc_gene = {}
    for i, tissue in enumerate(TISSUE_NAMES):
        # compute isoform-level AUROC
        sorted_predicted = sorted(all_predicted[tissue])
        predicted_ranked = [sorted_predicted.index(x) / len(sorted_predicted) for x in predicted[tissue]]
        tissue_to_auroc_isoform[tissue] = roc_auc_score(true_observed[tissue], predicted_ranked)

        # compute gene-level AUROC
        gene_predicted = []
        gene_true = []
        for gene in tissue_to_gene_to_predicted[tissue].keys():
            gene_predicted.extend(tissue_to_gene_to_predicted[tissue][gene])
            gene_true.extend([tissue_to_gene_to_max_abundance[tissue][gene]] * len(tissue_to_gene_to_predicted[tissue][gene]))
        sorted_gene_predicted = sorted(gene_predicted)
        gene_predicted_ranked = [sorted_gene_predicted.index(x) / len(sorted_gene_predicted) for x in gene_predicted]
        tissue_to_auroc_gene[tissue] = roc_auc_score(gene_true, gene_predicted_ranked)
        
    # plot
    max_pred = []
    pred = []
    for tissue in TISSUE_NAMES:
        pred.append(tissue_to_auroc_isoform[tissue])
        max_pred.append(tissue_to_auroc_gene[tissue])
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.scatterplot(x=max_pred, y=pred, ax=ax, s=230, color='steelblue', alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('AUROC, gene-level', fontsize=17.5)
    ax.set_ylabel('AUROC, isoform-specific', fontsize=17.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlim([0.5,0.9])
    plt.ylim([0.5,0.9])
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.tight_layout()
    plt.savefig('figures/Otari_isoform_vs_gene_AUROC.png', dpi=600, bbox_inches='tight')
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
    dataset = IsoAbundanceDataset(root='../resources', file_name='all_transcript_graphs_directed_espresso.pt')
    dataset = [d for d in dataset if d.chrom in [8]] # subset to holdout chrom for testing

    # predict on ESPRESSO holdout set
    auroc, auprc, pearsonr = test_on_ESPRESSO(model, dataset)
    plot_ESPRESSO(auroc, 'auroc')
    plot_ESPRESSO(auprc, 'auprc')
    plot_ESPRESSO(pearsonr, 'pearsonr')

    # other analyses: transcript complexity, per-gene, global AUROC
    transcript_complexity_analysis(model, dataset)
    compute_per_gene_correlation(model, dataset)
    isoform_vs_global_AUROC(model, dataset)

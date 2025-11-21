import os
import re
import time
from typing import List, Dict, Tuple

import pickle as rick
import numpy as np
import pandas as pd
from sklearn.metrics import spearmanr
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

from alphagenome.models import dna_client
from alphagenome.data import genome

from utils.utils import draw_lines_and_stars, get_star_labels


API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "PASTE_YOUR_KEY_HERE")
OUTPUT_CSV = "data/alphagenome_predictions.csv"

# Context window for AG
PREFERRED_SEQ_LEN = 1048576

# bp window on each side of TSS for CAGE
CAGE_HALF_WINDOW = 500 

TISSUE_NAMES = ['Brain', 'Caudate_Nucleus', 'Cerebellum', 'Cerebral_Cortex', 
                'Corpus_Callosum', 'Fetal_Brain', 'Fetal_Spinal_Cord', 'Frontal_Lobe', 
                'Hippocampus', 'Medulla_Oblongata', 'Pons', 'Spinal_Cord', 
                'Temporal_Lobe', 'Thalamus', 'bladder', 'blood', 'colon', 'heart', 
                'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skeletal_muscle', 
                'small_intestine', 'spleen', 'stomach', 'testis', 'thyroid']


def parse_gtf_attributes(attr: str) -> Dict[str, str]:
    """
    Simple GTF attr parser: key "value"; pairs.
    Args:
        attr: string of GTF attributes
    Returns:
        dict of attributes
    """
    out = {}
    for m in re.finditer(r'(\S+)\s+"([^"]+)"', attr or ""):
        out[m.group(1)] = m.group(2)
    return out


def load_transcripts_and_exons() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      tx_df: transcript-level table with columns:
        transcript_id, gene_id, gene_name, chrom, strand, tss, transcript_length
      exons_df: exon rows with columns:
        transcript_id, chrom, start, end (1-based inclusive GTF coords)
    """
    cols = ["chrom","source","feature","start","end","score","strand","frame","attribute"]
    df = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v47.basic.annotation.gtf', sep="\t", comment="#", header=None, names=cols, dtype={"chrom":str})
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)

    # Extract attributes
    attrs = df["attribute"].apply(parse_gtf_attributes)
    df["gene_id"]        = attrs.apply(lambda d: d.get("gene_id"))
    df["transcript_id"]  = attrs.apply(lambda d: d.get("transcript_id"))
    df["gene_name"]      = attrs.apply(lambda d: d.get("gene_name"))

    # Subset to chr8 holdout set
    df = df[~df["transcript_id"].isna()].copy()
    df = df[df["chrom"].isin(["8", "chr8"])].copy()

    # Build exon table
    exons = df[df["feature"] == "exon"][["transcript_id","chrom","start","end"]].copy()

    # Build transcript table
    tx_like = df[df["feature"].isin(["transcript","mRNA"])][
        ["transcript_id","gene_id","gene_name","chrom","strand","start","end"]
    ].copy().drop_duplicates()

    # Compute TSS, transcript length per transcript
    def tss_row(row):
        return row["start"] if row["strand"] == "+" else row["end"]
    tx_like["tss"] = tx_like.apply(tss_row, axis=1)
    tx_like["transcript_length"] = abs(tx_like["end"] - tx_like["start"])

    tx_df = tx_like[["transcript_id","gene_id","gene_name","chrom","strand","tss","transcript_length"]].copy()
    return tx_df, exons


def create_model(api_key: str):
    """
    Create a DNA client model.
    Args:
        api_key: string of API key
    Returns:
        DNA client model
    """
    return dna_client.create(api_key)


def interval_around_tss(chrom: str, tss_1based: int, seq_len: int) -> genome.Interval:
    """
    Args:
        chrom: string of chromosome
        tss_1based: int of 1-based TSS
        seq_len: int of sequence length
    Returns:
        genome.Interval of interval around TSS
    """
    half_window = seq_len // 2
    start0 = max(0, tss_1based - 1 - half_window)
    end0 = start0 + seq_len 
    return genome.Interval(chromosome=str(chrom), start=start0, end=end0)


def fetch_output_metadata(model):
    """
    Args:
        model: DNA client model
    Returns:
        concatenated metadata DF for all output types for human
    """
    md = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).concatenate()
    return md


def aggregate_rnaseq_over_exons(rna_values: np.ndarray, interval: genome.Interval, exon_rows: pd.DataFrame) -> np.ndarray:
    """
    Create a boolean mask indicating positions within exons that intersect 'interval'.
    
    Args:
        rna_values: array shape (seq_len, n_tracks) - RNA-seq values
        interval: genome.Interval - the genomic interval corresponding to rna_values
        exon_rows: DataFrame with columns ['start', 'end', 'chrom'] - exons to mask (1-based inclusive GTF coords)
    
    Returns:
        Boolean mask of shape (seq_len,) indicating positions within exons
    """
    seq_len = rna_values.shape[0]
    mask = np.zeros(seq_len, dtype=bool)

    for _, ex in exon_rows.iterrows():
        ex_start0 = ex["start"] - 1
        ex_end0   = ex["end"]
        # Intersect with interval and convert to relative positions within the sequence
        s = max(ex_start0 - interval.start, 0)
        e = min(ex_end0 - interval.start, seq_len)
        if e > s:
            mask[s:e] = True

    return mask


def predict_AG_for_transcript(model,
                            tx_row: pd.Series,
                            exons_df: pd.DataFrame,
                            ontology_curies: List[str],
                            seq_len: int):
    """
    Args:
        model: DNA client model
        tx_row: pandas Series with transcript information
        exons_df: pandas DataFrame with exon information
        ontology_curies: list of ontology curies
        seq_len: int of sequence length
    Returns:
        list of dict rows for this transcript across tissues/tracks
        each dict contains:
            - transcript_id: string of transcript ID
            - tissue_name: string of tissue name
            - rna_exon_sum_norm: float of normalized RNA-seq expression
            - cage_tss_max: float of CAGE expression
    """
    chrom = tx_row["chrom"]
    tss = int(tx_row["tss"])
    sequence_length = int(tx_row["transcript_length"])
    tx_id = tx_row["transcript_id"]

    # Compute interval around TSS for CAGE
    interval_tss = interval_around_tss(chrom, tss, seq_len)

    # Compute interval for RNA-seq centered on transcript mid-point
    transcript_mid_point_1based = tss + sequence_length // 2
    transcript_mid_point_0based = transcript_mid_point_1based - 1
    half_window = seq_len // 2
    start0 = max(0, transcript_mid_point_0based - half_window)
    end0 = start0 + seq_len
    interval_rna = genome.Interval(
        chromosome=str(chrom), 
        start=start0, 
        end=end0
    )
    
    rna_outputs = model.predict_interval(
        interval=interval_rna,
        requested_outputs=[dna_client.OutputType.RNA_SEQ],
        ontology_terms=ontology_curies,
    )

    cage_outputs = model.predict_interval(
        interval=interval_tss,
        requested_outputs=[dna_client.OutputType.CAGE],
        ontology_terms=ontology_curies,
    )

    rna = rna_outputs.rna_seq 
    cage = cage_outputs.cage 

    # Convert TSS from genome coordinates to relative position within interval_tss
    tss_relative = tss - 1 - interval_tss.start 
    cage_s = max(0, tss_relative - CAGE_HALF_WINDOW)
    cage_e = min(cage.values.shape[0], tss_relative + CAGE_HALF_WINDOW + 1)
    cage_tss_window = cage.values[cage_s:cage_e, :]

    # Define exon mask - filter exons for this transcript
    tx_exons = exons_df[exons_df["transcript_id"] == tx_id]
    exon_mask = aggregate_rnaseq_over_exons(rna.values, interval_rna, tx_exons)

    out_rows = []
    for curie in ontology_curies:
        # Find all RNA tracks matching current CURIE
        rna_matches = rna.metadata[rna.metadata["ontology_curie"] == curie]
        if len(rna_matches) == 0:
            continue
        
        # Get all column indices for tracks with this CURIE
        rna_indices = rna_matches.index.tolist()
        tissue_name = rna_matches.iloc[0].get("biosample_name")

        # Extract values for all matching tracks: shape (exon_positions, n_matching_tracks)
        rna_exon_values = rna.values[exon_mask, :][:, rna_indices]
        
        # Compute sum, normalize by sequence_length, then average across tracks
        track_sums = rna_exon_values.sum(axis=0)
        track_sums_norm = track_sums / sequence_length 
        rna_sum_norm = float(track_sums_norm.mean())

        # Get CAGE value for matching CURIE
        cage_val = float("nan")
        cage_matches = cage.metadata[cage.metadata["ontology_curie"] == curie]
        if len(cage_matches) > 0:
            cage_indices = [idx for idx in cage_matches.index if idx < cage_tss_window.shape[1]]
            if len(cage_indices) > 0:
                # Extract max values for each track, then take overall mean
                cage_max_values = [float(cage_tss_window[:, idx].max()) for idx in cage_indices]
                cage_val = float(np.mean(cage_max_values))

        out_rows.append({
            "transcript_id": tx_id,
            "tissue_name": tissue_name,
            "rna_exon_sum_norm": rna_sum_norm,
            "cage_tss_max": cage_val,
        })

    return out_rows


def predict_alphagenome_for_chr8():
    """
    Predict with AlphaGenome RNA-seq and CAGE tracks for the chr8 holdout set.
    """
    if API_KEY is None or API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit("Please set ALPHAGENOME_API_KEY or paste your API key.")

    print("Loading GTF…")
    tx_df, exons_df = load_transcripts_and_exons()
    print(f"Transcripts: {len(tx_df)}; Exons: {len(exons_df)}")

    print("Creating AlphaGenome client…")
    model = create_model(API_KEY)

    print("Fetching output metadata…")
    md = fetch_output_metadata(model)

    print("Selecting tracks corresponding to specified tissues...")

    # Subset to tissues
    md_tissues = md[md['biosample_type'] == 'tissue'].copy()
    track_names = ['brain', 'head of caudate nucleus', 'cerebellum', 'frontal cortex', 'corpus callosum',
        'layer of hippocampus', 'medulla oblongata',
        'pons', 'spinal cord', 'temporal lobe', 'hypothalamus', 'urinary bladder', 'venous blood', 'colon',
        'heart', 'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate gland', 'skeletal muscle tissue',
        'small intestine', 'spleen', 'stomach', 'thyroid gland']
    md_tissues = md_tissues[md_tissues['biosample_name'].isin(track_names)]
    
    # Get ontology curies
    ontology_curies = md_tissues["ontology_curie"].dropna().unique().tolist()
    
    # Process all transcripts
    print(f"Processing {len(tx_df)} transcripts…")
    all_rows = []
    for i, (_, tx) in enumerate(tx_df.iterrows(), start=1):
        print(f"[{i}/{len(tx_df)}] {tx['transcript_id']} ({tx.get('gene_name','')})")
        rows = predict_AG_for_transcript(
            model=model,
            tx_row=tx,
            exons_df=exons_df,
            ontology_curies=ontology_curies,
            seq_len=PREFERRED_SEQ_LEN,
        )
        all_rows.extend(rows)
        time.sleep(0.2)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Wrote:", OUTPUT_CSV)


def normalize_tissue_name(name: str) -> str:
    replacements = [
        ('spinal cord', 'Spinal_Cord'),
        ('urinary bladder', 'bladder'),
        ('venous blood', 'blood'),
        ('prostate gland', 'prostate'),
        ('small intestine', 'small_intestine'),
        ('thyroid gland', 'thyroid'),
        ('brain', 'Brain'),
        ('skeletal muscle tissue', 'skeletal_muscle'),
        ('frontal cortex', 'Frontal_Lobe'),
        ('temporal lobe', 'Temporal_Lobe'),
        ('cerebellum', 'Cerebellum'),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def get_alphagenome_predictions(track: str):
    """
    Args:
        track: string of track name
    Returns:
        list of Spearman r values for each tissue for the given track
    """
    # Load Otari predictions and Espresso ground truth abundances
    otari_predictions = pd.read_csv('data/espresso_chr8_lncrna_predictions.csv')
    espresso_abundances = pd.read_csv('data/espresso_chr8_lncrna_true_abundances.csv')
    transcript_ids = otari_predictions['transcript_id'].unique().tolist()

    # Define available tissue names for AlphaGenome
    tissue_names_ag = ['stomach', 'ovary', 'pancreas', 'thyroid gland', 'lung', 
                    'spleen', 'liver', 'kidney', 'heart', 'brain', 'skeletal muscle tissue', 
                    'urinary bladder', 'frontal cortex', 'temporal lobe', 'cerebellum', 
                    'small intestine', 'spinal cord', 'prostate gland', 'venous blood']
    
    # Load AlphaGenome predictions
    out_df = pd.read_csv(OUTPUT_CSV)
    out_df['transcript_id'] = out_df['transcript_id'].apply(lambda x: x.split('.')[0])

    # Normalize tissue names
    out_df = out_df[out_df['tissue_name'].isin(tissue_names_ag)]
    out_df['tissue_name'] = out_df['tissue_name'].apply(normalize_tissue_name)

    # Subset to transcripts in otari_predictions
    out_df['transcript_id'] = out_df['transcript_id'].apply(lambda x: x.split('.')[0])
    out_df = out_df[out_df['transcript_id'].isin(transcript_ids)]

    # Define tissue order
    tissue_order_alphagenome = out_df['tissue_name'].unique().tolist()
    
    alphagenome_proc = []
    espresso_proc = []
    for _, row in espresso_abundances.iterrows():
        transcript_id = row['transcript_id']
        transcript_data = out_df[out_df['transcript_id'] == transcript_id]
        
        # Order tissues by tissue_order_alphagenome
        order_map = {tissue: idx for idx, tissue in enumerate(tissue_order_alphagenome)}
        transcript_data = transcript_data[transcript_data['tissue_name'].isin(order_map)]
        transcript_data = transcript_data.copy()
        transcript_data['_order'] = transcript_data['tissue_name'].map(order_map)
        transcript_data = transcript_data.sort_values('_order').drop(columns='_order')
        
        # Fetch AlphaGenome values
        alphagenome_values = transcript_data[track].tolist()
        alphagenome_values = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in alphagenome_values]
        alphagenome_proc.append(alphagenome_values)

        # Fetch espresso gt values
        espresso_values = [row[tissue] for tissue in tissue_order_alphagenome]
        espresso_proc.append(espresso_values)
    
    # Get a correlation for each tissue
    ag_spearmanr = []
    for t in range(len(tissue_order_alphagenome)):
        ag_tissue = [x[t] for x in alphagenome_proc]
        espresso_tissue = [x[t] for x in espresso_proc]
        ag_spearmanr.append(spearmanr(ag_tissue, espresso_tissue).statistic)
    
    return ag_spearmanr


def plot_AlphaGenome_benchmark():
    ag_spearmanr_cage = get_alphagenome_predictions(track='cage_tss_max')
    ag_spearmanr_rna = get_alphagenome_predictions(track='rna_exon_sum_norm')

    # Load Otari predictions and Espresso ground truth
    with open('data/otari_predicted_espresso.pkl', 'rb') as f:
        otari_pred = rick.load(f)
    with open('data/ground_truth_espresso.pkl', 'rb') as f:
        otari_true = rick.load(f)

    # Shared tissues between Otari and AlphaGenome
    shared_tissues = ['stomach', 'ovary', 'pancreas', 'thyroid', 'lung', 
                    'spleen', 'liver', 'kidney', 'heart', 'Brain', 
                    'skeletal_muscle', 'bladder', 'Frontal_Lobe', 
                    'Temporal_Lobe', 'Cerebellum', 'small_intestine', 
                    'Spinal_Cord', 'prostate', 'blood']
    
    otari_spearmanr = []
    for _, t in enumerate(TISSUE_NAMES):
        if t not in shared_tissues:
            continue
        otari_spearmanr.append(spearmanr(otari_pred[t], otari_true[t]).statistic)
    
    # Compute p-values
    p_value_rna = mannwhitneyu(ag_spearmanr_rna, otari_spearmanr, alternative='two-sided').pvalue
    p_value_cage = mannwhitneyu(ag_spearmanr_cage, otari_spearmanr, alternative='two-sided').pvalue
    pvals = [p_value_rna, p_value_cage]
    _, corrected_pvals, _, _ = multipletests(pvals, method='fdr_bh')

    # Plot results
    _, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    sns.boxplot(data=[ag_spearmanr_cage, ag_spearmanr_rna, otari_spearmanr], palette=['darkturquoise', 'darkorange', 'crimson'])
    ax.set_xticklabels(['AlphaGenome\n(CAGE)', 'AlphaGenome\n(RNA-seq)', 'Otari'], fontsize=14)
    ax.set_title('Chrom 8 benchmark', fontsize=14.5)
    ax.set_ylim(0.35, 0.68)
    ax.set_ylabel('Spearman r', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    custom_thresholds = {
        0.001: '***',
        0.01: '**',
        0.05: '*',
        1: 'ns'
    }
    star_labels = get_star_labels(corrected_pvals, custom_thresholds)
    pairs = [(0, 2), (1, 2)] 
    y_positions = [0.64, 0.66]
    draw_lines_and_stars(ax, pairs, y_positions, star_labels)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/AlphaGenome_chr8_benchmark.png', dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    predict_alphagenome_for_chr8()
    plot_AlphaGenome_benchmark()

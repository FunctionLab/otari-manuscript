# sQTL Benchmark for Splice Variant Predictors

Benchmark comparing splice variant predictors on GTEx fine-mapped sQTL variants.

## Predictors Evaluated
- **Otari** - Tissue-specific splice effect predictor
- **SpliceAI** - Deep learning splice site predictor
- **MTSplice** - Tissue-specific MMSplice
- **Pangolin** - Sequence-based splice predictor

## Benchmark Design
- **Positive**: Variants with PIP >= 0.9 (high-confidence causal sQTLs)
- **Negative**: Variants with PIP <= 0.1
- **Tissues**: Brain Cortex, Liver, Whole Blood, Lung, Heart, Colon

## Quick Start

```bash
# Generate plot from provided scores
python plot_benchmark_results.py
```

To reproduce from scratch:
```bash
# 1. Download SuSiE data (see below)
# 2. Place prediction files in data/predictions/
# 3. Run benchmark
python consolidate_scores.py
python plot_benchmark_results.py
```

## Output Files

| File | Description |
|------|-------------|
| `results/consolidated_scores.csv` | Per-variant scores (provided for reference) |
| `results/roc_pr_curves_all_tissues.png` | ROC and PR curves |

> **Note**: Raw prediction files (`data/predictions/`) are ~550 MB and not included in this repository. We provide `results/consolidated_scores.csv` which contains all scores needed to reproduce the analysis and figures.

## Downloading SuSiE Fine-Mapping Data

1. Go to [GTEx Portal](https://gtexportal.org/home/datasets)
2. Navigate to **GTEx Analysis V10** → **Single-Tissue sQTL**
3. Download SuSiE fine-mapping results for each tissue
4. Place `.parquet` files in `data/susie/`
5. Gencode version: v47

Required files:
- `Brain_Cortex.v10.sQTLs.SuSiE_summary.parquet`
- `Liver.v10.sQTLs.SuSiE_summary.parquet`
- `Whole_Blood.v10.sQTLs.SuSiE_summary.parquet`
- `Lung.v10.sQTLs.SuSiE_summary.parquet`
- `Heart_Atrial_Appendage.v10.sQTLs.SuSiE_summary.parquet`
- `Colon_Sigmoid.v10.sQTLs.SuSiE_summary.parquet`

## Generating Predictions

Below are the commands used to generate predictions in `data/predictions/`.

### 1. Prepare Variant List

Extract variants (PIP >= 0.9 or PIP <=0.1) from SuSiE parquet files and convert to VCF:

```bash
# Output: sQTL_variants.vcf
```

### 2. SpliceAI

```bash
spliceai -I sQTL_variants.vcf \
         -O sQTL_variants.spliceai.vcf \
         -R hg38.fa \
         -A grch38 \
         -D 1000
```
- **Output**: `sQTL_variants.spliceai.vcf`
- **Score**: Max of DS_AG, DS_AL, DS_DG, DS_DL

### 3. Pangolin

```bash
pangolin sQTL_variants.vcf \
         hg38.fa \
         gencode.v47.annotation.db \
         sQTL_variants.pangolin.vcf
```
- **Output**: `sQTL_variants.pangolin.vcf`
- **Score**: Splice impact score from INFO field

### 4. MMSplice/MTSplice

```bash
python run_mmsplice.py sQTL_variants.vcf \
                       sQTL_variants.mmsplice.tsv \
                       gencode.v47.annotation.gtf \
                       hg38.fa
```
- **Output**: `sQTL_variants.mmsplice.tsv`
- **Score**: Mean absolute tissue-specific delta logit PSI

### 5. Otari

```bash
# Run Otari model for tissue-specific splice effect prediction
# Output: sQTL_variants.Otari.tsv
```
- **Score**: Sum of max absolute tissue effects across transcripts

## consolidated_scores.csv Format

```csv
variant_id,tissue,label,Otari,SpliceAI,MTSplice,Pangolin
chr1:12345:A:G,Brain_Cortex,1,0.523,0.12,0.089,0.34
chr1:67890:C:T,Brain_Cortex,0,0.001,0.01,0.005,0.02
...
```

| Column | Description |
|--------|-------------|
| `variant_id` | chr:pos:ref:alt format |
| `tissue` | GTEx tissue name |
| `label` | 1 = positive (PIP >= 0.9), 0 = negative (PIP <= 0.1) |
| `Otari` | Otari prediction score |
| `SpliceAI` | SpliceAI max delta score |
| `MTSplice` | MTSplice tissue-specific score |
| `Pangolin` | Pangolin splice impact score |

## Reference Data
- **Genome**: GRCh38 (hg38)
- **Annotation**: GENCODE v47
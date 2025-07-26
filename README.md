# Otari framework manuscript code

This repository contains code to generate the results from the manuscript, "Variant-resolved prediction of context-specific isoform variation with a graph-based attention model."

## Overview

We have organized the repository by analysis. For example, `QTL_analyses` contains the code to generate all eQTL/sQTL figures in the manuscript (Figs 3a-c). 

This code has been tested on Python 3.12.3, and includes a number of Python scripts. Please set up a conda environment, and install the packages listed in the `requirements.yml` file. Example commands:

```
git clone https://github.com/FunctionLab/otari-manuscript.git
conda env create -n otari -f requirements.yml
conda activate otari
```

## Data

Additionally, run

```
sh ./download_data.sh
```

to get the data and resource files used in some of these analyses. 
**NOTE**: Some of the data used in analysis is protected. In order to abide by the informed consents that individuals with autism and their family members signed when agreeing to participate in a SFARI cohort (e.g. SPARK), researchers must be approved by SFARI Base (https://base.sfari.org).

### Inference and variant effect prediction

If you are interested in running the Otari framework on your own list of variants, you can refer to the [otari](https://github.com/FunctionLab/otari) repository. 

## Code for results/figures

The directories correspond to the following figures/analyses:
- `holdout_eval`: Evaluations on Gao et al. holdout test set, transcript complexity analysis, per-gene correlation, and isoform v. global performance (Figs 2a, 2c-d, Sup. Figs 6-7)
- `independent_eval`: Evaluations on Glinos et al. and Leung et al. holdout validation test sets (Fig 2b, Supp. Table 1)
- `QTL_analyses`: eQTL variant effect direction prediction, sQTL-associated abundance changes, and regulatory burden of eQTLs v. sQTLs (Figs 3a-c)
- `disease_var_analyses`: Comparison of isoform-level regulatory HGMD variant effects, PTEN clinical variant case study, H19 cancer polymorphisms case study (Figs 4a-c, Supp. Figs 8-10)
- `novel_isoforms_eval`: Evaluation on Gao et al. novel isoform structures (Supp. Fig. 4)
- `ASD_analyses`: Comparisons of de novo variant effects in autistic probands v. unaffected siblings, tissue-specific analyses, and microexons analysis (Figs 5a-d)
- `utils`: Various helpful util functions

## Help
Please post in the Github issues or e-mail Aviya Litman (aviya@princeton.edu) with any questions about the repository, requests for more data, additional information about the results, etc.  

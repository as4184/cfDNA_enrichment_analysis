# FRAGILE

FRAGILE score analyzes cell-free DNA fragmentation within user-defined genomic
contexts. It comprises two complementary scores:

- **FRAGILE.b** summarizes fragment-length and GC-content characteristics.
- **FRAGILE.e** measures end-sequence enrichment relative to a sample-specific
  whole-genome baseline.

This repository accompanies the manuscript:

**Cell-free DNA fragmentation patterns reveal functional contexts of their
origin and capture age-associated variations**

## Repository contents

| Script | Analysis |
|---|---|
| `python_calculate_fragile.py` | Calculates FRAGILE.b and FRAGILE.e for genomic contexts. |
| `python_machine_learning_fragile.py` | Trains a neural-network and random-forest ensemble to classify genomic contexts from FRAGILE scores. |
| `python_global_aging_fragile.py` | Trains a global age-prediction model using nested cross-validation. |
| `python_pathway_aging_fragile.py` | Evaluates pathway-specific residual aging signatures using a frozen global age-prediction model. |

## Installation

Clone the repository:

```bash
git clone https://github.com/as4184/cfDNA_enrichment_analysis.git
cd cfDNA_enrichment_analysis
```

Create the environment using Micromamba:

```bash
micromamba create -n fragile-release -f environment.yml
```

Alternatively, create the environment from the resolved package specification:

```bash
micromamba create -n fragile-release -f environment.resolved.yml
```

The environment includes Python, bedtools, and the Python dependencies
required by the analysis scripts. The `requirements.lock.txt` file records
Python package versions; it does not install Python or bedtools.

The release environment was checked on Linux. Package availability may
differ on other operating systems.

Before execution, disable external Python package paths:

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
```

Run the commands below from the repository directory.

## Calculate FRAGILE scores

### Required inputs

1. **Fragment coordinates:** a table whose first three columns contain
   chromosome, start, and end coordinates.
2. **Genomic contexts:** one or more BED files defining the regions to analyze.
3. **Reference genome:** a FASTA file matching the coordinate assembly and
   chromosome naming used in the fragment and BED files.

Fragment and BED coordinates must use the same 0-based, half-open
coordinate convention. Reference genome and annotation files are supplied
separately by the user.

Example fragment coordinates:

```text
chr1	100000	100167
chr1	100200	100350
chr1	100400	100580
```

### Execution

Replace the example input paths with the locations of the corresponding files:

```bash
micromamba run -n fragile-release \
  python -I python_calculate_fragile.py \
  --fragments sample.fragments.tsv \
  --features contexts.bed \
  --feature-labels context \
  --genome hg38.fa \
  --outdir results/fragile
```

Multiple contexts can be supplied together:

```bash
micromamba run -n fragile-release \
  python -I python_calculate_fragile.py \
  --fragments sample.fragments.tsv \
  --features "TSS.bed CpG_islands.bed" \
  --feature-labels "TSS CpG" \
  --genome hg38.fa \
  --outdir results/fragile
```

Feature labels must match the number and order of the supplied BED files.

The calculation retains chromosomes chr1–chr22 and chrX and excludes
chrM, chrY, and contigs containing an underscore.

### Outputs

The script produces tab-separated results for each sample and context,
with the following columns:

| Column | Description |
|---|---|
| `Sample_ID` | Sample identifier |
| `Context` | Genomic-context label |
| `FRAGILE.b` | Fragment-body score |
| `FRAGILE.e` | Fragment-end score |

## Genomic-context classification

Provide a table containing `Sample_ID`, `Context`, `FRAGILE.b`, and
`FRAGILE.e`:

```bash
micromamba run -n fragile-release \
  python -I python_machine_learning_fragile.py \
  --input context_scores.fragile.tsv
```

This script trains a neural-network and random-forest ensemble using
the supplied context labels.

## Global age prediction

Provide a tab-separated training table containing:

```text
Sample_ID
Age
CpG_FRAGILE.b
CpG_FRAGILE.e
TSS_FRAGILE.b
TSS_FRAGILE.e
```

Each sample occupies one row, with these fields as columns.

```bash
micromamba run -n fragile-release \
  python -I python_global_aging_fragile.py \
  --input global_training.tsv \
  --outdir results/global_aging \
  --seed 42 \
  --outer-folds 10 \
  --iters 70 \
  --jobs 1
```

The analysis performs nested cross-validation for model selection and
fits a final global age-prediction model.

## Pathway-specific aging analysis

This analysis requires:

- A fitted global age-prediction model in Joblib format.
- Out-of-fold global age predictions with columns `Sample_ID` and
  `Age_global_oof`.
- Global FRAGILE features with columns `Sample_ID`, `CpG_FRAGILE.b`,
  `CpG_FRAGILE.e`, `TSS_FRAGILE.b`, and `TSS_FRAGILE.e`.
- A pathway feature table containing `Sample_ID`, `Age`, and pathway-specific
  FRAGILE measurements. Pathway column names must follow the naming convention
  used by the script, including the `_CpG_FRAGILE.b` suffix.

```bash
micromamba run -n fragile-release \
  python -I python_pathway_aging_fragile.py \
  --model global_model.joblib \
  --age-oof age_global_oof.tsv \
  --global global_features.tsv \
  --pathway pathway_features.tsv \
  --outdir results/pathway_aging \
  --seed 42 \
  --outer-folds 10 \
  --iters 100 \
  --jobs 1
```

Sample identifiers must correspond across the input tables. Use the fitted
model and out-of-fold predictions generated by the global aging analysis.

PDF-to-image conversion optionally requires `pdf2image` and Poppler.
These dependencies are not included in the core environment.

## Command-line reference

Each script provides a help message describing its command-line options:

```bash
micromamba run -n fragile-release \
  python -I python_calculate_fragile.py --help

micromamba run -n fragile-release \
  python -I python_machine_learning_fragile.py --help

micromamba run -n fragile-release \
  python -I python_global_aging_fragile.py --help

micromamba run -n fragile-release \
  python -I python_pathway_aging_fragile.py --help
```

## Reproducibility

The repository provides version-pinned environment specifications and
explicit random-seed options for the aging analyses. Preserve input
tables, reference resources, command-line arguments, and software versions
when reproducing an analysis.

Model results may vary across hardware, numerical libraries, and parallel
execution settings. Input sequencing data and reference annotations are
not bundled with the software.

## Citation

Please cite the accompanying manuscript and the archived software release
when using FRAGILE. The repository's Releases page identifies available
software versions.

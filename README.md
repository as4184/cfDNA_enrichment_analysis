# cfDNA Fragmentomics Analysis

This repository provides Python scripts for analyzing cell-free DNA (cfDNA) fragmentomics data. It includes a pipeline to compute the **FRAGILE** score (**FR**agment of **A**typical **G**C content, **I**rregular **L**ength, and **E**nd-motif enrichment score) for user-defined genomic contexts.

## Script
- **python_calculate_fragile.py**: Builds a per-sample whole-genome baseline from fragment coordinates, intersects one or more context BED files, and outputs:
  - **FRAGILE.b** — body component from fragment length and GC probabilities  
  - **FRAGILE.e** — end-sequence enrichment between context and baseline end-motif profiles

## Requirements
- Python 3.8+
- pandas, numpy, pysam
- bedtools (for `bedtools nuc` and `bedtools intersect`)
- Reference genome FASTA (indexed for both pysam and bedtools)

### Install Python packages
```bash
python -m pip install pandas numpy pysam
```

## Inputs
- **Fragments (TSV)**: one or more files via `--fragments` (comma, space, tab, or newline separated). Each file must have at least the first three columns:
    ```
    chr    start    end    ...
    ```
  Only the first three columns are read; extra columns are ignored. Fragment TSVs may be comma or tab separated, and outputs are tab separated.

- **Features (BED)**: one or more files via `--features` (comma, space, tab, or newline separated). Each file must contain:
    ```
    chr    start    end    ...
    ```

- **Reference genome FASTA**: provided with `--genome`.

## Usage
1. Clone the repository:
    ```bash
    git clone /path/to/your_repo.git
    ```
2. Change directory:
    ```bash
    cd /path/to/your_repo
    ```
3. Run the pipeline with the required arguments.

### Example 1: Single fragment + single feature
```bash
python_calculate_fragile.py \
  --fragments /path/to/sample.fragments.tsv \
  --features  /path/to/housekeeping_TSS.bed \
  --genome    /path/to/hg38.fa \
  --outdir    /path/to/fragile_out \
  --bedtools  /path/to/bedtools
```

### Example 2: One fragment + multiple features (comma OR whitespace separated)
```bash
python_calculate_fragile.py \
  --fragments /path/to/sample.fragments.tsv \
  --features  "/path/to/housekeeping_TSS.bed /path/to/housekeeping_CpG.bed /path/to/olfactory_TSS.bed" \
  --feature-labels "HK_TSS HK_CpG OL_TSS" \
  --genome   /path/to/hg38.fa \
  --outdir   /path/to/fragile_out \
  --bedtools /path/to/bedtools
```

### Example 3: Many fragments and features from newline-separated lists
```bash
python_calculate_fragile.py \
  --fragments "$(tr '\n' ' ' < /path/to/frags.txt)" \
  --features  "$(tr '\n' ' ' < /path/to/features.txt)" \
  --feature-labels "$(tr '\n' ' ' < /path/to/feature_labels.txt)" \
  --genome   /path/to/hg38.fa \
  --outdir   /path/to/fragile_out \
  --bedtools /path/to/bedtools
```

> `--feature-labels` must match the number and order of `--features`. Labels appear as the **Context** and in output filenames.

## Output
For each `(sample, feature)` pair, the script writes:
```
/path/to/fragile_out/<sample>_<label>.fragile.tsv
```
with columns:
- **Sample_ID**
- **Context**
- **FRAGILE.b**
- **FRAGILE.e**

## Brief Method
- **Length binning (60 bins total)**:
  - bin 0: `<50`
  - bins 1–59: `50–349` in 5-bp steps
  - bin 59: `≥350`

- **GC binning (50 bins total)**:
  - divide `[0,1]` into 50 bins of width `0.02`
  - for a GC fraction `g` in `[0,1]`, the bin index is `b = min(49, floor(50*g))`

- **Chromosome filtering**: keep `chr1 to chr22` and `chrX`; exclude `chrM`, `chrY`, and any contig with `_`.

- **End-motifs**: 7-nt windows on both ends labeled `L1 to L7` (left) and `R1 to R7` (right).

### FRAGILE.b (fragment body component)
- $W_{improbability}=\sum_{n=1}^{N}\log_{2}\left(\frac{1}{P_{size}\times P_{GC}}\right)$
- $W_{probability}=\sum_{n=1}^{N}\log_{2}\left(\frac{1}{\left(1-P_{size}\right)\times\left(1-P_{GC}\right)}\right)$
- $\mathrm{FRAGILE}.b\ \mathrm{score}=\frac{W_{improbability}}{W_{probability}+W_{improbability}}$

### FRAGILE.e (fragment end-sequence component)
Build A/T/G/C counts at positions `L1 to L7, R1 to R7` for the context and the whole genome baseline, normalize per position, and sum:
- $\mathrm{FRAGILE}.e\ \mathrm{score}=\sum_{i=1}^{14}\sum_{j=1}^{4}P_{i,j}(k)\log_{2}\left(\frac{P_{i,j}(k)}{q_{i,j}}\right)$


## Citation
If this software is useful in your work, please cite this repository and any associated publication (upcoming).

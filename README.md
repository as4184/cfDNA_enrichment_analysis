# cfDNA Fragmentomics Analysis

This repository provides Python scripts for analyzing cell-free DNA (cfDNA) fragmentomics data. It includes tools for calculating metrics such as the FRAGILE score and Kullback-Leibler Divergence (KLD).

## Scripts
- **python_FRAGILE_chunk_efficient.py**: Processes .bed files in chunks to calculate the FRAGILE score efficiently. The FRAGILE score quantifies the proportion of atypical and irregular cfDNA fragment length and GC distribution.
- **python_kld.py**: Calculates the Kullback-Leibler Divergence (KLD) between cfDNA fragment end motifs in different genomic contexts and the whole genome (or a user defined control context).

## Requirements
- Python 3
- Pandas
- Numpy
- argparse

## Usage
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/as4184/cfDNA_enrichment_analysis.git
   ```

2. Navigate to the repository directory:
   ```bash
   cd cfDNA_enrichment_analysis
   ```

3. Run the scripts with the required arguments:

### FRAGILE Score Calculation
To calculate the FRAGILE score:
   ```bash
   python python_FRAGILE_chunk_efficient.py <input_directory> <output_directory>
   ```
- `<input_directory>`: Directory containing .bed files.
- `<output_directory>`: Directory where the FRAGILE scores will be saved.

Example:
   ```bash
   python python_FRAGILE_chunk_efficient.py ./data ./results
   ```

### KLD Calculation
To calculate the Kullback-Leibler Divergence:
   ```bash
   python python_kld.py <end_seq_dir> <end_seq_intersect_dir> <output_file_path>
   ```
- `<end_seq_dir>`: Directory containing whole genome base count summary files.
- `<end_seq_intersect_dir>`: Directory containing genomic context base count summary files.
- `<output_file_path>`: Path to save the KLD results file.

Example:
   ```bash
   python python_kld.py ./end_seq ./end_seq_intersect ./results/kld_values.tsv
   ```

## Output
### FRAGILE Score Script
The script generates a `FRAGILE_scores.tsv` file with the following columns:
- **Sample ID**
- **Genomic Context**
- **FRAGILE score**

### KLD Script
The script generates a `kld_values.tsv` file with the following columns:
- **Sample ID**
- **Genomic Context**
- **KLD**

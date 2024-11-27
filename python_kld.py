import os
import pandas as pd
import numpy as np
import argparse

def calculate_kld(p, q):
    """Calculate Kullback-Leibler Divergence between two probability distributions for each position."""
    # Convert arrays to float to avoid type mismatch issues when adding pseudocounts
    p = p.astype(float)
    q = q.astype(float)

    pseudocount = 1e-16
    p += pseudocount
    q += pseudocount

    # Normalize to get probabilities
    p = p / p.sum(axis=0)
    q = q / q.sum(axis=0)

    # Calculate KLD for each position and sum across all positions
    kld_per_position = np.sum(p * np.log2(p / q), axis=0)  # Calculate KLD for each position
    return np.sum(kld_per_position)  # Sum KLD scores across all positions

def get_genomic_context_intersect(filename):
    """Extract genomic context from the file name. Modify this part according to the context coordinate .tsv file name"""
    parts = filename.split('_')
    data_index = parts.index('data') + 1
    context_parts = parts[data_index:-3]  # Excluding 'base_count_summary.tsv'
    context = '_'.join(context_parts).replace('.bed', '').replace('hg38', '').rstrip('_')
    return context

def main(end_seq_dir, end_seq_intersect_dir, output_file_path):
    """Main function to calculate KLD for all files."""
    kld_results = []
    print(f"Starting KLD calculation for files in {end_seq_dir}")
    for sample_file in os.listdir(end_seq_dir):
        if sample_file.endswith("_base_count_summary.tsv"):
            sample_id = sample_file.split('_base_count_summary.tsv')[0]
            sample_path = os.path.join(end_seq_dir, sample_file)
            print(f"Processing whole genome file: {sample_file}")
            df_whole_genome = pd.read_csv(sample_path, sep='\t', index_col='Base').fillna(0)
            print(f"Whole genome dataframe for {sample_id} loaded. Columns: {df_whole_genome.columns.tolist()}")

            for context_file in os.listdir(end_seq_intersect_dir):
                if context_file.startswith(sample_id) and context_file.endswith("_base_count_summary.tsv"):
                    context_path = os.path.join(end_seq_intersect_dir, context_file)
                    print(f"Processing context file: {context_file}")
                    df_context = pd.read_csv(context_path, sep='\t', index_col='Base').fillna(0)
                    print(f"Context dataframe for {sample_id}, {context_file} loaded. Columns: {df_context.columns.tolist()}")

                    # Ensure alignment of columns (positions) in both dataframes
                    aligned_positions = df_whole_genome.columns.intersection(df_context.columns)
                    df_whole_genome_aligned = df_whole_genome[aligned_positions]
                    df_context_aligned = df_context[aligned_positions]

                    # Calculate the KLD for each position individually and sum for total divergence
                    total_kld = calculate_kld(df_context_aligned.values, df_whole_genome_aligned.values)

                    genomic_context = get_genomic_context_intersect(context_file)
                    print(f"Calculated KLD for {sample_id} in {genomic_context}: {total_kld}")

                    kld_results.append({
                        'sample_ID': sample_id,
                        'Genomic Context': genomic_context,
                        'KLD': total_kld
                    })

    # Save KLD results to a .tsv file
    print(f"Saving KLD results to {output_file_path}")
    df_kld = pd.DataFrame(kld_results)
    df_kld.to_csv(output_file_path, sep='\t', index=False)
    print("KLD calculation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate KLD values for genomic contexts.")
    parser.add_argument("end_seq_dir", help="Directory containing whole genome (or the control context) base count summary files")
    parser.add_argument("end_seq_intersect_dir", help="Directory containing genomic context base count summary files")
    parser.add_argument("output_file_path", help="Path to save the output KLD results file")
    args = parser.parse_args()

    main(args.end_seq_dir, args.end_seq_intersect_dir, args.output_file_path)

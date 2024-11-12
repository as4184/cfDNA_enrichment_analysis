import os
import pandas as pd
import numpy as np
import argparse

def process_chunk(df):
    """Process a chunk of the DataFrame."""
    df['W_improbability'] = np.log2(1 / (df['p_length'] * df['p_gc']))
    df['W_probability'] = np.log2(1 / ((1 - df['p_length']) * (1 - df['p_gc'])))
    return df['W_improbability'].sum(), df['W_probability'].sum()

def main(bed_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path = os.path.join(output_directory, "FRAGILE_scores.tsv")
    aggregated_data = {}

    for filename in os.listdir(bed_directory):
        if filename.endswith(".bed"):
            parts = filename.split('_')
            sample_id = parts[-1].split('.')[0]
            genomic_context = '_'.join(parts[1:-1]).replace("data_genomic_", "").replace("data_chromatin_", "")
            file_path = os.path.join(bed_directory, filename)

            for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['chr', 'start', 'end', 'length', 'gc_content', 'p_length', 'p_gc'], chunksize=10000):
                chunk_filtered = chunk[(chunk['p_length'] > 0) & (chunk['p_length'] < 1) & (chunk['p_gc'] > 0) & (chunk['p_gc'] < 1)].copy()
                if not chunk_filtered.empty:
                    W_improbability_sum, W_probability_sum = process_chunk(chunk_filtered)
                    key = (sample_id, genomic_context)
                    if key not in aggregated_data:
                        aggregated_data[key] = [W_improbability_sum, W_probability_sum]
                    else:
                        aggregated_data[key][0] += W_improbability_sum
                        aggregated_data[key][1] += W_probability_sum

    with open(output_file_path, 'w', newline='') as file_out:
        csv_writer = csv.writer(file_out, delimiter='\t')
        csv_writer.writerow(['Sample ID', 'Genomic Context', 'FRAGILE score'])
        for (sample_id, genomic_context), (W_improbability, W_probability) in aggregated_data.items():
            fragile_score = W_improbability / (W_improbability + W_probability)
            csv_writer.writerow([sample_id, genomic_context, fragile_score])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FRAGILE scores from .bed files.")
    parser.add_argument("bed_directory", help="Directory containing .bed files")
    parser.add_argument("output_directory", help="Directory to save output files")
    args = parser.parse_args()
    main(args.bed_directory, args.output_directory)

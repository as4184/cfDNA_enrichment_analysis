#!/usr/bin/env python3
# python_calculate_fragile.py

# This program computes the FRAGILE score (FRagments of Atypical GC, Irregular Length and End Sequence Enrichment)
# and reports its two components:
#   • FRAGILE.b -> fragment body component from fragment length and GC probabilities
#   • FRAGILE.e -> fragment end-sequence enrichment component from Kullback–Leibler divergence of end-motif profiles

# Required input files:
#   • One or more fragment TSV files via --fragments (comma- or whitespace-separated).
#     Each fragment TSV must have at least the first three columns: chr, start, end. Extra columns are allowed and ignored.
#   • One or more feature BED files via --features (comma- or whitespace-separated).
#     Each feature BED must contain genomic intervals with columns: chr, start, end.
#   • A reference genome FASTA via --genome for nucleotide content and end-sequence extraction.

# Output:
#   One TSV per (sample, feature) with columns: Sample_ID, Context, FRAGILE.b, FRAGILE.e

# Usage examples:
#   # Single fragment file + single feature
#   python_calculate_fragile.py \
#     --fragments /path/to/sample.fragments.tsv \
#     --features  /path/to/housekeeping_TSS.bed \
#     --genome    /path/to/hg38.fa \
#     --outdir    /path/to/fragile_out \
#     --bedtools  /path/to/bedtools
#
#   # One fragment + multiple features (comma OR whitespace separated)
#   python_calculate_fragile.py \
#     --fragments /path/to/sample.fragments.tsv \
#     --features  "/path/to/housekeeping_TSS.bed /path/to/housekeeping_CpG.bed /path/to/olfactory_TSS.bed" \
#     --feature-labels "HK_TSS HK_CpG OL_TSS" \
#     --genome   /path/to/hg38.fa \
#     --outdir   /path/to/fragile_out \
#     --bedtools /path/to/bedtools
#
#   # Many fragments and features from newline-separated lists
#   # (each file has one path per line)
#   python_calculate_fragile.py \
#     --fragments "$(tr '\n' ' ' < /path/to/frags.txt)" \
#     --features  "$(tr '\n' ' ' < /path/to/features.txt)" \
#     --feature-labels "$(tr '\n' ' ' < /path/to/feature_labels.txt)" \
#     --genome   /path/to/hg38.fa \
#     --outdir   /path/to/fragile_out \
#     --bedtools /path/to/bedtools

import os, sys, re
import numpy as np
import pandas as pd
import pysam
import tempfile
import subprocess
from optparse import OptionParser

# Define the functions for the script

def filter_autosomes_X(df: pd.DataFrame) -> pd.DataFrame:
    """Keep chr1 to chr22 and chrX; exclude chrM, chrY, and contigs"""
    df = df.copy()
    df["chr"] = df["chr"].astype(str)
    keep = df["chr"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X)$")
    return df.loc[keep]

def ensure_length_1based_inclusive(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fragment length as end - start for 0-based, half-open BED coordinates."""
    df = df.copy()
    df["len"] = (df["end"].astype(int) - df["start"].astype(int))
    return df

def numeric_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert probability columns to numeric if present."""
    for c in ("p_length", "p_gc"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def gc_size_distributions(df: pd.DataFrame):
    """
    Build genome-wide empirical probability distributions.

    Length bins (62 total):
      0:      <50 bp
      1-60:   50-349 bp, in 5-bp bins
      61:     >=350 bp

    GC bins (50 total):
      0-49:   GC fractions in 2% intervals

    Returns size_prob and gc_prob.
    """
    n = len(df)
    if n == 0:
        raise ValueError("Cannot calculate probabilities from an empty dataset")

    L = df["len"].to_numpy(dtype=int)
    size_idx = np.where(
        L < 50,
        0,
        np.where(L >= 350, 61, ((L - 50) // 5) + 1)
    ).astype(int)

    size_prob = np.bincount(size_idx, minlength=62) / n

    G = np.clip(
        np.floor(df["gc"].to_numpy(dtype=float) * 50).astype(int),
        0, 49
    )
    gc_prob = np.bincount(G, minlength=50) / n

    return size_prob, gc_prob

def assign_probs_from_distributions(df: pd.DataFrame, size_prob: np.ndarray, gc_prob: np.ndarray) -> pd.DataFrame:
    """If the intersected file lacks the probability columns, attach p_length and p_gc to each fragment from the whole genome's distributions."""
    df = df.copy()
    L = df["len"].to_numpy()
    size_idx = np.where(L < 50,0,np.where(L >= 350, 61, ((L - 50) // 5) + 1)).astype(int)
    df["p_length"] = size_prob[size_idx]

    G = np.clip(np.floor(df["gc"].to_numpy() * 50).astype(int), 0, 49)
    df["p_gc"] = gc_prob[G]
    return df

def fragile_b_component(df: pd.DataFrame) -> float:
    """
    Compute FRAGILE.b from per-fragment probabilities.
    Uses fragments with 0 < p_length < 1 and 0 < p_gc < 1.
    """
    sub = df[(df["p_length"] > 0) & (df["p_length"] < 1) &
             (df["p_gc"]     > 0) & (df["p_gc"]     < 1)]
    if sub.empty:
        return float("nan")
    w_improbability  = np.log2(1.0 / (sub["p_length"].to_numpy() * sub["p_gc"].to_numpy())).sum()
    w_probability = np.log2(1.0 / ((1.0 - sub["p_length"].to_numpy()) * (1.0 - sub["p_gc"].to_numpy()))).sum()
    denom = w_improbability + w_probability
    return (w_improbability / denom) if denom > 0 else float("nan")

def end_counts(df: pd.DataFrame, fasta: pysam.FastaFile) -> pd.DataFrame:
    """
    Count A/T/G/C at positions flanking both ends:
      Left end: 7 nt window centered at the start (0-based): [start-3, start+4)
      Right end: 7 nt window centered at the end: [end-4, end+3)
    Returns a DataFrame with index A,T,G,C and 14 columns labeled L1 to L7 for left end, and R1 to R7 for right end.
    """
    chrom_len = {c: fasta.get_reference_length(c) for c in fasta.references}
    cnt = {b: [0]*14 for b in "ATGC"}

    for r in df[["chr", "start", "end"]].itertuples(index=False):
        chr_, s, e = r
        l_start = max(0, int(s) - 3); l_end = min(chrom_len.get(chr_, 0), int(s) + 4)
        r_start = max(0, int(e) - 4); r_end = min(chrom_len.get(chr_, 0), int(e) + 3)

        lseq = fasta.fetch(chr_, l_start, l_end).upper()
        rseq = fasta.fetch(chr_, r_start, r_end).upper()

        if len(lseq) < 7: lseq = lseq + "N"*(7 - len(lseq))
        if len(rseq) < 7: rseq = rseq + "N"*(7 - len(rseq))

        merged = lseq + rseq  # length 14
        for i, b in enumerate(merged):
            if b in cnt:
                cnt[b][i] += 1

    pos_labels = [f"L{i}" for i in range(1,8)] + [f"R{i}" for i in range(1,8)]
    return pd.DataFrame(cnt, index=pos_labels).T  # rows: A/T/G/C; cols: L1 to L7, R1 to R7

def kld_end_profiles(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Kullback–Leibler divergence between end-motif profiles.
    p and q are DataFrames with rows A,T,G,C and identical position columns.
    """
    cols = [c for c in q.columns if c in p.columns]
    if not cols:
        return float("nan")
    P = p[cols].to_numpy(dtype=float)
    Q = q[cols].to_numpy(dtype=float)

    eps = 1e-16
    P += eps; Q += eps
    P /= P.sum(axis=0, keepdims=True)
    Q /= Q.sum(axis=0, keepdims=True)

    kld_per_pos = np.sum(P * np.log2(P / Q), axis=0)
    return float(np.sum(kld_per_pos))

def read_frag_file(path: str) -> pd.DataFrame:
    """
    Read a per-fragment file with columns:
      chr, start, end, len, gc, p_length, p_gc
    Extra columns are ignored.
    """
    cols = ["chr","start","end","len","gc","p_length","p_gc"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, usecols=range(min(7, len(cols))), engine="c")
    df[["start","end"]] = df[["start","end"]].astype(int)
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df = ensure_length_1based_inclusive(df)
    df = numeric_probs(df)
    df = filter_autosomes_X(df)
    df[["start","end"]] = np.sort(df[["start","end"]].to_numpy(), axis=1)
    return df

def read_fragments_first_three(fragment_tsv: str) -> pd.DataFrame:
    """
    Read the first three columns chr,start,end from a fragment TSV that may be comma- or tab-separated.
    Returns a DataFrame with columns chr,start,end (integers for start/end).
    """
    # Try regex separator supporting commas or any whitespace
    df = pd.read_csv(fragment_tsv, sep=r'[,\s]+', engine="python", header=None, usecols=[0,1,2])
    df.columns = ["chr","start","end"]
    df = df.dropna()
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df

def run_bedtools_nuc_from_fragments(fragment_tsv: str, bedtools_bin: str, genome_fa: str, tmpdir: str) -> str:
    """
    Prepare a BED from the first three columns (chr, start, end) of a fragment TSV, filter to chr1 to chr22 and chrX,
    and run: bedtools nuc -fi <genome_fa> -bed <bedfile> > <gcfile>.
    Returns the path to the generated GC file.
    """
    base = os.path.splitext(os.path.basename(fragment_tsv))[0]
    bedfile = os.path.join(tmpdir, f"{base}.bed")
    gcfile  = os.path.join(tmpdir, f"{base}.gc.tsv")

    df = read_fragments_first_three(fragment_tsv)
    df = filter_autosomes_X(df)
    df[["start","end"]] = np.sort(df[["start","end"]].to_numpy(), axis=1)
    df.to_csv(bedfile, sep="\t", header=False, index=False)

    with open(gcfile, "w") as outfh:
        subprocess.run([bedtools_bin, "nuc", "-fi", genome_fa, "-bed", bedfile],
                       check=True, stdout=outfh)

    return gcfile

def parse_bedtools_nuc_gc(gcfile: str) -> pd.DataFrame:
    """
    Parse bedtools nuc output using its actual column headers.

    Calculate GC fraction consistently, excluding N bases
    where applicable. Discard fragments consisting entirely of Ns.

    Returns:
        chr, start, end, len, gc
    """
    with open(gcfile, "r") as fh:
        header_line = fh.readline().strip()

    if not header_line.startswith("#"):
        raise ValueError(
            "Missing bedtools nuc header. Refusing to guess GC column."
        )

    headers = [
        re.sub(r"^\d+_", "", x.lstrip("#").strip().lower())
        for x in header_line.split("\t")
    ]

    required = ["pct_gc", "num_c", "num_g", "num_n", "seq_len"]

    missing = [x for x in required if x not in headers]
    if missing:
        raise ValueError(
            f"Missing expected bedtools nuc columns: {missing}"
        )

    df = pd.read_csv(
        gcfile,
        sep="\t",
        comment="#",
        header=None
    )

    if df.empty:
        raise ValueError("No fragment records found")

    if df.shape[1] != len(headers):
        raise ValueError(
            "bedtools nuc header and data column counts differ"
        )

    def get_numeric(name):
        idx = headers.index(name)
        return pd.to_numeric(
            df.iloc[:, idx],
            errors="raise"
        ).to_numpy(dtype=float)

    pct_gc = get_numeric("pct_gc")
    num_c = get_numeric("num_c")
    num_g = get_numeric("num_g")
    num_n = get_numeric("num_n")
    seq_len = get_numeric("seq_len")

    starts = pd.to_numeric(
        df.iloc[:, 1], errors="raise"
    ).to_numpy(dtype=int)

    ends = pd.to_numeric(
        df.iloc[:, 2], errors="raise"
    ).to_numpy(dtype=int)

    fragment_len = ends - starts

    if not np.array_equal(fragment_len, seq_len):
        raise ValueError(
            "Fragment coordinates do not match sequence lengths"
        )

    if np.any(num_n > seq_len):
        raise ValueError("N counts exceed fragment lengths")

    # Exclude zero-length and entirely N-containing fragments.
    keep = (seq_len > 0) & (num_n < seq_len)

    # Match the original Perl handling of N-containing fragments.
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected_gc = (num_c + num_g) / (seq_len - num_n)

    gc = np.where(num_n > 0, corrected_gc, pct_gc)

    if not np.all(
        np.isfinite(gc[keep])
        & (gc[keep] >= 0)
        & (gc[keep] <= 1)
    ):
        raise ValueError("Invalid GC fractions detected")

    out = pd.DataFrame({
        "chr": df.iloc[:, 0].to_numpy()[keep],
        "start": starts[keep],
        "end": ends[keep],
        "len": fragment_len[keep],
        "gc": gc[keep]
    })

    out = filter_autosomes_X(out)

    return out[["chr", "start", "end", "len", "gc"]]

def write_size_gc_summary(df: pd.DataFrame, outpath: str):
    """
    From per-fragment chr, start, end, len, gc, compute p_length and p_gc based on this file's distributions
    and write: chr, start, end, len, gc, p_length, p_gc
    """
    size_prob, gc_prob = gc_size_distributions(df)
    df2 = assign_probs_from_distributions(df, size_prob, gc_prob)
    df2[["chr","start","end","len","gc","p_length","p_gc"]].to_csv(outpath, sep="\t", header=False, index=False)

def run_bedtools_intersect(a_file: str, b_file: str, bedtools_bin: str, out_file: str):
    """Run: bedtools intersect -wa -a <a_file> -b <b_file> > <out_file>"""
    with open(out_file, "w") as outfh:
        subprocess.run([bedtools_bin, "intersect", "-u", "-a", a_file, "-b", b_file],
                       check=True, stdout=outfh)

def read_frag_file_from_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame with columns [chr,start,end,len,gc,p_length,p_gc] using the same policies as file reads."""
    df = df_in.copy()
    df.columns = ["chr","start","end","len","gc","p_length","p_gc"]
    df[["start","end"]] = df[["start","end"]].astype(int)
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df = ensure_length_1based_inclusive(df)
    df = numeric_probs(df)
    df = filter_autosomes_X(df)
    df[["start","end"]] = np.sort(df[["start","end"]].to_numpy(), axis=1)
    return df

# Main function

def main():
    parser = OptionParser()
    parser.add_option("--fragments", dest="fragments_csv",
                      help="List of fragment TSV paths (comma- or whitespace-separated). First three columns must be chr,start,end.")
    parser.add_option("--features", dest="features_csv",
                      help="List of feature BED paths (comma- or whitespace-separated).")
    parser.add_option("--feature-labels", dest="feature_labels_csv",
                      help="Labels for --features (comma- or whitespace-separated); defaults to basename without extension")
    parser.add_option("--genome", dest="genome_fa", help="Reference genome FASTA")
    parser.add_option("--outdir", dest="outdir", help="Output directory for TSV results")
    parser.add_option("--bedtools", dest="bedtools_bin", default="bedtools",
                      help="Path to bedtools executable (default: bedtools)")
    parser.add_option("--tmpdir", dest="tmpdir", default=None,
                      help="Temporary directory for intermediate BED/TSV files")

    (opt, args) = parser.parse_args()

    if not opt.fragments_csv or not opt.features_csv or not opt.genome_fa or not opt.outdir:
        sys.exit("Required: --fragments, --features, --genome, --outdir")

    os.makedirs(opt.outdir, exist_ok=True)
    tmpdir = opt.tmpdir or tempfile.mkdtemp(prefix="fragile_tmp_")

    # accept commas or any whitespace (also newlines)
    split_paths = lambda s: [p for p in re.split(r'[,\s]+', s.strip()) if p] if s else []

    fasta = pysam.FastaFile(opt.genome_fa)
    fragments = split_paths(opt.fragments_csv)
    features  = split_paths(opt.features_csv)
    if opt.feature_labels_csv:
        feat_labels = split_paths(opt.feature_labels_csv)
        if len(feat_labels) != len(features):
            sys.exit("Number of --feature-labels must match number of --features.")
    else:
        feat_labels = [os.path.splitext(os.path.basename(p))[0] for p in features]

    # Process each fragment file as a separate sample baseline, then compute FRAGILE scores
    for frag_path in fragments:
        if not os.path.exists(frag_path):
            print(f"Skipping missing fragments file: {frag_path}", file=sys.stderr)
            continue

        sample_id = os.path.splitext(os.path.basename(frag_path))[0]

        # 1) bedtools nuc to calculate GC content
        gcfile = run_bedtools_nuc_from_fragments(frag_path, opt.bedtools_bin, opt.genome_fa, tmpdir)

        # 2) parse nuc output to (chr,start,end,len,gc)
        df_gc = parse_bedtools_nuc_gc(gcfile)

        # 3) build per-sample size.gc.summary.bed with probabilities
        ref_summary = os.path.join(tmpdir, f"{sample_id}.size.gc.summary.bed")
        write_size_gc_summary(df_gc, ref_summary)

        # 4) read reference summary and end profiles
        ref_df  = read_frag_file(ref_summary)
        ref_end = end_counts(ref_df, fasta)

        # distributions for assigning probabilities to contexts
        size_prob, gc_prob = gc_size_distributions(ref_df)

        # 5) intersect each feature and compute FRAGILE components
        for feat_path, label in zip(features, feat_labels):
            if not os.path.exists(feat_path):
                print(f"Skipping missing feature: {feat_path}", file=sys.stderr)
                continue

            ctx_out = os.path.join(tmpdir, f"out_{label}_{sample_id}.bed")
            run_bedtools_intersect(ref_summary, feat_path, opt.bedtools_bin, ctx_out)

            # read context fragments that already should have columns intersected from ref_summary
            ctx_df_raw = pd.read_csv(ctx_out, sep="\t", header=None, engine="c")

            if ctx_df_raw.shape[1] >= 7:
                ctx_df = ctx_df_raw.iloc[:, :7].copy()
                ctx_df.columns = ["chr","start","end","len","gc","p_length","p_gc"]
                ctx_df = read_frag_file_from_dataframe(ctx_df)
            else:
                coords = ctx_df_raw.iloc[:, :3].copy()
                coords.columns = ["chr","start","end"]
                coords["start"] = coords["start"].astype(int)
                coords["end"]   = coords["end"].astype(int)
                coords = filter_autosomes_X(coords)
                coords[["start","end"]] = np.sort(coords[["start","end"]].to_numpy(), axis=1)
                coords = ensure_length_1based_inclusive(coords)
                ctx_df = coords.merge(ref_df[["chr","start","end","gc"]], on=["chr","start","end"], how="left")
                ctx_df = assign_probs_from_distributions(ctx_df, size_prob, gc_prob)

            fragile_b = fragile_b_component(ctx_df)
            ctx_end   = end_counts(ctx_df, fasta)
            fragile_e = kld_end_profiles(ctx_end, ref_end)

            out_path = os.path.join(opt.outdir, f"{sample_id}_{label}.fragile.tsv")
            pd.DataFrame([[sample_id, label, fragile_b, fragile_e]],
                         columns=["Sample_ID","Context","FRAGILE.b","FRAGILE.e"]).to_csv(out_path, sep="\t", index=False)
            print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()

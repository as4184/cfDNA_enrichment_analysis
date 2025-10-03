#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python_pathway_aging_fragile.py
===============================

Build pathway-specific aging models on top of a global FRAGILE age clock. For each pathway, the script learns a residual model (pathway features -> Age - Age_global_oof), then combines that residual with the global prediction to obtain a pathway-refined age.
It also summarizes performance across pathways and produces figures and helper tables.

Inputs
------
1) --model      : joblib of the global clock (trained previously using python_global_aging_fragile.py)
2) --age-oof    : TSV with columns: Sample_ID, Age_global_oof
3) --global     : TSV with columns: Sample_ID, CpG_FRAGILE.b, CpG_FRAGILE.e, TSS_FRAGILE.b, TSS_FRAGILE.e
4) --pathway    : TSV with columns:
                  - Sample_ID, Age
                  - For each <PATHWAY>, four columns:
                        <PATHWAY>_CpG_FRAGILE.b
                        <PATHWAY>_CpG_FRAGILE.e
                        <PATHWAY>_TSS_FRAGILE.b
                        <PATHWAY>_TSS_FRAGILE.e

Outputs (in --outdir)
---------------------
- <PATHWAY>_aging_predictions.tsv  : per-sample predictions and deltas for each pathway
- pathway_summary.tsv              : CV metrics and chosen model per pathway
- all_pathways_deltaAge.tsv        : matrix of (Predicted_Age - Chronological_Age) across pathways
- all_pathways_deltaAge2.tsv       : matrix of residuals added to global age
- mean_delta_values.tsv            : mean delta per pathway
- slopes_global.tsv                : slope( PathwayPredAge ~ GlobalPredAge ) per pathway
- mean_deltaAge_bar.png            : bar plot of mean deltas (descending)
- slope_bar.png                    : bar plot of slopes (descending)
- deltaAge_heatmap_top9.png        : heatmap for top-9 pathways by slope
- deltaAge2_heatmap_top9.png       : heatmap for top-9 residuals by slope
- deltaAge_heatmap_bottom9.png     : heatmap for bottom-9 pathways by slope
- deltaAge2_heatmap_bottom9.png    : heatmap for bottom-9 residuals by slope
- pathway_vs_global_grid_top9.pdf  : scatter grid (Pathway vs Global) for top-9
- pathway_vs_global_grid_top9.png  : first-page thumbnail (if pdf2image available)
- pathway_vs_global_grid_bottom9.pdf / .png : scatter grid (Pathway vs Global) for bottom-9

Usage
-----
python python_pathway_aging_fragile.py \
  --model path/to/best_clock.joblib \
  --age-oof path/to/age_global_oof.tsv \
  --global  path/to/merged_patient_data_global.tsv \
  --pathway path/to/merged_patient_data_pathway.tsv \
  --outdir  path/to/outdir \
  --seed 42 --outer-folds 10 --iters 100 --jobs 8

Requirements
------------
- Python 3.x
- pandas, numpy, matplotlib, seaborn, joblib, SciPy
- scikit-learn
- (optional) pdf2image for PNG thumbnail of scatter grids
"""

import sys
import re
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from optparse import OptionParser
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import randint, uniform, loguniform, linregress

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")


def ts() -> str:
    """Timestamp label for logs."""
    return f"[{datetime.now():%H:%M:%S}]"


def parse_options():
    parser = OptionParser(
        usage="usage: %prog --model <joblib> --age-oof <tsv> --global <tsv> --pathway <tsv> --outdir <dir> [options]",
        description="Pathway-specific residual aging models built on a frozen global cfDNA age clock.",
    )
    parser.add_option("--model", dest="model_path", metavar="FILE",
                      help="Path to frozen global clock joblib file.")
    parser.add_option("--age-oof", dest="age_oof_tsv", metavar="FILE",
                      help="TSV with columns: Sample_ID, Age_global_oof.")
    parser.add_option("--global", dest="global_tsv", metavar="FILE",
                      help="TSV with columns: Sample_ID, CpG_FRAGILE.b, CpG_FRAGILE.e, TSS_FRAGILE.b, TSS_FRAGILE.e.")
    parser.add_option("--pathway", dest="pathway_tsv", metavar="FILE",
                      help="TSV with Sample_ID, Age, and per-pathway fragile columns.")
    parser.add_option("--outdir", dest="outdir", metavar="DIR",
                      help="Output directory (created if missing).")

    parser.add_option("--seed", dest="seed", type="int", default=42,
                      help="Random seed (default: %default)")
    parser.add_option("--outer-folds", dest="outer_folds", type="int", default=10,
                      help="Outer CV folds (default: %default)")
    parser.add_option("--iters", dest="n_param_iter", type="int", default=100,
                      help="RandomizedSearch iterations per model (default: %default)")
    parser.add_option("--jobs", dest="n_jobs", type="int", default=8,
                      help="Parallel jobs for CV (default: %default)")

    (opts, args) = parser.parse_args()
    for flag in ("model_path", "age_oof_tsv", "global_tsv", "pathway_tsv", "outdir"):
        if not getattr(opts, flag):
            parser.error(f"Missing required option: --{flag.replace('_', '-')}")
    return opts


def build_model_zoo(random_state: int):
    """Return dict[name] -> (Pipeline, param_space) with PCA denoising."""
    zoo = {}

    en_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("enet", ElasticNet(max_iter=6000, random_state=random_state)),
    ])
    zoo["ElasticNet"] = (
        en_pipe,
        dict(
            enet__alpha=loguniform(1e-6, 1e-1),
            enet__l1_ratio=uniform(0.0, 1.0),
        ),
    )

    gbr_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("gbr", GradientBoostingRegressor(random_state=random_state)),
    ])
    zoo["GBR"] = (
        gbr_pipe,
        dict(
            gbr__n_estimators=randint(400, 1200),
            gbr__learning_rate=loguniform(0.01, 0.3),
            gbr__max_depth=randint(2, 5),
            gbr__subsample=uniform(0.6, 0.4),
        ),
    )

    hgb_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("hgb", HistGradientBoostingRegressor(random_state=random_state)),
    ])
    zoo["HGB"] = (
        hgb_pipe,
        dict(
            hgb__max_depth=[None, 2, 3, 4],
            hgb__learning_rate=loguniform(0.02, 0.3),
            hgb__l2_regularization=loguniform(1e-4, 10.0),
            hgb__max_iter=randint(300, 900),
        ),
    )

    rfr_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("rfr", RandomForestRegressor(random_state=random_state)),
    ])
    zoo["RFR"] = (
        rfr_pipe,
        dict(
            rfr__n_estimators=randint(400, 1200),
            rfr__max_depth=[None, 4, 6, 8],
            rfr__min_samples_leaf=randint(1, 4),
        ),
    )

    mlp_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("mlp", MLPRegressor(max_iter=6000, activation="relu", random_state=random_state)),
    ])
    zoo["MLP"] = (
        mlp_pipe,
        dict(
            mlp__hidden_layer_sizes=[(60,), (80,), (60, 30)],
            mlp__alpha=loguniform(1e-6, 1e-2),
            mlp__learning_rate_init=loguniform(1e-4, 1e-2),
        ),
    )

    return zoo


def main():
    opts = parse_options()

    # Config
    RANDOM_STATE = int(opts.seed)
    N_OUTER_FOLD = int(opts.outer_folds)
    N_PARAM_ITER = int(opts.n_param_iter)
    N_JOBS       = int(opts.n_jobs)

    OUTDIR = Path(opts.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

    BASE = ["CpG_FRAGILE.b", "CpG_FRAGILE.e", "TSS_FRAGILE.b", "TSS_FRAGILE.e"]

    # Load frozen global clock and inputs
    print(ts(), "Loading frozen global clock")
    clf_global = joblib.load(Path(opts.model_path))

    print(ts(), "Reading Age_global_oof table")
    oof_df = pd.read_csv(opts.age_oof_tsv, sep="\t", usecols=["Sample_ID", "Age_global_oof"])
    age_global_oof = dict(zip(oof_df["Sample_ID"], oof_df["Age_global_oof"]))

    print(ts(), "Reading global feature table")
    gdf = pd.read_csv(opts.global_tsv, sep="\t", usecols=["Sample_ID"] + BASE).dropna()
    age_global_full = dict(
        zip(
            gdf["Sample_ID"],
            clf_global.predict(gdf[BASE].astype(float).values),
        )
    )

    print(ts(), "Reading pathway feature table")
    rdf = pd.read_csv(opts.pathway_tsv, sep="\t").dropna(subset=["Age"])

    # Detect pathway prefixes by columns ending with _CpG_FRAGILE.b
    suffix_pat = re.compile(r"_CpG_FRAGILE\.b$")
    pathways = sorted({suffix_pat.sub("", c) for c in rdf.columns if suffix_pat.search(c)})
    if not pathways:
        sys.exit("No pathways detected from columns ending with '_CpG_FRAGILE.b'.")

    print(ts(), f"Detected {len(pathways)} pathways")

    # Containers
    summary_rows = []
    delta_matrix = []
    delta2_matrix = []
    delta_cols = []

    # Per-pathway modeling
    for pw in pathways:
        print(ts(), "Processing pathway:", pw)

        cols = [
            f"{pw}_CpG_FRAGILE.b",
            f"{pw}_CpG_FRAGILE.e",
            f"{pw}_TSS_FRAGILE.b",
            f"{pw}_TSS_FRAGILE.e",
        ]
        if not all(c in rdf.columns for c in cols):
            print(ts(), "  Missing one or more required columns; skipping.")
            continue

        dfp = rdf[["Sample_ID", "Age"] + cols].copy()
        dfp["Age_global_full"] = dfp["Sample_ID"].map(age_global_full)
        dfp["Age_global_oof"]  = dfp["Sample_ID"].map(age_global_oof)

        if dfp[["Age_global_full", "Age_global_oof"]].isna().any().any():
            print(ts(), "  Missing global age values for some samples; skipping.")
            continue

        X_full = dfp[cols].astype(float).values
        y_res  = dfp["Age"] - dfp["Age_global_oof"]

        outer = KFold(n_splits=N_OUTER_FOLD, shuffle=True, random_state=RANDOM_STATE)
        y_pred_outer = np.empty_like(y_res, dtype=float)
        winning_fold_models = []

        zoo = build_model_zoo(RANDOM_STATE)

        # Outer CV with inner RandomizedSearch
        for tr, te in outer.split(X_full):
            X_tr, X_te = X_full[tr], X_full[te]
            y_tr = y_res.iloc[tr]

            best_name, best_est, best_params, best_score = None, None, None, -np.inf

            for name, (pipe, space) in zoo.items():
                inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                rs = RandomizedSearchCV(
                    pipe, space,
                    n_iter=N_PARAM_ITER,
                    scoring="r2",
                    cv=inner,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                    verbose=0,
                )
                rs.fit(X_tr, y_tr)
                if rs.best_score() if callable(getattr(rs, "best_score", None)) else rs.best_score_ > best_score:
                    best_name   = name
                    best_est    = rs.best_estimator_
                    best_params = rs.best_params_
                    best_score  = rs.best_score_  # scikit stores in attribute

            # Guard against degenerate constant predictors
            if np.std(best_est.predict(X_te)) < 1e-4:
                # Try alternative models in a deterministic order
                for alt_name, (alt_pipe, alt_space) in zoo.items():
                    if alt_name == best_name:
                        continue
                    inner_alt = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
                    rs2 = RandomizedSearchCV(
                        alt_pipe, alt_space,
                        n_iter=N_PARAM_ITER,
                        scoring="r2",
                        cv=inner_alt,
                        n_jobs=N_JOBS,
                        random_state=RANDOM_STATE + 1,
                        verbose=0,
                    )
                    rs2.fit(X_tr, y_tr)
                    if np.std(rs2.predict(X_te)) >= 1e-4:
                        best_name, best_est, best_params = alt_name, rs2.best_estimator_, rs2.best_params_
                        break

            winning_fold_models.append((best_name, best_params))
            y_pred_outer[te] = best_est.predict(X_te)

        cv_r2   = r2_score(y_res, y_pred_outer)
        cv_rmse = mean_squared_error(y_res, y_pred_outer, squared=False)
        print(ts(), f"  CV R²={cv_r2:.3f}  CV RMSE={cv_rmse:.2f}")

        # Choose the most frequent winner across folds; refit on all
        winner_name = pd.Series([m[0] for m in winning_fold_models]).mode()[0]
        base_pipe, base_space = build_model_zoo(RANDOM_STATE)[winner_name]
        inner_full = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 2)
        final_rs = RandomizedSearchCV(
            base_pipe, base_space,
            n_iter=N_PARAM_ITER,
            scoring="r2",
            cv=inner_full,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE + 2,
            verbose=0,
        )
        final_rs.fit(X_full, y_res)
        final_est = final_rs.best_estimator_
        best_pars = final_rs.best_params_

        resid_hat = final_est.predict(X_full)
        age_path  = dfp["Age_global_full"] + resid_hat

        out = pd.DataFrame({
            "Sample_ID":         dfp["Sample_ID"],
            "Chronological_Age": dfp["Age"],
            "Age_global_full":   dfp["Age_global_full"],
            "Predicted_Age":     age_path,
            "Delta_Age":         age_path - dfp["Age"],
            "Delta_Age_2":       resid_hat,
        })
        out.to_csv(OUTDIR / f"{pw}_aging_predictions.tsv", sep="\t", index=False)

        # Collect matrices
        delta_matrix.append(out.set_index("Sample_ID")["Delta_Age"])
        delta2_matrix.append(out.set_index("Sample_ID")["Delta_Age_2"])
        delta_cols.append(pw)

        # Per-pathway scatter/diagnostic plot
        x, y_plot = dfp["Age"].values, age_path.values
        slope, intercept, r_val, p_val, _ = linregress(x, y_plot)
        quad = np.polyfit(x, y_plot, 2)
        xx = np.linspace(x.min() - 1, x.max() + 1, 200)

        plt.figure(figsize=(5, 5))
        plt.scatter(x, y_plot, s=25, alpha=0.7)
        plt.plot(xx, intercept + slope * xx, "r--", lw=2, label="linear")
        plt.plot(xx, quad[2] + quad[1] * xx + quad[0] * xx**2, "g-", lw=1.5, label="quadratic")
        rmse_full = np.sqrt(((y_plot - x) ** 2).mean())
        plt.xlabel("Chronological age (y)")
        plt.ylabel("Predicted age (y)")
        plt.title(f"{pw}\nCV R²={cv_r2:.3f}  CV RMSE={cv_rmse:.1f}\n"
                  f"r={r_val:.3f}  p={p_val:.1e}  slope={slope:.3f}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{pw}_plot.png", dpi=300)
        plt.close()

        summary_rows.append([pw, cv_r2, cv_rmse, winner_name, json.dumps(best_pars)])

    # Summary table
    summary_df = pd.DataFrame(summary_rows, columns=["Pathway", "CV_R2", "CV_RMSE", "BestModel", "BestParams"])
    summary_df.to_csv(OUTDIR / "pathway_summary.tsv", sep="\t", index=False)

    # Assemble matrices
    delta_df  = pd.concat(delta_matrix,  axis=1)
    delta2_df = pd.concat(delta2_matrix, axis=1)
    delta_df.columns  = delta_cols
    delta2_df.columns = delta_cols
    delta_df.to_csv(OUTDIR / "all_pathways_deltaAge.tsv",  sep="\t", index_label="Sample_ID")
    delta2_df.to_csv(OUTDIR / "all_pathways_deltaAge2.tsv", sep="\t", index_label="Sample_ID")

    # Mean delta bar plot
    mean_delta = delta_df.mean(axis=0)
    mean_delta.to_csv(OUTDIR / "mean_delta_values.tsv", sep="\t", header=False)

    def bar_means(series, fname, title):
        order = series.sort_values(ascending=False).index
        plt.figure(figsize=(7, 5))
        plt.barh(order, series[order], edgecolor="black")
        plt.axvline(0, color="k")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(OUTDIR / fname, dpi=300)
        plt.close()

    bar_means(mean_delta, "mean_deltaAge_bar.png", "Mean delta-Age (descending)")

    # Slopes: PathwayPredAge vs GlobalPredAge
    slopes = []
    for pw in delta_cols:
        tsv = OUTDIR / f"{pw}_aging_predictions.tsv"
        if not tsv.exists():
            slopes.append(np.nan)
            continue
        dfp = pd.read_csv(tsv, sep="\t", usecols=["Age_global_full", "Predicted_Age"]).dropna()
        slopes.append(linregress(dfp["Age_global_full"], dfp["Predicted_Age"]).slope)

    slopes_series = pd.Series(slopes, index=delta_cols)
    slopes_series.to_csv(OUTDIR / "slopes_global.tsv", sep="\t", header=False)

    def bar_slopes(series, fname, title):
        order = series.sort_values(ascending=False).index
        plt.figure(figsize=(7, 5))
        plt.barh(order, series[order], edgecolor="black")
        plt.axvline(1.0, ls="--", color="k")
        plt.xlabel("Linear slope")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(OUTDIR / fname, dpi=300)
        plt.close()

    bar_slopes(slopes_series, "slope_bar.png", "Pathway PredAge vs Global PredAge (descending)")

    # Select top/bottom 9 by slope
    top9_paths    = slopes_series.nlargest(9).index.tolist()
    bottom9_paths = slopes_series.nsmallest(9).index.tolist()

    delta_df[top9_paths].to_csv(OUTDIR / "top9_deltaAge_plot_data.tsv",   sep="\t", index_label="Sample_ID")
    delta2_df[top9_paths].to_csv(OUTDIR / "top9_deltaAge2_plot_data.tsv", sep="\t", index_label="Sample_ID")
    delta_df[bottom9_paths].to_csv(OUTDIR / "bottom9_deltaAge_plot_data.tsv",   sep="\t", index_label="Sample_ID")
    delta2_df[bottom9_paths].to_csv(OUTDIR / "bottom9_deltaAge2_plot_data.tsv", sep="\t", index_label="Sample_ID")

    # Heatmaps
    def heatmap_block(ddf, d2df, tag):
        plt.figure(figsize=(0.6 * len(ddf.columns) + 2, 6))
        sns.heatmap(ddf.T, cmap="vlag", center=0)
        plt.title(f"delta-Age heatmap ({tag})")
        plt.ylabel("Pathway")
        plt.xlabel("Sample")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"deltaAge_heatmap_{tag}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(0.6 * len(d2df.columns) + 2, 6))
        sns.heatmap(d2df.T, cmap="vlag", center=0)
        plt.title(f"delta-Age2 heatmap ({tag})")
        plt.ylabel("Pathway")
        plt.xlabel("Sample")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"deltaAge2_heatmap_{tag}.png", dpi=300)
        plt.close()

    heatmap_block(delta_df[top9_paths],  delta2_df[top9_paths],  "top9")
    heatmap_block(delta_df[bottom9_paths],  delta2_df[bottom9_paths],  "bottom9")

    # Scatter grids (Pathway Pred Age vs Global Pred Age)
    def scatter_grids(paths, tag, pdf_name, png_name):
        pdf_path = OUTDIR / pdf_name
        with PdfPages(pdf_path) as pdf:
            for pw in paths:
                tsv_path = OUTDIR / f"{pw}_aging_predictions.tsv"
                if not tsv_path.exists():
                    continue
                dfp = pd.read_csv(tsv_path, sep="\t",
                                  usecols=["Chronological_Age", "Age_global_full", "Predicted_Age"]).dropna()
                plt.figure(figsize=(4, 4))
                plt.scatter(dfp["Age_global_full"], dfp["Predicted_Age"], s=10, alpha=0.6)
                m, b = np.polyfit(dfp["Age_global_full"], dfp["Predicted_Age"], 1)
                xx = np.linspace(dfp["Age_global_full"].min(), dfp["Age_global_full"].max(), 100)
                plt.plot(xx, m * xx + b, "r--", lw=1)
                plt.xlabel("Global Pred Age")
                plt.ylabel("Pathway Pred Age")
                plt.title(pw)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        # Optional first-page PNG thumbnail
        try:
            from pdf2image import convert_from_path
            page = convert_from_path(pdf_path, first_page=1, last_page=1)[0]
            page.save(OUTDIR / png_name, "PNG")
        except (ImportError, OSError):
            print(ts(), f"pdf2image unavailable - PNG thumbnail for {tag} grid skipped")

    scatter_grids(top9_paths,    "top9",    "pathway_vs_global_grid_top9.pdf",    "pathway_vs_global_grid_top9.png")
    scatter_grids(bottom9_paths, "bottom9", "pathway_vs_global_grid_bottom9.pdf", "pathway_vs_global_grid_bottom9.png")

    print(ts(), "Finished. Outputs in", OUTDIR.resolve())


if __name__ == "__main__":
    main()

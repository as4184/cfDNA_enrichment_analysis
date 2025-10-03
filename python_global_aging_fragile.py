#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python_global_aging_fragile.py
===================================

Train a *global* age-prediction regressor from four cfDNA-derived biomarkers:
    - CpG_FRAGILE.b
    - CpG_FRAGILE.e
    - TSS_FRAGILE.b
    - TSS_FRAGILE.e

The script performs:
  1) Nested model selection across:
        - HistGradientBoostingRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor
        - Polynomial (degree 2) + ElasticNet
        - MLPRegressor
  2) Hyper-parameter tuning via RandomizedSearchCV
  3) Honest generalization estimate via outer K-Fold CV
  4) Final refit on the full dataset, export fitted model and feature importance / coefficients
  5) Scatter plot of predicted vs. chronological age

Input
-----
A TAB-delimited file containing at least the following columns:
    Sample_ID, Age, CpG_FRAGILE.b, CpG_FRAGILE.e, TSS_FRAGILE.b, TSS_FRAGILE.e

Output
------
- best_clock.joblib         : serialized best model (RandomizedSearchCV object with best_estimator_)
- age_global_oof.tsv        : out-of-fold predictions from the best model (outer CV)
- model_comparison.tsv      : outer-CV R² and RMSE for all candidate models
- feature_importance.tsv    : feature importances (tree models)  [if available]
- coefficients.tsv          : coefficients (linear models)       [if available]
- pred_vs_actual.png        : scatter plot with best-fit line, R² and RMSE
- Console logs              : concise progress and metrics

Usage
-----
  python python_global_aging_fragile.py \
      -i path/to/merged_patient_data.tsv \
      -o path/to/output_dir \
      --seed 42 --outer-folds 10 --iters 70 --jobs 1

Requirements
------------
- Python 3.x
- pandas, numpy, matplotlib
- scikit-learn >= 1.1, joblib
- SciPy
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from optparse import OptionParser

from scipy.stats import randint, uniform, linregress

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import r2_score, mean_squared_error


def log(msg: str) -> None:
    """Standardized console logger with timestamps."""
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def parse_options():
    """Parse command-line options (optparse for compatibility)."""
    parser = OptionParser(
        usage="usage: %prog -i <train.tsv> -o <outdir> [options]",
        description="Nested CV model selection and final refit for a global cfDNA age clock."
    )
    parser.add_option("-i", "--input",
                      dest="train_file",
                      metavar="FILE",
                      help="Training TSV with columns: Sample_ID, Age, CpG_FRAGILE.b, CpG_FRAGILE.e, TSS_FRAGILE.b, TSS_FRAGILE.e")
    parser.add_option("-o", "--outdir",
                      dest="outdir",
                      metavar="DIR",
                      help="Output directory (will be created if it does not exist)")
    parser.add_option("--seed",
                      dest="seed",
                      type="int",
                      default=42,
                      help="Random seed (default: %default)")
    parser.add_option("--outer-folds",
                      dest="outer_folds",
                      type="int",
                      default=10,
                      help="Number of outer CV folds (default: %default)")
    parser.add_option("--iters",
                      dest="n_param_iter",
                      type="int",
                      default=70,
                      help="RandomizedSearchCV iterations per model (default: %default)")
    parser.add_option("--jobs",
                      dest="n_jobs",
                      type="int",
                      default=1,
                      help="Parallel jobs for CV (default: %default)")

    (opts, args) = parser.parse_args()
    if not opts.train_file:
        parser.error("Please provide -i/--input path to the training TSV.")
    if not opts.outdir:
        parser.error("Please provide -o/--outdir path for outputs.")
    return opts


def main():
    opts = parse_options()

    # Configuration
    TRAIN_FILE   = opts.train_file
    OUTDIR       = Path(opts.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
    RANDOM_STATE = int(opts.seed)
    N_OUTER_FOLD = int(opts.outer_folds)
    N_PARAM_ITER = int(opts.n_param_iter)
    N_JOBS       = int(opts.n_jobs)

    FEATURES = ["CpG_FRAGILE.b", "CpG_FRAGILE.e", "TSS_FRAGILE.b", "TSS_FRAGILE.e"]
    TARGET   = "Age"

    # Load the data
    log(f"Loading training data: {TRAIN_FILE}")
    usecols = ["Sample_ID", TARGET] + FEATURES
    df = pd.read_csv(TRAIN_FILE, sep="\t", usecols=usecols).dropna()
    if df.empty:
        sys.exit("No rows remaining after NA removal. Please verify the input file and columns.")

    sample_ids = df["Sample_ID"].to_numpy()
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(float)

    log(f"Data loaded: n_samples={X.shape[0]}  n_features={X.shape[1]}")

    # DEFINE MODELS & PARAMETER GRIDS
    models = {}

    # HistGradientBoosting (robust with small/medium N)
    hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    hgb_param = dict(
        max_depth=[None, 2, 3, 4],
        learning_rate=uniform(0.02, 0.2),
        l2_regularization=uniform(0.0, 1.0),
        max_iter=randint(150, 600)
    )
    models["HGB"] = (hgb, hgb_param)

    # Classic GradientBoosting
    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gbr_param = dict(
        n_estimators=randint(200, 800),
        learning_rate=uniform(0.01, 0.2),
        max_depth=randint(2, 5),
        subsample=uniform(0.6, 0.4)
    )
    models["GBR"] = (gbr, gbr_param)

    # RandomForest
    rfr = RandomForestRegressor(random_state=RANDOM_STATE)
    rfr_param = dict(
        n_estimators=randint(200, 800),
        max_depth=[None, 3, 4, 6],
        min_samples_leaf=randint(1, 4)
    )
    models["RFR"] = (rfr, rfr_param)

    # Polynomial (deg=2) + ElasticNet
    poly_enet = Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("enet",   ElasticNet(max_iter=5000, random_state=RANDOM_STATE))
    ])
    poly_param = dict(
        enet__alpha=uniform(1e-3, 1.0),
        enet__l1_ratio=uniform(0.1, 0.9)
    )
    models["PolyEN"] = (poly_enet, poly_param)

    # Shallow MLP
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(
            batch_size='auto',
            max_iter=3000,
            activation='relu',
            random_state=RANDOM_STATE
        ))
    ])
    mlp_param = dict(
        mlp__hidden_layer_sizes=[(20,), (40,), (20, 10)],
        mlp__alpha=uniform(1e-4, 1e-2),
        mlp__learning_rate_init=uniform(1e-4, 1e-2)
    )
    models["MLP"] = (mlp, mlp_param)

    # NESTED CV SEARCH (outer CV with inner RandomizedSearchCV)
    outer = KFold(n_splits=N_OUTER_FOLD, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = {}
    best_global_model   = None   # RandomizedSearchCV object (unfitted yet)
    best_global_name    = None
    best_global_r2_mean = -np.inf
    cv_summary          = []

    for name, (base_model, param_dist) in models.items():
        log(f"Tuning {name} ...")
        inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=N_PARAM_ITER,
            scoring='r2',
            cv=inner,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=0
        )

        # Outer-CV performance using cross_val_predict on the SEARCH object
        y_hat = cross_val_predict(
            search, X, y, cv=outer, n_jobs=N_JOBS, verbose=0
        )
        oof_preds[name] = y_hat
        r2   = r2_score(y, y_hat)
        rmse = mean_squared_error(y, y_hat, squared=False)
        cv_summary.append((name, r2, rmse))
        log(f"  Outer-CV performance: R²={r2:0.3f}  RMSE={rmse:0.2f}")

        if r2 > best_global_r2_mean:
            best_global_r2_mean = r2
            best_global_model   = search
            best_global_name    = name

    # REFIT BEST MODEL ON FULL DATA
    if best_global_model is None:
        sys.exit("No model was selected. Please verify input and parameters.")

    log(f"Selected best model: {best_global_model.estimator.__class__.__name__} "
        f"(outer-CV R²={best_global_r2_mean:.3f})")

    best_global_model.fit(X, y)  # refit best (with inner CV) on all data
    joblib.dump(best_global_model, OUTDIR / "best_clock.joblib")
    log("Saved fitted model -> best_clock.joblib")

    age_oof = pd.DataFrame({
        "Sample_ID":      sample_ids,
        "Age_global_oof": oof_preds[best_global_name]
    })
    age_oof.to_csv(OUTDIR / "age_global_oof.tsv", sep="\t", index=False)
    log("Saved out-of-fold predictions -> age_global_oof.tsv")

    # SAVE MODEL SUMMARY
    (pd.DataFrame(cv_summary, columns=["Model", "OuterCV_R2", "OuterCV_RMSE"])
        .sort_values("OuterCV_R2", ascending=False)
        .to_csv(OUTDIR / "model_comparison.tsv", sep="\t", index=False))
    log("Saved model comparison -> model_comparison.tsv")

    y_pred_full = best_global_model.predict(X)
    global_r2   = r2_score(y, y_pred_full)
    global_rmse = mean_squared_error(y, y_pred_full, squared=False)

    slope, intercept, r_val, p_val, _ = linregress(y, y_pred_full)
    log(f"Full-data fit: R²={global_r2:.3f}  RMSE={global_rmse:.2f}  "
        f"Pearson r={r_val:.3f}  p={p_val:.2e}  slope={slope:.3f}")

    # Export coefficients / feature importances when available
    bst = best_global_model.best_estimator_
    if hasattr(bst, "feature_importances_"):
        pd.DataFrame({"Feature": FEATURES, "FI": bst.feature_importances_}) \
          .to_csv(OUTDIR / "feature_importance.tsv", sep="\t", index=False)
        log("Saved feature_importance.tsv")

    elif hasattr(bst, "coef_"):
        coef = bst.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[:len(FEATURES)]
        pd.DataFrame({"Feature": FEATURES, "Coef": coef}) \
          .to_csv(OUTDIR / "coefficients.tsv", sep="\t", index=False)
        log("Saved coefficients.tsv")

    # PLOT: Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred_full, s=25, alpha=0.7)
    z = np.polyfit(y, y_pred_full, 1)
    p = np.poly1d(z)
    x_lin = np.linspace(y.min() - 1, y.max() + 1, 100)
    plt.plot(x_lin, p(x_lin), lw=2)
    plt.xlabel("Chronological age (y)")
    plt.ylabel("Predicted age (y)")
    plt.title(f"Best global cfDNA clock\nR²={global_r2:.3f}  RMSE={global_rmse:.1f}")
    plt.tight_layout()
    plt.savefig(OUTDIR / "pred_vs_actual.png", dpi=300)
    plt.close()
    log("Saved plot -> pred_vs_actual.png")

    log(f"All outputs written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()

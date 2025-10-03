#!/usr/bin/env python3
# python_machine_learning_fragile.py
# This script trains a supervised multi-class classifier to predict genomic context labels (e.g., Genes, housekeeping_CpG, TSS, etc.) from per-sample FRAGILE score file. The model is an ensemble of a deep neural network (Keras) and a RandomForest (scikit-learn).
# The script performs stratified train/test split, 5-fold cross-validation with SMOTE and feature scaling, early stopping, and reports accuracy, confusion matrix, per-class sensitivity/specificity, and ROC curves.

# Input
# A TAB-delimited file (the "fragile score" table) with columns:
#   - Sample_ID : sample identifier string (e.g., SRR123456, 131-001, etc.)
#   - Context   : context label string (e.g., Genes, housekeeping_CpG, olfactory_TSS, etc.)
#   - FRAGILE.b : fragment body component score
#   - FRAGILE.e : fragment end-sequence enrichment component score

# Output
#   - Console logs: training/validation metrics, final test metrics, confusion matrix, classification report, per-class sensitivity and specificity, and overall summary.
#   - multi_class_roc_ensemble.png : per-class one-vs-rest ROC curves with AUCs.
#   - overall_roc_ensemble.png     : micro-averaged ROC curve across all classes.

# Usage
#   python python_machine_learning_fragile.py -i path/to/my_data.fragile.tsv

# Requirements
#   - Python 3.x
#   - numpy, pandas, matplotlib
#   - scikit-learn, imblearn
#   - tensorflow / keras

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from optparse import OptionParser


def print_debug(message):
    """
    A helper function to print debug messages
    to track each step clearly.
    """
    print(f"[DEBUG] {message}")


def load_and_prepare_data(input_file):
    """
    Load the FRAGILE score table
    Parameters
    ----------
    input_file : str
        Path to the TAB-delimited fragile file with columns:
        Sample_ID, Context, FRAGILE.b, FRAGILE.e

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with the expected columns for downstream steps.
    """
    print_debug(f"Loading FRAGILE score file into DataFrame: {input_file} ...")
    df = pd.read_csv(input_file, sep='\t')
    print_debug("FRAGILE score DataFrame loaded. Shape: " + str(df.shape))

    required_cols = {'Sample_ID', 'Context', 'FRAGILE.b', 'FRAGILE.e'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    long_df = df.loc[:, ['Sample_ID', 'Context', 'FRAGILE.b', 'FRAGILE.e']].copy()
    long_df.rename(columns={'Sample_ID': 'SampleID'}, inplace=True)

    print_debug("Prepared DataFrame with columns: " + ", ".join(long_df.columns))
    print_debug("Long-format DataFrame created. Shape: " + str(long_df.shape))
    return long_df


def preprocess_for_training(long_df):
    """
    Encode context labels, build feature matrix X and target y, then split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, int_to_label
    """
    print_debug("Converting 'Context' into numeric labels.")
    context_labels = long_df['Context'].unique()
    label_to_int = {ctx: i for i, ctx in enumerate(context_labels)}
    int_to_label = {v: k for k, v in label_to_int.items()}

    long_df['ContextLabel'] = long_df['Context'].map(label_to_int)

    # We have two features: FRAGILE.b and FRAGILE.e
    X = long_df[['FRAGILE.b', 'FRAGILE.e']].values
    y = long_df['ContextLabel'].values

    print_debug("Performing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print_debug(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Distribution check: train vs test
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for i in range(len(int_to_label)):
        train_count = counts_train[unique_train == i][0] if i in unique_train else 0
        test_count = counts_test[unique_test == i][0] if i in unique_test else 0
        total_count = train_count + test_count
        train_pct = 100.0 * train_count / total_count if total_count > 0 else 0
        test_pct = 100.0 * test_count / total_count if total_count > 0 else 0
        print_debug(f"Context '{int_to_label[i]}': Train={train_count}, "
                    f"Test={test_count}, Train%={train_pct:.2f}, Test%={test_pct:.2f}")

    return X_train, X_test, y_train, y_test, int_to_label


def build_deep_learning_model(input_dim, num_classes):
    """
    Build a multi-layer (deep) neural network using Keras Sequential API. Includes batch normalization and dropout for regularization.
    """
    print_debug("Building deep learning model...")
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    print_debug("Compiling model with 'adam' optimizer and 'categorical_crossentropy' loss.")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, int_to_label):
    """
    Cross-validate, train, and evaluate an ensemble of NN + RandomForest.

    Steps
    -----
    1) 5-Fold Stratified CV on the training split:
       - SMOTE on the training fold
       - Standard scaling
       - Early stopping
       - Class weights (balanced)
    2) Train RandomForest on the same fold features.
    3) Average NN and RF probabilities for fold validation evaluation; keep best NN weights.
    4) Retrain on the entire training split (with SMOTE + scaling).
    5) Evaluate on the held-out test split, print reports and save ROC plots.
    """
    print_debug("Converting integer labels to one-hot vectors for final test data...")
    num_classes = len(int_to_label)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print_debug("Starting 5-Fold Cross-Validation on the train set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_val_acc = -1.0
    best_model_weights = None
    best_rf_model = None

    fold_idx = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_idx += 1
        print_debug(f"===== K-FOLD {fold_idx} / 5 =====")

        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        classes_unique = np.unique(y_tr_fold)
        cw = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_tr_fold)
        class_weight_dict = {cls: w for cls, w in zip(classes_unique, cw)}

        print_debug("Applying SMOTE oversampling on fold training data...")
        sm = SMOTE(random_state=42)
        X_tr_fold_sm, y_tr_fold_sm = sm.fit_resample(X_tr_fold, y_tr_fold)

        scaler = StandardScaler()
        X_tr_fold_sm_sc = scaler.fit_transform(X_tr_fold_sm)
        X_val_fold_sc = scaler.transform(X_val_fold)

        y_tr_fold_sm_cat = to_categorical(y_tr_fold_sm, num_classes=num_classes)
        y_val_fold_cat = to_categorical(y_val_fold, num_classes=num_classes)

        fold_model = build_deep_learning_model(input_dim=2, num_classes=num_classes)
        early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        print_debug("Training the neural network for this fold...")
        history_fold = fold_model.fit(
            X_tr_fold_sm_sc,
            y_tr_fold_sm_cat,
            batch_size=16,
            epochs=50,
            validation_data=(X_val_fold_sc, y_val_fold_cat),
            callbacks=[early_stop],
            verbose=0,
            class_weight=class_weight_dict
        )

        train_acc_fold = history_fold.history['accuracy'][-1]
        val_acc_fold = history_fold.history['val_accuracy'][-1]
        if (train_acc_fold - val_acc_fold) > 0.1:
            print_debug(f"Fold {fold_idx} Overfitting suspected: "
                        f"train_acc={train_acc_fold:.4f}, val_acc={val_acc_fold:.4f}")
        else:
            print_debug(f"Fold {fold_idx} No significant overfitting: "
                        f"train_acc={train_acc_fold:.4f}, val_acc={val_acc_fold:.4f}")

        val_preds_nn = fold_model.predict(X_val_fold_sc)
        val_acc_nn = accuracy_score(np.argmax(y_val_fold_cat, axis=1), np.argmax(val_preds_nn, axis=1))
        print_debug(f"Fold {fold_idx} NN Validation Accuracy: {val_acc_nn:.4f}")

        print_debug("Training RandomForest on the same fold data (with SMOTE + scaling).")
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf_model.fit(X_tr_fold_sm_sc, y_tr_fold_sm)

        val_probs_rf = rf_model.predict_proba(X_val_fold_sc)
        val_ensemble_probs = (val_preds_nn + val_probs_rf) / 2.0
        val_ensemble_preds = np.argmax(val_ensemble_probs, axis=1)
        val_acc_ensemble = accuracy_score(np.argmax(y_val_fold_cat, axis=1), val_ensemble_preds)
        print_debug(f"Fold {fold_idx} Ensemble Validation Accuracy: {val_acc_ensemble:.4f}")

        if val_acc_ensemble > best_val_acc:
            best_val_acc = val_acc_ensemble
            best_model_weights = fold_model.get_weights()
            best_rf_model = rf_model

    print_debug(f"Best Ensemble Validation Accuracy across folds = {best_val_acc:.4f}")
    print_debug("Retraining final models (NN + RF) on the entire training set...")

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    scaler_final = StandardScaler()
    X_train_sm_sc = scaler_final.fit_transform(X_train_sm)

    final_model = build_deep_learning_model(input_dim=2, num_classes=num_classes)
    final_model.set_weights(best_model_weights)

    y_train_sm_cat = to_categorical(y_train_sm, num_classes=num_classes)
    early_stop_final = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    classes_unique_final = np.unique(y_train_sm)
    cw_final = compute_class_weight(class_weight='balanced', classes=classes_unique_final, y=y_train_sm)
    class_weight_final_dict = {cls: w for cls, w in zip(classes_unique_final, cw_final)}

    print_debug("Fine-tuning final neural network with entire training set...")
    history_final = final_model.fit(
        X_train_sm_sc,
        y_train_sm_cat,
        batch_size=16,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stop_final],
        verbose=0,
        class_weight=class_weight_final_dict
    )

    train_acc_final = history_final.history['accuracy'][-1]
    val_acc_final = history_final.history['val_accuracy'][-1]
    if (train_acc_final - val_acc_final) > 0.1:
        print_debug(f"Final Overfitting suspected: train_acc={train_acc_final:.4f}, "
                    f"val_acc={val_acc_final:.4f}")
    else:
        print_debug(f"Final No significant overfitting: train_acc={train_acc_final:.4f}, "
                    f"val_acc={val_acc_final:.4f}")

    print_debug("Refitting the best random forest on the entire training set...")
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_final.fit(X_train_sm_sc, y_train_sm)

    print_debug("Evaluating on the test set...")
    X_test_sc = scaler_final.transform(X_test)

    nn_test_probs = final_model.predict(X_test_sc)
    rf_test_probs = rf_final.predict_proba(X_test_sc)
    ensemble_test_probs = (nn_test_probs + rf_test_probs) / 2.0
    y_pred_ensemble = np.argmax(ensemble_test_probs, axis=1)

    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    print_debug(f"Final Ensemble Test Accuracy: {acc_ensemble:.4f}")

    cm = confusion_matrix(y_test, y_pred_ensemble)
    print_debug("Confusion Matrix:")
    print(cm)

    cr = classification_report(
        y_test,
        y_pred_ensemble,
        target_names=[int_to_label[i] for i in range(num_classes)],
        zero_division=0
    )
    print_debug("Classification Report:")
    print(cr)

    print_debug("Calculating sensitivity (recall) and specificity for each class...")
    all_sens = []
    all_spec = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        all_sens.append(sensitivity)
        all_spec.append(specificity)
        print_debug(f"Class: {int_to_label[i]}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    overall_sensitivity = np.mean(all_sens)
    overall_specificity = np.mean(all_spec)
    overall_accuracy = acc_ensemble
    print_debug(f"Overall Sensitivity (mean): {overall_sensitivity:.4f}, Overall Specificity (mean): {overall_specificity:.4f}")
    print_debug(f"Overall Accuracy: {overall_accuracy:.4f}")

    print_debug("Calculating and plotting multi-class ROC (one-vs-rest) for ensemble.")
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], ensemble_test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6), dpi=600)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 label='Class {0} (AUC = {1:0.2f})'.format(int_to_label[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC (Ensemble)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.figtext(
        0.99, 0.01,
        f"Overall Acc: {overall_accuracy:.4f}\n"
        f"Overall Sens: {overall_sensitivity:.4f}\n"
        f"Overall Spec: {overall_specificity:.4f}",
        horizontalalignment='right',
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig("multi_class_roc_ensemble.png", dpi=600, bbox_inches='tight')
    plt.close()
    print_debug("ROC figure saved as 'multi_class_roc_ensemble.png'.")

    print_debug("Calculating micro-average ROC for a single overall curve.")
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_cat.ravel(), ensemble_test_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(fpr["micro"], tpr["micro"],
             label='Overall ROC (area = %0.2f)' % roc_auc["micro"],
             color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC (Micro-average)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig("overall_roc_ensemble.png", dpi=600, bbox_inches='tight')
    plt.close()
    print_debug("Overall micro-average ROC figure saved as 'overall_roc_ensemble.png'.")


def main():
    print_debug("Starting the main program...")

    parser = OptionParser(
        usage="usage: %prog -i path/to/data.fragile.tsv",
        description="Train an NN+RF ensemble on FRAGILE.b and FRAGILE.e per-context scores."
    )
    parser.add_option(
        "-i", "--input",
        dest="fragile_tsv",
        metavar="FILE",
        help="Path to *.fragile.tsv with columns: Sample_ID, Context, FRAGILE.b, FRAGILE.e"
    )
    (options, args) = parser.parse_args()

    if not options.fragile_tsv:
        parser.error("Please provide -i/--input pointing to a *.fragile.tsv")

    long_df = load_and_prepare_data(options.fragile_tsv)
    X_train, X_test, y_train, y_test, int_to_label = preprocess_for_training(long_df)
    model = build_deep_learning_model(input_dim=2, num_classes=len(int_to_label))
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test, int_to_label)
    print_debug("All steps completed.")


if __name__ == "__main__":
    main()

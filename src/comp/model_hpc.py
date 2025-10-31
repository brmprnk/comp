#!/usr/bin/env python

"""
HPC-optimized script for running nested cross-validation on genomic data.

This script converts the logic from the original Jupyter Notebook into a parallelized,
command-line-driven application suitable for execution on a high-performance cluster.

It iterates through multiple classifiers, genomic features, and feature selection
settings (top_genes), running each combination as an independent parallel job.
The results for each experiment are saved to a unique .pkl file.

Example usage on an HPC with 64 cores:
python model_hpc.py --cores 64 --output_dir ./best_model_results/
"""

import argparse
import glob
import itertools
import multiprocessing
import os
import pickle
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# =================================================================================
# --- Helper & Data Loading Functions (from Notebook) ---
# =================================================================================


def sync_file_lists(list1: list[str], list2: list[Path]) -> list[Path | None]:
    """
    Syncs two lists of file identifiers based on the longest matching substring.
    """
    synced_list: list[Path | None] = []
    if not list2:
        return [None] * len(list1)

    for path1 in list1:
        str_path1 = str(path1)
        best_match_path = None
        max_match_size = -1

        for path2 in list2:
            str_path2 = str(path2)
            matcher = SequenceMatcher(a=str_path1, b=str_path2, autojunk=False)
            match = matcher.find_longest_match()

            if match.size > max_match_size:
                max_match_size = match.size
                best_match_path = path2

        synced_list.append(best_match_path)

    return synced_list


def _worker_load_feature(file_path: Path, feature_name: str):
    """
    Worker function to read one feature column from a single CSV file.
    Returns the file_path and the data, or None if it fails.
    """
    if file_path is None:
        return None
    try:
        df = pd.read_csv(file_path, usecols=[feature_name])
        return (file_path, df[feature_name].values)
    except Exception as e:
        print(f"Error processing file {os.path.basename(str(file_path))} (feature: {feature_name}): {e}")
        return None


def load_feature_in_parallel(folder_path: str, feature_name: str, file_filter: list[str] | None = None, num_cores: int = 4) -> np.ndarray | None:
    """
    Loads a single feature from all matching CSV files in a folder in parallel.
    """
    print(f"--- Loading feature '{feature_name}' in parallel ---")
    all_files = sorted(glob.glob(os.path.join(folder_path, "*tsv_BIN.csv")))
    if not all_files:
        print(f"Warning: No '*tsv_BIN.csv' files found in {folder_path}")
        return None

    if file_filter:
        all_files = sync_file_lists(file_filter, [Path(f) for f in all_files])
        # Filter out None values that may result from sync_file_lists
        all_files = [f for f in all_files if f is not None]
        print(f"Filtered to {len(all_files)} files after applying the filter.")

    if not all_files:
        print(f"Warning: No files remained after filtering for feature '{feature_name}'.")
        return None

    results = Parallel(n_jobs=num_cores)(delayed(_worker_load_feature)(f, feature_name) for f in tqdm(all_files))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        print(f"Error: Could not successfully load the feature '{feature_name}' from any file.")
        return None

    final_array = np.stack(valid_results, axis=0)
    print(f"Successfully created array of shape: {final_array.shape}")
    return final_array


def normalize_feature_arrays(feature_dict: dict) -> dict:
    """
    Performs row-wise Z-score normalization on a dictionary of feature arrays.
    """
    normalized_dict = {}
    for feature, array in feature_dict.items():
        if array is None:
            normalized_dict[feature] = None
            continue
        row_means = np.nanmean(array, axis=1, keepdims=True)
        row_stds = np.nanstd(array, axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1
        normalized_array = (array - row_means) / row_stds
        normalized_dict[feature] = np.nan_to_num(normalized_array)
        print(f"Normalized feature '{feature}' with shape {array.shape} to shape {normalized_dict[feature].shape}")
    return normalized_dict


def format_distribution(labels: np.ndarray) -> str:
    """Formats the class distribution counts into a readable string."""
    unique, counts = np.unique(labels, return_counts=True)
    return ", ".join([f"{label}: {count}" for label, count in zip(unique, counts, strict=False)])


def select_top_genes(X_train: np.ndarray, y_train: np.ndarray, n_top_genes: int, selection_type="weighted") -> np.ndarray:
    """
    Selects the top genes based on a weighted score of log2 fold change and p-value.
    """
    cancer_samples = X_train[y_train == 1]
    control_samples = X_train[y_train == 0]

    p_values = []
    for gene_idx in range(X_train.shape[1]):
        cancer_gene_values = cancer_samples[:, gene_idx]
        control_gene_values = control_samples[:, gene_idx]
        if len(cancer_gene_values) == 0 or len(control_gene_values) == 0:
            p_values.append(1.0)
            continue
        _, p = mannwhitneyu(cancer_gene_values, control_gene_values, alternative="two-sided")
        p_values.append(p)
    p_values = np.array(p_values)

    mean_cancer = np.mean(cancer_samples, axis=0)
    mean_control = np.mean(control_samples, axis=0)

    # Calculate the log2 fold change
    def safe_log2_fold_change(mean_cancer, mean_control, eps=1e-10):
        with np.errstate(divide="ignore", invalid="ignore"):
            fold_change = np.divide(
                mean_cancer,
                mean_control,
                out=np.full_like(mean_cancer, np.nan),
                where=mean_control != 0,
            )

        log2_fold_change = np.log2(fold_change, out=np.full_like(fold_change, np.nan), where=fold_change > eps)

        return np.nan_to_num(log2_fold_change)

    log2_fold_change = safe_log2_fold_change(mean_cancer, mean_control)
    p_values[p_values == 0] = 1e-10

    if selection_type == "weighted":
        weighted_score = np.abs(log2_fold_change) * -np.log10(p_values)
        weighted_score = np.nan_to_num(weighted_score)
        num_genes_to_select = min(n_top_genes, len(weighted_score))

        # if n_top_genes < 25:
        #     plt.scatter(log2_fold_change, -np.log10(p_values), alpha=0.5)
        #     plt.xlabel('Log2 Fold Change')
        #     plt.ylabel('-Log10 p-value')
        #     plt.title('Volcano Plot of Gene Selection')
        #     plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        #     plt.axvline(1, color='blue', linestyle='--', label='Log2 FC=1')
        #     plt.axvline(-1, color='blue', linestyle='--')
        #     top_100_genes = np.argsort(weighted_score)[-100:]
        #     plt.scatter(log2_fold_change[top_100_genes], -np.log10(p_values[top_100_genes]), color='orange', label='Top 100 Weighted Score', s=50)
        #     top_100_log2fc = np.argsort(np.abs(log2_fold_change))[-100:]
        #     plt.scatter(log2_fold_change[top_100_log2fc], -np.log10(p_values[top_100_log2fc]), color='green', label='Top 100 Log2 FC', s=20)
        #     top_100_pvalues = np.argsort(p_values)[:100]
        #     plt.scatter(log2_fold_change[top_100_pvalues], -np.log10(p_values[top_100_pvalues]), color='purple', label='Top 100 p-values', s=5)

        #     plt.legend()
        #     plt.savefig(f'volcano_plot_{n_top_genes}_genes_model_hpc.png')
        #     plt.close()

        return np.argsort(weighted_score)[-num_genes_to_select:]
    if selection_type == "fold_change":
        abs_log2_fc = np.abs(log2_fold_change)
        abs_log2_fc = np.nan_to_num(abs_log2_fc)
        num_genes_to_select = min(n_top_genes, len(abs_log2_fc))
        return np.argsort(abs_log2_fc)[-num_genes_to_select:]
    if selection_type == "p_value":
        num_genes_to_select = min(n_top_genes, len(p_values))
        return np.argsort(p_values)[:num_genes_to_select]
    msg = f"Unknown selection_type: {selection_type}"
    raise ValueError(msg)


def run_single_experiment(params: dict):
    """
    Worker function to run the full nested CV pipeline for a single experiment configuration.
    An experiment is defined by a unique combination of classifier, feature, and top_genes count.
    """
    # Unpack parameters
    clf_name = params["classifier"]
    feature_name = params["feature"]
    n_top_genes = params["top_genes"]
    all_feature_arrays = params["feature_data"]
    y_labels = params["labels"]
    stratification_labels = params["stratification_labels"]
    sample_ids = np.array(params["sample_ids"])
    output_dir = params["output_dir"]
    selection_type = params["selection_type"]

    # Get classifier, its parameter grid, and the specific feature matrix
    classifier = params["classifiers"][clf_name]
    param_grid = params["clf_param_grids"][clf_name]
    X = all_feature_arrays[feature_name]

    # Define output file path
    clf_name_safe = clf_name.replace(" ", "_").lower()
    output_filename = f"results_{clf_name_safe}_{feature_name}_{n_top_genes}_genes_{selection_type}.pkl"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Skipping existing results file: {output_path}")
        return output_path

    print("\n======================================================================================")
    print(f"STARTING: {clf_name_safe} | Feature: '{feature_name}' | Top Genes: {n_top_genes}")
    print("======================================================================================")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, stratification_labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        stratification_train = stratification_labels[train_idx]
        stratification_test = stratification_labels[test_idx]
        sample_ids_test = sample_ids[test_idx]  # Get sample IDs for the test set

        # --- Print Outer Loop Distribution ---
        # print(f"\n[Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}:")
        # print(f"  Train distribution: {format_distribution(stratification_train)}")
        # print(f"  Test distribution:  {format_distribution(stratification_test)}")

        top_gene_indices = select_top_genes(X_train, y_train, n_top_genes, selection_type=selection_type)

        if len(top_gene_indices) == 0:
            print(f"  Fold {fold_idx + 1}/5: No genes selected. Skipping.")
            continue

        X_train_selected = X_train[:, top_gene_indices]
        X_test_selected = X_test[:, top_gene_indices]

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(X_train_selected, stratification_train))

        # --- Print Inner Loop Distributions ---
        # print("  [Inner CV Splits for GridSearchCV]:")
        # for inner_fold_idx, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
        #     strat_inner_train = stratification_train[inner_train_idx]
        #     strat_inner_test = stratification_train[inner_test_idx]
        # print(f"    - Inner Fold {inner_fold_idx + 1}/{inner_cv.get_n_splits()}:")
        # print(f"        Train ({len(strat_inner_train)} samples): {format_distribution(strat_inner_train)}")
        # print(f"        Test  ({len(strat_inner_test)} samples): {format_distribution(strat_inner_test)}")

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="roc_auc", cv=inner_splits, n_jobs=1)

        grid_search.fit(X_train_selected, y_train)
        best_model = grid_search.best_estimator_

        y_pred_proba_full = best_model.predict_proba(X_test_selected)[:, 1]
        y_pred_binary_full = best_model.predict(X_test_selected)

        # Calculate overall metrics
        fpr_full, tpr_full, _ = roc_curve(y_test, y_pred_proba_full)
        auc_full = auc(fpr_full, tpr_full)
        precision_full, recall_full, _ = precision_recall_curve(y_test, y_pred_proba_full)
        report_full = classification_report(y_test, y_pred_binary_full, output_dict=True, zero_division=0)
        confusion_matrix_full = confusion_matrix(y_test, y_pred_binary_full).tolist()

        # Identify misclassified samples for the entire fold
        misclassified_indices = np.where(y_test != y_pred_binary_full)[0]
        misclassified_sample_ids = sample_ids_test[misclassified_indices].tolist()

        # --- PER-CANCER-TYPE EVALUATION ---
        auc_per_type, roc_per_type, pr_per_type = {}, {}, {}
        unique_cancer_types_in_test = np.unique(stratification_test[stratification_test != "Healthy"])

        print(f"  [Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}: Overall AUC = {auc_full:.3f}. Evaluating on {len(unique_cancer_types_in_test)} cancer types.")

        for cancer_type in unique_cancer_types_in_test:
            mask = (stratification_test == "Healthy") | (stratification_test == cancer_type)
            y_test_subset = (stratification_test[mask] == cancer_type).astype(int)
            y_pred_proba_subset = y_pred_proba_full[mask]

            fpr, tpr, _ = roc_curve(y_test_subset, y_pred_proba_subset)
            precision, recall, _ = precision_recall_curve(y_test_subset, y_pred_proba_subset)

            auc_per_type[cancer_type] = auc(fpr, tpr)
            roc_per_type[cancer_type] = {"fpr": fpr, "tpr": tpr}
            pr_per_type[cancer_type] = {"precision": precision, "recall": recall}

        fold_results.append(
            {
                "feature": feature_name,
                "top_genes": n_top_genes,
                "fold": fold_idx + 1,
                "classifier": clf_name,
                "best_params": grid_search.best_params_,
                "num_genes_used": len(top_gene_indices),
                "misclassified_sample_ids": misclassified_sample_ids,
                "test_sample_ids": sample_ids_test.tolist(),
                # Overall model metrics
                "auc_full_model": auc_full,
                "fpr_full_model": fpr_full,
                "tpr_full_model": tpr_full,
                "precision_full_model": precision_full,
                "recall_full_model": recall_full,
                "classification_report_full_model": report_full,
                "confusion_matrix_full_model": confusion_matrix_full,
                # Per-cancer-type metrics
                "auc_per_cancer_type": auc_per_type,
                "roc_per_cancer_type": roc_per_type,
                "pr_per_cancer_type": pr_per_type,
            }
        )

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(fold_results)

    with open(output_path, "wb") as f:
        pickle.dump(results_df, f)

    print(f"--- Finished and saved to {output_path} in {elapsed:.2f}s ---")
    return output_path


# =================================================================================
# --- Main Execution Function ---
# =================================================================================


def main(args):
    """
    Main function to orchestrate data loading, preprocessing, and parallel model training.
    """
    # --- Define Constants ---
    CLASSIFIERS = {"Support Vector Machine": SVC(probability=True, random_state=42), "Random Forest": RandomForestClassifier(random_state=42), "K-Nearest Neighbors": KNeighborsClassifier(), "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000)}

    CLF_PARAM_GRIDS = {
        "Support Vector Machine": {"clf__C": np.logspace(-2, 2, 5), "clf__kernel": ["linear", "rbf"]},
        "Random Forest": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [2, 3, 4, 5, None], "clf__min_samples_leaf": [3, 5, 10], "clf__max_features": [1.0, "sqrt", "log2"]},
        "K-Nearest Neighbors": {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
        "Logistic Regression": {"clf__C": np.logspace(-3, 3, 7), "clf__penalty": ["elasticnet"], "clf__solver": ["saga"], "clf__l1_ratio": [0.25, 0.5, 0.75]},
    }

    DATA_PARAM_GRIDS = {"feature": ["fslr_coverage", "fslr", "griffin_diff", "rcov", "mwh"], "top_genes": [10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000]}

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and Preprocess Data ---
    print("--- Loading and preprocessing data ---")

    if "cris" in args.metadata_file.lower():
        metadata = pd.read_csv(args.metadata_file)
        healthy_files_df = metadata[metadata["disease"] == "Healthy"].iloc[50:]
        healthy_file_names = healthy_files_df["ID"].tolist()

        # --- Process cancer types argument ---
        selected_cancers = args.cancer_types
        if "PANCAN" in [c.upper() for c in selected_cancers]:
            cancer_types_to_filter = list(CANCER_MAP.values())
            print("PANCAN option selected. Including all available cancer types.")
        else:
            try:
                cancer_types_to_filter = [CANCER_MAP[c.lower()] for c in selected_cancers]
            except KeyError as e:
                print(f"Error: Invalid cancer type '{e.args[0]}'. Please choose from {list(CANCER_MAP.keys())} or PANCAN.")
                return

    elif "jiang" in args.metadata_file.lower():
        metadata = pd.read_csv(args.metadata_file)
        healthy_to_filter = ["Healthy", "Cirrhosis", "Hepatitis B"]
        cancer_types_to_filter = ["Liver cancer"]

    elif "lucas" in args.metadata_file.lower():
        metadata = pd.read_csv(args.metadata_file)
        healthy_to_filter = ["No baseline cancer", "no lung cancer, other cancer", "Benign", "no lung cancer, later lung cancer", "no lung cancer, other cancer, later lung cancer", "no lung cancer, prior lung cancer", "no lung cancer, prior lung cancer, later lung cancer"]
        cancer_types_to_filter = ["lung cancer", "lung cancer, other cancer", "met to lung, other cancer", "lung cancer, prior lung cancer", "met to lung", "lung cancer, later lung cancer"]

    elif "mathios" in args.metadata_file.lower():
        metadata = pd.read_csv(args.metadata_file)
        healthy_to_filter = ["healthy"]
        cancer_types_to_filter = ["cancer"]

    print(f"Filtering out healthy samples with conditions: {', '.join(healthy_to_filter)}")
    healthy_files_df = metadata[metadata["disease"].isin(healthy_to_filter)]
    healthy_file_names = healthy_files_df["ID"].tolist()
    print(f"Filtering for cancer types: {', '.join(cancer_types_to_filter)}")
    cancer_files_df = metadata[metadata["disease"].isin(cancer_types_to_filter)]
    cancer_file_names = cancer_files_df["ID"].tolist()
    print(f"Found {len(cancer_file_names)} cancer samples.")
    print(f"Found {len(healthy_file_names)} healthy samples.")

    # Create a list of sample IDs so that after classification we can determine which samples were used
    combined_df = pd.concat([healthy_files_df, cancer_files_df], ignore_index=True)
    sample_ids = combined_df["ID"].tolist()

    tss_features = DATA_PARAM_GRIDS["feature"]
    feature_arrays_healthy = {}
    feature_arrays_cancer = {}

    for feature_to_load in tss_features:
        feature_arrays_healthy[feature_to_load] = load_feature_in_parallel(args.features_dir, feature_to_load, file_filter=healthy_file_names, num_cores=args.cores)
        feature_arrays_cancer[feature_to_load] = load_feature_in_parallel(args.features_dir, feature_to_load, file_filter=cancer_file_names, num_cores=args.cores)

    # Remove blacklisted regions
    to_be_removed = np.load(args.blacklist_indices)
    for feature in tss_features:
        if feature_arrays_healthy[feature] is not None:
            feature_arrays_healthy[feature] = np.delete(feature_arrays_healthy[feature], to_be_removed, axis=1)
        if feature_arrays_cancer[feature] is not None:
            feature_arrays_cancer[feature] = np.delete(feature_arrays_cancer[feature], to_be_removed, axis=1)

    print("\nRemoved blacklisted regions from all feature arrays.")

    # Normalize data
    feature_arrays_healthy_normalized = normalize_feature_arrays(feature_arrays_healthy)
    feature_arrays_cancer_normalized = normalize_feature_arrays(feature_arrays_cancer)
    print("\nAll features normalized.")

    # Combine healthy and cancer data
    all_feature_arrays = {}
    for feature in tss_features:
        healthy_data = feature_arrays_healthy_normalized.get(feature)
        cancer_data = feature_arrays_cancer_normalized.get(feature)
        if healthy_data is not None and cancer_data is not None:
            all_feature_arrays[feature] = np.vstack([healthy_data, cancer_data])

    # Per feature, check for duplicate columns and remove them
    unique_feature_arrays = {}
    for feature, array in all_feature_arrays.items():
        if array is None:
            continue
        _, unique_indices = np.unique(array, axis=1, return_index=True)
        unique_array = array[:, np.sort(unique_indices)]
        unique_feature_arrays[feature] = unique_array
        print(f"Feature '{feature}': Reduced from {array.shape[1]} to {unique_array.shape[1]} unique columns.")

    # Replace the old dictionary and feature list with the unique ones
    all_feature_arrays = unique_feature_arrays
    DATA_PARAM_GRIDS["feature"] = list(all_feature_arrays.keys())
    print(f"Proceeding with {len(all_feature_arrays)} unique features.")

    n_healthy = feature_arrays_healthy_normalized[tss_features[0]].shape[0] if feature_arrays_healthy_normalized.get(tss_features[0]) is not None else 0
    n_cancer = feature_arrays_cancer_normalized[tss_features[0]].shape[0] if feature_arrays_cancer_normalized.get(tss_features[0]) is not None else 0
    y_labels = np.array([0] * n_healthy + [1] * n_cancer)

    print(f"\nTotal samples: {len(y_labels)} ({n_healthy} healthy, {n_cancer} cancer)")

    # Create detailed labels for stratification
    healthy_strat_labels = healthy_files_df["disease"].tolist()
    cancer_strat_labels = cancer_files_df["disease"].tolist()
    stratification_labels = np.array(healthy_strat_labels + cancer_strat_labels)
    print(f"Stratification will be based on {len(np.unique(stratification_labels))} unique classes.")

    print("--- Data preparation complete ---")

    # --- Create All Experiment Combinations ---
    print("--- Generating experiment configurations ---")
    classifier_names = list(CLASSIFIERS.keys())
    feature_names = DATA_PARAM_GRIDS["feature"]
    top_genes_list = DATA_PARAM_GRIDS["top_genes"]

    # Create the Cartesian product of all parameters
    param_combinations = list(itertools.product(classifier_names, feature_names, top_genes_list))

    experiment_configs = [{"classifier": clf, "feature": feat, "top_genes": n_genes, "feature_data": all_feature_arrays, "labels": y_labels, "stratification_labels": stratification_labels, "sample_ids": sample_ids, "output_dir": args.output_dir, "selection_type": args.selection_type, "classifiers": CLASSIFIERS, "clf_param_grids": CLF_PARAM_GRIDS} for clf, feat, n_genes in param_combinations]

    print(f"--- Starting {len(experiment_configs)} experiments on {args.cores} cores ---")
    Parallel(n_jobs=args.cores, verbose=10)(delayed(run_single_experiment)(config) for config in experiment_configs)

    print("\n--- All experiments completed successfully ---")


# =================================================================================
# --- Command-Line Interface & Main Entry Point ---
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nested cross-validation for genomic classifiers on an HPC.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--cores", type=int, default=64, help="Number of CPU cores to use for parallel execution.")
    parser.add_argument("--features_dir", type=str, default="./extracted_features/biomart_10kb", help="Path to the directory containing the extracted feature CSV/TSV files.")
    parser.add_argument("--metadata_file", type=str, default="accessory_files/cris_metadata.csv", help="Path to the sample metadata CSV file.")
    parser.add_argument("--blacklist_indices", type=str, default="accessory_files/blacklist_and_std_outlier_and_chrX_indices.npy", help="Path to the .npy file with indices of regions to be removed.")
    parser.add_argument("-o", "--output_dir", type=str, default="nested_cv_results", help="Directory to save the output .pkl files.")
    parser.add_argument("-s", "--selection_type", type=str, choices=["p_value", "fold_change", "weighted"], default="weighted", help="Feature selection method to use.")

    CANCER_MAP = {"breast": "Breast cancer", "bileduct": "Bile duct cancer", "colorectal": "Colorectal cancer", "gastric": "Gastric cancer", "lung": "Lung cancer naive nsclc", "ovarian": "Ovarian cancer", "pancreatic": "Pancreatic cancer"}

    parser.add_argument(
        "-t",
        "--cancer_types",
        nargs="+",
        # default=['breast', 'colorectal'],
        default=["PANCAN"],
        help=f"List of cancer types. Use short names (e.g., breast, colorectal). Provide 'PANCAN' for all types. Available: {', '.join(CANCER_MAP.keys())}",
    )

    args = parser.parse_args()

    # Use the number of cores specified, but not more than available
    args.cores = min(args.cores, multiprocessing.cpu_count())

    main(args)

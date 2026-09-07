"""Nested cross-validation for classification of cancer types based on fragmentomic features calculated per TSS."""
import os
import glob
import pickle
import itertools
from itertools import product
from joblib import Parallel, delayed
import pickle
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from time import time
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

# Show more all rows in a DataFrame
pd.set_option("display.max_rows", 25)

CLASSIFIERS = {
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42)
}

CLF_PARAM_GRIDS = {
    "Support Vector Machine": {
        "C": np.logspace(0.01, 2, 10),
        "kernel": ["linear", "rbf"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    # "K-Nearest Neighbors": {
    #     "n_neighbors": [3, 5, 7, 9],
    #     "weights": ["uniform", "distance"],
    #     "p": [1, 2],  # p=1 for Manhattan distance, p=2 for Euclidean distance
    # },
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.001, 0.25, 0.5, 0.75, 0.999],
        'max_iter': [10000],
    }
}


def centerscale(x, xtest=None):
    """Manual implementation of StandardScaler.fit_transform

    Args:
        x (np.array): Training data (samples x features).
        xtest (np.array, optional): Test data. Defaults to None.
    """
    xtransformed = np.zeros(x.shape)
    if xtest is not None:
        xtestTransformed = np.zeros(xtest.shape)

    for feature in range(x.shape[1]):
        mean = np.mean(x[:, feature])
        std = np.std(x[:, feature], ddof=1)

        if std == 0:
            std = 1

        xtransformed[:, feature] = (x[:, feature] - mean) / std

        if xtest is not None:
            xtestTransformed[:, feature] = (xtest[:, feature] - mean) / std

    if xtest is not None:
        return xtransformed, xtestTransformed

    return xtransformed


def jaccard_index(set1, set2):
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union != 0 else 0

def jaccard_index_average(gene_sets: List[np.ndarray]) -> float:
    """
    Computes the average Jaccard Index for multiple sets.
    JI = (JI_1 + JI_2 + ... + JI_n) / n

    Parameters:
    - gene_sets: List of NumPy arrays, each containing gene names (dtype=str)

    Returns:
    - Average Jaccard index as a float
    """
    if len(gene_sets) < 2:
        raise ValueError("Need at least two sets to compute Jaccard Index.")
    
    jaccard_indices = []
    
    # Compute pairwise Jaccard indices
    for i in range(len(gene_sets)):
        for j in range(i + 1, len(gene_sets)):
            jaccard_indices.append(jaccard_index(gene_sets[i], gene_sets[j]))
    
    return np.mean(jaccard_indices)

def generalized_jaccard_index(gene_sets: List[np.ndarray]) -> float:
    """
    Computes the generalized Jaccard Index for multiple sets.
    JI = |Intersection of all sets| / |Union of all sets|

    Parameters:
    - gene_sets: List of NumPy arrays, each containing gene names (dtype=str)

    Returns:
    - Jaccard index as a float
    """
    if len(gene_sets) < 2:
        raise ValueError("Need at least two sets to compute Jaccard Index.")
    
    # Convert to sets
    sets = [set(arr) for arr in gene_sets]

    # Compute intersection and union
    intersection = set.intersection(*sets)
    union = set.union(*sets)

    # Handle empty union case
    if not union:
        return 1.0 if not intersection else 0.0  # both empty → JI = 1, else undefined

    return len(intersection) / len(union)

def get_downsampled_features(down_sampled_X, down_sampled_X_test, feature, gene_indices):
    """
    Get the downsampled features based on the gene indices."""
    down_sampled_X_feature = np.array(
        [
            down_sampled_X[i][feature].values[gene_indices]
            for i in range(len(down_sampled_X))
        ]
    )
    down_sampled_X_feature_test = np.array(
        [
            down_sampled_X_test[i][feature].values[gene_indices]
            for i in range(len(down_sampled_X_test))
        ]
    )

    down_sampled_X_feature, down_sampled_X_feature_test = centerscale(down_sampled_X_feature, down_sampled_X_feature_test)

    return down_sampled_X_feature, down_sampled_X_feature_test


def filter_genes(hyper_params, bed, X_fold, y_fold, X_fold_test=None, random=False):
    """Filter the data based on the hyperparameters, select the top genes based on log2 fold change.

    Args:
        hyper_params (dict): Dictionary containing the hyperparameters.
        bed (pd.DataFrame): Dataframe containing the gene information.
        X_fold (List): List of dataframes containing the features.
        y_fold (List): List of labels.
        X_fold_test (List, optional): List of dataframes containing the features for the test set. Defaults to None.
        random (bool, optional): Whether to use random TSS selection. Defaults to False.

    Returns:
        np.array: Feature matrix for the training set.
        np.array: Indices of the genes with the highest log2 fold change.
        np.array: Feature matrix for the test set. (if X_fold_test is not None)
    """
    X = pd.concat(X_fold)  # X_fold should be the train set
    X = X.groupby(X.index).mean()  # Collapse the data to get average values for each gene
    X["gene"] = bed["gene"]  # Add the gene column back to the dataframe to get names of the genes

    for param in hyper_params.keys():
        if param in ["top_genes", "feature"]:
            continue
        # Filter the genes based on a hyperparameter threshold, note that the values are averaged over the samples for each gene
        X = X[X[param] > hyper_params[param]]  

    if X.shape[0] == 0:
        print("No genes left after filtering")
        return (None, None, None, None) if X_fold_test is not None else (None, None, None)

    # Now that we know which genes qualify, we can filter the each sample to only keep selected genes
    genes_idx = X.index
    X_feature = np.array(
        [
            X_fold[i][hyper_params["feature"]].values[genes_idx]
            for i in range(len(X_fold))
        ]
    )
    if X_fold_test is not None:
        X_feature_test = np.array(
            [
                X_fold_test[i][hyper_params["feature"]].values[genes_idx]
                for i in range(len(X_fold_test))
            ]
        )

    if random:
        sample_size = hyper_params["top_genes"]
        if sample_size > X_feature.shape[1]:
            sample_size = X_feature.shape[1]

        # Randomly select TSS's from the filtered genes
        selected_indices = np.random.choice(
            X_feature.shape[1], hyper_params["top_genes"], replace=False
        )        

    else:
        # Now we select TSS's based on log2 fold change and Mann-Whitney U test
        cancer_samples = X_feature[y_fold == 1]
        control_samples = X_feature[y_fold == 0]

        # Statistical testing
        p_values = []
        for gene_idx in range(cancer_samples.shape[1]):
            cancer_gene_values = cancer_samples[:, gene_idx]
            control_gene_values = control_samples[:, gene_idx]

            # if hyper_params.get("test_method", "ttest") == "mannwhitneyu":
            _, p = mannwhitneyu(cancer_gene_values, control_gene_values, alternative="two-sided")
            # else:  # Default to Welch's t-test
            # _, p = ttest_ind(cancer_gene_values, control_gene_values, equal_var=False, nan_policy="omit")

            p_values.append(p)

        p_values = np.array(p_values)

        # Perform FDR correction
        # _, fdr_corrected_p_values = fdrcorrection(p_values, alpha=0.05)

        # Calculate the log2 fold change
        # Custom log2 function with safe division handling
        def safe_log2_fold_change(mean_cancer, mean_control, eps=1e-10):
            with np.errstate(divide="ignore", invalid="ignore"):
                fold_change = np.divide(
                    mean_cancer,
                    mean_control,
                    out=np.full_like(mean_cancer, np.nan),
                    where=mean_control != 0,
                )

            # Apply log2 safely to valid fold change values
            log2_fold_change = np.log2(
                fold_change, out=np.full_like(fold_change, np.nan), where=fold_change > eps
            )

            # Set nan to 0, which means that 0 values in mean_control do not contribute to the log2 fold change
            log2_fold_change = np.nan_to_num(log2_fold_change)

            return log2_fold_change

        # TODO: This is a very simple way to calculate the log2 fold change. We should consider more advanced methods.
        # Especially since we are averaging the fold change over all samples, refer to the scratch repo for more data visualization.
        # With the variety of 0's, features with different distributions, etc. we should consider more advanced methods.

        # Calculate log2 fold change for each gene between cancer and control samples (average fold change)
        mean_cancer = np.mean(cancer_samples, axis=0)
        mean_control = np.mean(control_samples, axis=0)
        log2_fold_change = safe_log2_fold_change(mean_cancer, mean_control)

        ## @TODO: Should we sort by absolute value of log2 fold change or multiply by p-values?
        # Weigh the log2 fold change by the FDR-corrected p-values
        weighted_log2_fold_change = log2_fold_change * -np.log10(p_values)
        sort_indices = np.argsort(np.abs(weighted_log2_fold_change))[::-1]
        selected_indices = sort_indices[: hyper_params["top_genes"]]

        # significant_genes = np.where(fdr_corrected_p_values < 0.05)[0]

        # if len(significant_genes) == 0:
        #     print("No significant genes found")
        #     return (None, None, None, None) if X_fold_test is not None else (None, None, None)

        # Sort the significant genes by log2 fold change

        # # Plot volcano plots
        # plot_volcano(log2_fold_change, p_values, "Volcano Plot: Raw P-values")
        # top_5_gene_idx_unsig = plot_volcano(log2_fold_change, p_values, "Volcano Plot: Unsignificant FDR-Corrected P-values", highest_log_fold_indices_unsig)
        # top_5_gene_idx = plot_volcano(log2_fold_change, fdr_corrected_p_values, "Volcano Plot: FDR-Corrected P-values", highest_log_fold_indices)

        # # Plot the distributions of the top 5 genes with the highest log2 fold change
        # plot_gene_distributions(top_5_gene_idx, cancer_samples, control_samples, X, hyper_params, title="sig")
        # plot_gene_distributions(top_5_gene_idx_unsig, cancer_samples, control_samples, X, hyper_params, title="unsignif")

        # Plot the number of zeroes in the genes
        # plot_zeroes(X_feature, hyper_params, highest_log_fold_indices)

        # sort_indices = np.argsort(np.abs(log2_fold_change))[::-1]
        # highest_log_fold_indices = sort_indices[: hyper_params["top_genes"]]

    if X_fold_test is not None:
        return X_feature, selected_indices, X_feature_test, X['gene']

    return X_feature, selected_indices, None, X['gene']

def plot_zeroes(X_feature, hyper_params, highest_log_fold_indices):
    # For every gene, plot the number of samples with a value of 0
    zeroes_top_genes = []
    zeroes_other_genes = []
    for gene_idx in range(X_feature.shape[1]):
        zeroes = np.sum(X_feature[:, gene_idx] == 0)
        if gene_idx in highest_log_fold_indices:
            zeroes_top_genes.append(zeroes)
        else:
            zeroes_other_genes.append(zeroes)

    plt.figure(figsize=(10, 6))
    print(len(zeroes_top_genes), len(zeroes_other_genes))
    hist, bins, _ = plt.hist(zeroes_top_genes, bins=50, alpha=0.5, label="Top genes", color="red")
    print("Top genes histogram values: ", hist)
    hist, bins, _ = plt.hist(zeroes_other_genes, bins=50, alpha=0.5, label="Other genes", color="blue")
    print("Other genes histogram values: ", hist)

    plt.title("Number of samples with a value of 0 for each gene")
    plt.xlabel("Number of samples with a value of 0")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f'notebooks/nested_cv_testdir/zeroes_{hyper_params["feature"]}_{hyper_params["top_genes"]}.png')

# Volcano Plot Function
def plot_volcano(hyper_params, log2_fold_change, p_values, title, highest_log_fold_indices=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_change, -np.log10(p_values), alpha=0.5, edgecolor="k", linewidth=0.1)
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
    plt.axvline(-1, color='blue', linestyle='--', label="Log2FC threshold (-1)")
    plt.axvline(1, color='blue', linestyle='--', label="Log2FC threshold (1)")
    plt.title(title)
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10(p-value)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    # plt.show(block=True)

    # Add the names of the top 5 genes with the highest log2 fold change
    top_5_gene_idx = []
    if highest_log_fold_indices is not None:
        for i, gene_idx in enumerate(highest_log_fold_indices[:5]):
            plt.text(log2_fold_change[gene_idx], -np.log10(p_values[gene_idx]), X["gene"].iloc[gene_idx], fontsize=8)
            top_5_gene_idx.append(gene_idx)


    plt.savefig(f'notebooks/nested_cv_testdir/vc_{hyper_params["feature"]}_{hyper_params["top_genes"]}_{title}.png')
    return top_5_gene_idx


# Assuming top_5_gene_idx, cancer_samples, control_samples, and X are already defined
def plot_gene_distributions(top_5_gene_idx, cancer_samples, control_samples, X, hyper_params, title=""):
    """Create a single plot with 5 subplots for gene distributions.

    Args:
        top_5_gene_idx (list): List of indices for the top 5 genes.
        cancer_samples (np.array): Cancer sample feature values.
        control_samples (np.array): Control sample feature values.
        X (pd.DataFrame): DataFrame containing gene information.
        hyper_params (dict): Dictionary containing hyperparameters.

    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    for i, gene_idx in enumerate(top_5_gene_idx):
        ax = axes[i]
        ax.hist(cancer_samples[:, gene_idx], bins=50, alpha=0.5, label="Cancer", color="red")
        ax.hist(control_samples[:, gene_idx], bins=50, alpha=0.5, label="Control", color="blue")
        ax.set_title(f"{X['gene'].iloc[gene_idx]}")
        ax.set_xlabel("Feature value")
        if i == 0:
            ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'notebooks/nested_cv_testdir/{title}_dist_{hyper_params["feature"]}_{hyper_params["top_genes"]}_subplots.png')
    plt.close()


def prepare_data_for_clf(hyper_params, bed, X_fold, y_fold, X_fold_test, return_genes=False, random=False):
    """Prepare the data for classification by filtering genes and standardizing the data.

    Args:
        hyper_params (dict): Dictionary containing the hyperparameters (a combination from config data_params).
        X_fold (List): List of dataframes containing the features (from the training set).
        y_fold (List): List of labels.
        X_fold_test (List): List of dataframes containing the features for the test set.
        return_genes (bool, optional): Whether to return the gene names. Defaults to False.
        random (bool, optional): Whether to use random TSS selection. Defaults to False.

    Returns:
        np.array: Feature matrix for the training set.
        np.array: Feature matrix for the test set.
    """
    X_feature, selected_indices, X_feature_test, genes = filter_genes(
        hyper_params, bed, X_fold, y_fold, X_fold_test=X_fold_test, random=random
    )

    if X_feature is None:
        return None, None

    # We've filtered out genes, now create the feature matrix based on those genes
    X_feature = X_feature[:, selected_indices]
    X_feature_test = X_feature_test[:, selected_indices]

    # Standardize the data
    X_feature, X_feature_test = centerscale(X_feature, X_feature_test)

    if return_genes:
        return X_feature, X_feature_test, selected_indices

    return X_feature, X_feature_test

def model_cv(args: dict, bed, X, y, X_external, y_external, custom_data_params) -> dict:

    def process_data_param_combo(data_param_combo):
        cv = StratifiedKFold(n_splits=args['outer_folds'], shuffle=True, random_state=42)
        start_time = time()

        data_param_combo_results = {
            "data_param_combo": data_param_combo,
            "folds": [],
        }

        # Setup clf dict
        clf_results = {}
        for clf_name in args['classifiers']:
            clf_results[clf_name] = {}
            clf_param_grid = CLF_PARAM_GRIDS[clf_name]
            for clf_params in product(*clf_param_grid.values()):
                clf_results[clf_name][clf_params] = {
                    'roc_auc': [],
                    'tss_used': [],
                    'fprs': [],
                    'tprs': [],
                    'classification_report': [],
                    'confusion_matrix': [],
                    'test_num_0': [],
                    'test_num_1': []
                }

        gene_sets = []

        # Cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Prepare the data
            X_train, X_test, gene_indices = prepare_data_for_clf(data_param_combo, bed, X_train, y_train, X_test, return_genes=True, random=args['random'])
            if X_train is None:
                print("No genes left after filtering")
            else:
                gene_sets.append(bed["gene"].values[gene_indices].astype(str))

            for clf_name in args['classifiers']:
                clf_param_grid = CLF_PARAM_GRIDS[clf_name]
                for clf_params in product(*clf_param_grid.values()):
                    if X_train is None:
                        clf_results[clf_name][clf_params]['roc_auc'].append(0)
                        clf_results[clf_name][clf_params]['tss_used'].append(0)
                        clf_results[clf_name][clf_params]['fprs'].append([])
                        clf_results[clf_name][clf_params]['tprs'].append([])
                        clf_results[clf_name][clf_params]['classification_report'].append([])
                        clf_results[clf_name][clf_params]['confusion_matrix'].append([])
                    else:
                        clf = CLASSIFIERS[clf_name]
                        clf.set_params(**dict(zip(clf_param_grid.keys(), clf_params)))
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred)
                        auc_score = roc_auc_score(y_test, y_pred)

                        # Calculate the AIC
                        if clf_name == "Logistic Regression":
                            # Estimate number of parameters: non-zero coefficients + intercept
                            num_params = np.count_nonzero(clf.coef_) + int(clf.fit_intercept)
                            # Compute log loss
                            ll = log_loss(y_test, y_pred, normalize=False)
                            # AIC approximation
                            aic_value = 2 * num_params + 2 * ll
                            
                            clf_results[clf_name][clf_params].setdefault('aic', []).append(aic_value)

                        clf_results[clf_name][clf_params]['test_num_0'].append(np.sum(y_test == 0))
                        clf_results[clf_name][clf_params]['test_num_1'].append(np.sum(y_test == 1))
                        clf_results[clf_name][clf_params]['roc_auc'].append(auc_score)
                        clf_results[clf_name][clf_params]['fprs'].append(fpr)
                        clf_results[clf_name][clf_params]['tprs'].append(tpr)
                        clf_results[clf_name][clf_params]['tss_used'].append(len(gene_indices))

                        y_pred_binary = clf.predict(X_test)
                        clf_results[clf_name][clf_params]['classification_report'].append(classification_report(y_test, y_pred_binary, output_dict=True, zero_division=0))
                        clf_results[clf_name][clf_params]['confusion_matrix'].append(confusion_matrix(y_test, y_pred_binary))

                        # Validate on external data
                        if X_external is not None and y_external is not None:
                            X_external_feature_matrix = np.array(
                                [
                                    X_external[i][data_param_combo['feature']].values[gene_indices]
                                    for i in range(len(X_external))
                                ]
                            )
                            X_external_feature_matrix = centerscale(X_external_feature_matrix, None)

                            y_external_pred = clf.predict_proba(X_external_feature_matrix)[:, 1]
                            fpr_ext, tpr_ext, _ = roc_curve(y_external, y_external_pred)
                            auc_score_ext = auc(fpr_ext, tpr_ext)

                            clf_results[clf_name][clf_params]['external_roc_auc'] = auc_score_ext
                            clf_results[clf_name][clf_params]['external_fpr'] = fpr_ext
                            clf_results[clf_name][clf_params]['external_tpr'] = tpr_ext

                            y_external_pred_binary = clf.predict(X_external_feature_matrix)
                            clf_results[clf_name][clf_params]['external_classification_report'] = classification_report(y_external, y_external_pred_binary, output_dict=True, zero_division=0)
                            clf_results[clf_name][clf_params]['external_confusion_matrix'] = confusion_matrix(y_external, y_external_pred_binary)


            # Create the dataframe for the current fold
            fold_df = pd.DataFrame()
            data_param_combo_str = f"{data_param_combo['feature']}_{data_param_combo['top_genes']}_{data_param_combo['cov_x']}_{data_param_combo['cov_x_tss']}"

            for clf_name in args['classifiers']:

                for params in clf_results[clf_name].keys():
                    fold_dict = {
                        "data_param_combo": data_param_combo_str,
                        "fold_idx": fold_idx,
                        "clf_name": clf_name,
                        "clf_params": params,
                        "gene_sets": gene_sets,
                    }
                    if len(gene_sets) > 1:
                        fold_dict["inner_gene_sets_gji"] = generalized_jaccard_index(gene_sets)
                        fold_dict["inner_gene_sets_avgji"] = jaccard_index_average(gene_sets)

                    for key, value in clf_results[clf_name][params].items():
                        fold_dict[key] = value

                    fold_df = pd.concat([fold_df, pd.DataFrame([fold_dict])])
            
            data_param_combo_results["folds"].append(fold_df)

        # Concatenate all folds into a single dataframe
        data_param_combo_results["folds"] = pd.concat(data_param_combo_results["folds"], ignore_index=True)
        # Save the results to a file
        data_param_combo_results["folds"].to_csv(f"{args['output_dir']}/best_model_{data_param_combo['feature']}_{data_param_combo['top_genes']}_{data_param_combo['cov_x']}_{data_param_combo['cov_x_tss']}.csv", index=False)

        print(f"Finished processing data_param_combo: {data_param_combo} in {time() - start_time:.2f} seconds")
        return data_param_combo_results
   
    # Prepare the list of data parameter combinations
    data_param_combinations = [
        dict(zip(custom_data_params, v, strict=False))
        for v in itertools.product(*custom_data_params.values())
    ]
    print("The number of different data param combinations:", len(data_param_combinations))
    print("Starting Best Model Single CV with ", args['processes'], "processes")

    # Use joblib to parallelize the loop over data_param_combinations
    results = Parallel(n_jobs=args['processes'])(
        delayed(process_data_param_combo)(data_param_combo) for data_param_combo in data_param_combinations
    )

    return results

def best_model(results: dict, bed: pd.DataFrame, X: list, y: np.array) -> None:
    """Extract the best model and gene list from the results of the model_cv function."""
    highest_auc = -np.inf
    best_params = None
    for filter_key, clf_results in results.items():
        for clf_name, clf_params in clf_results.items():
            for params, res in clf_params.items():               
                if res['roc_auc'] > highest_auc:
                    highest_auc = res['roc_auc']
                    best_params = (filter_key, clf_name, params)

    best_params = {'feature': best_params[0][0], 'top_genes': best_params[0][1], 'cov_x': best_params[0][2], 'cov_x_tss': best_params[0][3]}
    print("Best parameters:", best_params)
    print("Highest AUC:", highest_auc)

    # Use the best params to filter the data and return the genes
    _, highest_log_fold_indices, _, genes = filter_genes(best_params, bed, X, y, X_fold_test=None, random=False)
    print("Number of genes:", len(highest_log_fold_indices))
    print("Total number of genes:", len(genes))
    final_genes = genes.values[highest_log_fold_indices].astype(str)

    return final_genes

def nested_cv_report(results: dict) -> None:
    """Create a report of the nested cross-validation results.
    Args:
        results (dict): Dictionary containing the results of the nested cross-validation.

    Returns:
        pd.DataFrame: DataFrame containing the results of the nested cross-validation.
    """
    
    data_param_df = pd.DataFrame()

    for fold_idx in range(len(results['folds'])):
        # Create a dataframe with the results
        results_dict = {}
        results_dict['fold'] = fold_idx

        fold_keys = list(results['folds'][fold_idx].keys())
        for key in fold_keys:
            results_dict[key] = results['folds'][fold_idx][key]

        data_param_df = pd.concat([data_param_df, pd.DataFrame([results_dict])])

    return data_param_df

def nested_cv(args, X, y, bed, down_sampled_X, X_external, y_external, custom_data_params, n_jobs=3):
    """Perform nested cross-validation to evaluate the generalization error of the model.

    Args:
        X (list): List of feature matrices for each sample.
        y (np.array): Array of labels.
        bed (pd.DataFrame): Dataframe containing gene information.
        output_file (str): Path to save the nested CV results.
        n_jobs (int): Number of CPU cores to use for parallel processing.

    Returns:
        dict: Results dictionary containing configurations, FPR, TPR, and ROC AUCs for all parameter combinations.
    """

    def process_data_param_combo(data_param_combo):
        outer_cv = StratifiedKFold(n_splits=args['outer_folds'], shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=args['inner_folds'], shuffle=True, random_state=42)

        data_param_combo_results = {
            "data_param_combo": data_param_combo,
            "folds": []
        }

        outer_cv_time_start = time()
        outer_gene_sets = []
        print(f"Processing data_param_combo: {data_param_combo}")
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            # print(f"\nProcessing outer fold {fold_idx + 1}/3...")
            X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Setup clf dict
            clf_inner_results = {}
            for clf_name in args['classifiers']:
                clf_inner_results[clf_name] = {}
                clf_param_grid = CLF_PARAM_GRIDS[clf_name]
                for clf_params in product(*clf_param_grid.values()):
                    clf_inner_results[clf_name][clf_params] = {
                        'roc_auc': [],
                        'tss_used': [],
                        'fprs': [],
                        'tprs': [],
                        'classification_report': [],
                        'confusion_matrix': [],
                        'test_num_0': [],
                        'test_num_1': []
                    }

            inner_gene_sets = []

            # Inner loop for hyperparameter tuning
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train)):
                X_inner_train = [X_train[i] for i in inner_train_idx]
                X_inner_val = [X_train[i] for i in inner_val_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]

                X_inner_train, X_inner_val, gene_indices = prepare_data_for_clf(
                    data_param_combo, bed, X_inner_train, y_inner_train, X_inner_val, return_genes=True, random=args['random']
                )
                if X_inner_train is None:
                    print("Skipped combination: no genes left after filtering")

                if X_inner_train is not None:
                    inner_gene_sets.append(bed["gene"].values[gene_indices].astype(str))

                for clf_name in args['classifiers']:
                    clf_param_grid = CLF_PARAM_GRIDS[clf_name]
                    for clf_params in product(*clf_param_grid.values()):
                        if X_inner_train is None:
                            clf_inner_results[clf_name][clf_params]['roc_auc'].append(0)
                            clf_inner_results[clf_name][clf_params]['tss_used'].append(0)
                            clf_inner_results[clf_name][clf_params]['fprs'].append([])
                            clf_inner_results[clf_name][clf_params]['tprs'].append([])
                            clf_inner_results[clf_name][clf_params]['classification_report'].append([])
                            clf_inner_results[clf_name][clf_params]['confusion_matrix'].append([])

                        else:
                            # print(f"Runnning inner loop {fold_idx}/{inner_fold_idx} for {clf_name} with params: {clf_params}")
                            clf = CLASSIFIERS[clf_name]
                            clf.set_params(**dict(zip(clf_param_grid.keys(), clf_params)))
                            clf.fit(X_inner_train, y_inner_train)
                            y_pred = clf.predict_proba(X_inner_val)[:, 1]
                            fpr, tpr, _ = roc_curve(y_inner_val, y_pred)
                            auc_score = roc_auc_score(y_inner_val, y_pred)

                            clf_inner_results[clf_name][clf_params]['test_num_0'].append(np.sum(y_inner_val == 0))
                            clf_inner_results[clf_name][clf_params]['test_num_1'].append(np.sum(y_inner_val == 1))
                            clf_inner_results[clf_name][clf_params]['roc_auc'].append(auc_score)
                            clf_inner_results[clf_name][clf_params]['tss_used'].append(X_inner_train.shape[1])
                            clf_inner_results[clf_name][clf_params]['fprs'].append(fpr)
                            clf_inner_results[clf_name][clf_params]['tprs'].append(tpr)

                            # Also extract other metrics like precision, recall, f1-score, etc.
                            y_pred_binary = clf.predict(X_inner_val)
                            clf_inner_results[clf_name][clf_params]['classification_report'].append(classification_report(y_inner_val, y_pred_binary, output_dict=True, zero_division=0))
                            clf_inner_results[clf_name][clf_params]['confusion_matrix'].append(confusion_matrix(y_inner_val, y_pred_binary))


            # print(f"Finished inner cross-validation for outer fold {fold_idx + 1}/3")

            # Find the best hyperparameters
            highest_inner_auc = -np.inf
            best_inner_clf_name = None
            best_inner_clf_params = None
            for clf_name, clf_params in clf_inner_results.items():
                for params, res in clf_params.items():
                    mean_auc = np.mean(res['roc_auc'])

                    if mean_auc > highest_inner_auc:
                        highest_inner_auc = mean_auc
                        best_inner_clf_name = clf_name
                        best_inner_clf_params = params

            fold_result = {
                "outer_fold_idx": fold_idx,
                "best_inner_clf_name": best_inner_clf_name,
                "best_inner_clf_params": best_inner_clf_params,
                "best_inner_clf_test_num_0": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['test_num_0'],
                "best_inner_clf_test_num_1": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['test_num_1'],
                "best_inner_auc": highest_inner_auc,
                "best_inner_clf": clf_inner_results[best_inner_clf_name],
                "best_inner_clf_roc_auc": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['roc_auc'],
                "best_inner_clf_tss_used": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['tss_used'],
                "best_inner_clf_fprs": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['fprs'],
                "best_inner_clf_tprs": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['tprs'],
                "best_inner_clf_classification_report": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['classification_report'],
                "best_inner_clf_confusion_matrix": clf_inner_results[best_inner_clf_name][best_inner_clf_params]['confusion_matrix'],
                "inner_gene_sets_gji": generalized_jaccard_index(inner_gene_sets),
                "inner_gene_sets_avgji": jaccard_index_average(inner_gene_sets),
                "inner_gene_sets": inner_gene_sets,
            }

            # print(f"Final Best AUC in inner loop: {highest_inner_auc:.4f} with clf {best_inner_clf_name} and params {best_inner_clf_params}")

            # For logging purposes, we are evaluating every data parom combo on outer fold, but always using the best clf
            outer_fold_X_train, outer_fold_X_test, selected_indices = prepare_data_for_clf(data_param_combo, bed, X_train, y_train, X_test, return_genes=True, random=args['random'])
            outer_gene_sets.append(bed["gene"].values[selected_indices].astype(str))

            if outer_fold_X_train is None:
                print("In outer fold, no genes left after filtering")
                fold_result["outer_fpr"] = []
                fold_result["outer_tpr"] = []
                fold_result["outer_roc_auc"] = 0
                fold_result["outer_tss_used"] = 0
                fold_result["outer_classification_report"] = []
                fold_result["outer_confusion_matrix"] = []
                fold_result["outer_test_num_0"] = 0
                fold_result["outer_test_num_1"] = 0
            else:
                best_inner_clf = CLASSIFIERS[best_inner_clf_name]
                best_inner_clf.set_params(**dict(zip(CLF_PARAM_GRIDS[best_inner_clf_name].keys(), best_inner_clf_params)))
                best_inner_clf.fit(outer_fold_X_train, y_train)
                y_pred = best_inner_clf.predict_proba(outer_fold_X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)

                fold_result["outer_test_num_0"] = np.sum(y_test == 0)
                fold_result["outer_test_num_1"] = np.sum(y_test == 1)
                fold_result["outer_fpr"] = fpr
                fold_result["outer_tpr"] = tpr
                fold_result["outer_roc_auc"] = roc_auc
                fold_result["outer_tss_used"] = outer_fold_X_train.shape[1]

                # Also extract other metrics like precision, recall, f1-score, etc.
                y_pred_binary = best_inner_clf.predict(outer_fold_X_test)
                fold_result["outer_classification_report"] = classification_report(y_test, y_pred_binary, output_dict=True, zero_division=0)
                fold_result["outer_confusion_matrix"] = confusion_matrix(y_test, y_pred_binary)

                if len(outer_gene_sets) > 1:
                    fold_result["outer_gene_sets_gji"] = generalized_jaccard_index(outer_gene_sets)
                    fold_result["outer_gene_sets_avgji"] = jaccard_index_average(outer_gene_sets)
                else:
                    fold_result["outer_gene_sets_gji"] = 0
                    fold_result["outer_gene_sets_avgji"] = 0

                fold_result["outer_gene_set"] = outer_gene_sets

                # Immediately also validate on external test set
                try:
                    X_external_feature_matrix = np.array(
                        [
                            X_external[i][data_param_combo["feature"]].values[selected_indices]
                            for i in range(len(X_external))
                        ]
                    )
                    X_external_feature_matrix = centerscale(X_external_feature_matrix, None)

                    y_external_pred = best_inner_clf.predict_proba(X_external_feature_matrix)[:, 1]
                    fpr_external, tpr_external, _ = roc_curve(y_external, y_external_pred)
                    roc_auc_external = auc(fpr_external, tpr_external)

                    fold_result["external_fpr"] = fpr_external
                    fold_result["external_tpr"] = tpr_external
                    fold_result["external_roc_auc"] = roc_auc_external

                    # Also extract other metrics like precision, recall, f1-score, etc.
                    y_external_pred_binary = best_inner_clf.predict(X_external_feature_matrix)
                    fold_result["external_classification_report"] = classification_report(y_external, y_external_pred_binary, output_dict=True, zero_division=0)
                    fold_result["external_confusion_matrix"] = confusion_matrix(y_external, y_external_pred_binary)

                except:
                    print("No external test set available")

            data_param_combo_results["folds"].append(fold_result)

        print(f"Finished processing data_param_combo: {data_param_combo} in {time() - outer_cv_time_start:.2f} seconds")        
        print(f"Inner-loop Best clf settings: {fold_result['best_inner_clf_name']} with params: {fold_result['best_inner_clf_params']} and mean AUC: {fold_result['best_inner_auc']:.4f}")
        print(f"The AUC over the outer folds in this data_param_combo: {np.mean([fold['outer_roc_auc'] for fold in data_param_combo_results['folds']])}")\
        
        # Format param_combo to string for file name
        data_param_combo_str = f"{data_param_combo['feature']}_{data_param_combo['top_genes']}_{data_param_combo['cov_x']}_{data_param_combo['cov_x_tss']}"

        with open(f"{args['output_dir']}/nestedcv_{data_param_combo_str}.pkl", "wb") as f:
            pickle.dump(data_param_combo_results, f)

        # Create .csv file with the results
        outer_report = nested_cv_report(data_param_combo_results)
        outer_report.to_csv(f"{args['output_dir']}/nestedcv_{data_param_combo_str}.csv")

        return data_param_combo_results

    data_param_combinations = [
        dict(zip(custom_data_params, v, strict=False))
        for v in itertools.product(*custom_data_params.values())
    ]
    print("Starting nested cross-validation with parallel processing...", args['processes'], "processes")
    results = Parallel(n_jobs=args['processes'])(
        delayed(process_data_param_combo)(data_param_combo) for data_param_combo in data_param_combinations
    )

    return results


def read_feature_file(path, data_columns, data_columns_dtypes):
    return pd.read_csv(path, usecols=data_columns, engine='pyarrow', dtype=data_columns_dtypes)


def load_data(args: dict, logger, feature_dir, diagnosis_table_path) -> tuple:
    """Load the data from the specified directories and return the feature matrices and labels.
    Args:
        args (dict): Dictionary containing the configuration parameters.
        logger: Logger object for logging messages.
    Returns:
        tuple: Tuple containing the feature matrices (X) and labels (y).
    """
    feature_files = glob.glob(f"{feature_dir}/*_features.csv")
    feature_files = sorted(feature_files)
    logger.info(f"Found {len(feature_files)} feature files in {feature_dir}")

    diagnosis_table = pd.read_csv(diagnosis_table_path, index_col=0)
    diagnosis_table = diagnosis_table[diagnosis_table["ctype"].isin(args['ctype'] + [args['controltype']])]

    # Loop through the diagnosis table and get the file_paths we need
    feature_files_experiment = []
    y = []
    for i, row in diagnosis_table.iterrows():
        sample_label, sample_ctype = row["label"], row["ctype"]
        for feature_file_path in feature_files:
            if sample_label in feature_file_path:
                feature_files_experiment.append(feature_file_path)
                if sample_ctype != args['controltype']:
                    y.append(1)
                else:
                    y.append(0)
                break
    
    logger.info(f"Found {len(feature_files_experiment)} feature files in {feature_dir} for the selected classes {args['ctype'] + [args['controltype']]}")

    # Columns to keep
    data_columns = args['feature'] + ['cov_x', 'cov_x_tss']
    data_columns_dtypes = {col: "float32" for col in data_columns}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        X = list(tqdm(
            executor.map(lambda f: read_feature_file(f, data_columns, data_columns_dtypes),
                        feature_files_experiment),
            total=len(feature_files_experiment),
            desc="Loading Data (Parallel)"
        ))

    y = np.array(y)

    logger.info(f"Loaded {len(X)} samples with {X[0].shape[1]} features each")
    logger.info(f"Loaded {len(y)} labels, {sum(y)} of which are cancer samples")
    logger.info(f"Diagnosis table value counts:\n{diagnosis_table['ctype'].value_counts()}")

    return X, y


def main(args: dict, logger) -> None:
    ## Load in the data and run Nested CV
    bed = pd.read_csv(args['bed'], sep="\t", header=None, usecols=[0, 1, 2, 3, 4])
    bed.columns = ["chr", "start", "end", "gene", "strand"]

    X, y = load_data(args, logger, args['feature_dir'], args['diagnosis_table'])
    X_external, y_external = load_data(args, logger, args['external_data_feature_dir'], args['external_data_diagnosis_table'])

    if args['best_model']:

        print("Running Best Model for all these features: ", args['feature'])
        for feature in args['feature']:
            print(f"Running Best Model for feature: {feature}")

            data_params = {
                "feature": [feature],
                "top_genes": args['top_genes'],
                "cov_x": args['cov_x'],
                "cov_x_tss": args['cov_x_tss']
            }

            # Run cross-validation for best model and use the best hyperparameters to extract genelist
            results_dict = model_cv(args, bed, X, y, X_external, y_external, custom_data_params=data_params)

            with open(f"{args['output_dir']}/best_model_results_{feature}.pkl", "wb") as f:
                pickle.dump(results_dict, f)

            # best_genes = best_model(results_dict, bed, X, y)
            # # Write the best genes to a line separated .txt file
            # with open(f"{args['output_dir']}/best_genes_{feature}.txt", "w") as f:
            #     f.write("\n".join(best_genes))
    
    else:
        # Run nested cross-validation for generalization performance
        print(f"Running Nested CV for all features: {args['feature']}")
        results_by_feature = {}

        for feature in args['feature']:
            print(f"Running nested CV for feature: {feature}")
                
            data_params = {
                "feature": [feature],
                "top_genes": args['top_genes'],
                "cov_x": args['cov_x'],
                "cov_x_tss": args['cov_x_tss']
            }

            # Run nested CV
            results = nested_cv(
                args=args,
                X=X,
                y=y,
                bed=bed,
                down_sampled_X={},
                X_external=X_external,
                y_external=y_external,
                custom_data_params=data_params
            )

            # Store results for this feature
            results_by_feature[feature] = results

            with open(f"{args['output_dir']}/nestedcv_results_{feature}.pkl", "wb") as f:
                pickle.dump(results_by_feature[feature], f)

        print("Completed nested_cv_multiple_features.")


if __name__ == "__main__":
    main()

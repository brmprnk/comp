import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # import numpy as np
    import pickle
    from glob import glob

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.font_manager import FontProperties  # Import for custom fonts
    from scipy.interpolate import interp1d
    from sklearn.metrics import auc

    # import seaborn as sns
    from tqdm import tqdm
    import os
    import gc
    from concurrent.futures import ThreadPoolExecutor, as_completed

    return (
        ThreadPoolExecutor,
        as_completed,
        gc,
        glob,
        interp1d,
        np,
        os,
        pd,
        pickle,
        plt,
        sns,
        tqdm,
    )


@app.cell
def _():
    # os.chdir("/mnt/u/KWFcfDNA/emc/comp")
    # print(os.getcwd())
    return


@app.cell
def _(ThreadPoolExecutor, as_completed, gc, glob, os, pd, pickle, tqdm):
    # Only the columns actually used by the analysis/plotting cells below.
    # Slimming each pickle on load keeps peak memory bounded.
    KEEP_COLS = [
        "feature", "classifier", "top_genes",
        "auc_full_model", "auc_per_cancer_type",
        "fpr_full_model", "tpr_full_model",
        "precision_full_model", "recall_full_model",
        "test_sample_ids",
    ]

    # Threads (not processes) share memory across workers — safer on small RAM.
    MAX_WORKERS = min(8, (os.cpu_count() or 2))


    def _load_slim(filepath, weighting_function):
        with open(filepath, "rb") as fh:
            df = pickle.load(fh)
        keep = [c for c in KEEP_COLS if c in df.columns]
        df = df[keep]
        df["weighting_function"] = weighting_function
        return df


    cris_metadata = pd.read_csv("accessory_files/cris_metadata.csv")

    tasks = []
    for weighting_function in ["mean_diff", "random", "p_value", "fold_change", "weighted"]:
        for fp in sorted(glob(f"nested_cv_results_cris/brca_crc_dd/results*{weighting_function}.pkl")):
            tasks.append((fp, weighting_function))

    dfs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_load_slim, fp, wf): fp for fp, wf in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading data-driven results"):
            dfs.append(fut.result())

    all_results_df = pd.concat(dfs, ignore_index=True, copy=False)
    all_results_df["top_genes"] = all_results_df["top_genes"].replace(15000, 14224)
    del dfs
    gc.collect()

    print(f"Successfully loaded {len(all_results_df):,} rows, "
          f"~{all_results_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return all_results_df, cris_metadata


@app.cell
def _():
    CLASSIFIER_COLORS = {
        'KNN': '#0173B2',  # Deep blue
        'Logistic Regression': '#DE8F05',  # Amber/gold
        'Multi-Layer Perceptron': '#029E73',  # Teal
        'Random Forest': '#CC78BC',  # Mauve
        'Support Vector Machine': '#CA3542',  # Crimson
    }

    WEIGHTING_COLORS = {
        'fold_change': '#56B4E9',  # Sky blue
        'mean_diff': '#E69F00',  # Orange
        'p_value': '#009E73',  # Green
        'random': '#999999',  # Medium gray
        'weighted': '#D55E00',  # Vermillion
    }

    FEATURE_COLORS = {
        'fslr': (46/255, 37/255, 133/255),      # Dark blue
        'fslr_coverage': (51/255, 117/255, 56/255),     # Green
        'griffin_diff': (93/255, 168/255, 153/255),        # Teal
        'rcov': (220/255, 205/255, 125/255),      # Gold
        'mwh': (159/255, 74/255, 150/255)        # Purple/Pink
    }

    # Colors for different approaches (RNA, ATAC, Data-driven, ichorCNA)
    APPROACH_COLORS = {
        'RNA': '#CA3542',  # Crimson (distinctive red)
        'ATAC': '#ECB22E',  # Gold/Amber (distinctive yellow-orange)
        'Data-driven': '#0173B2',  # Deep blue
        'ichorCNA': '#6A3D9A'  # Purple
    }
    return APPROACH_COLORS, CLASSIFIER_COLORS, FEATURE_COLORS, WEIGHTING_COLORS


@app.cell(hide_code=True)
def _(all_results_df, np, pd):
    # === Aggregate baseline: ichor data, ROC helpers, best config ===

    ichor_results_df = pd.read_csv(
        "accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv", index_col=0
    )


    def find_best_datadriven_config(results_df, feature, n_top_genes=None):
        """Best (classifier, weighting, top_genes) for a feature by mean full-model AUC."""
        df = results_df[results_df["feature"] == feature].copy()
        if n_top_genes is not None:
            df = df[df["top_genes"] == n_top_genes]
        if df.empty:
            return None
        grouped = (
            df.groupby(["classifier", "weighting_function", "top_genes"])
            .agg(mean_auc=("auc_full_model", "mean"))
            .reset_index()
        )
        best = grouped.loc[grouped["mean_auc"].idxmax()]
        return {
            "feature": feature,
            "classifier": best["classifier"],
            "weighting": best["weighting_function"],
            "top_genes": int(best["top_genes"]),
            "mean_auc": best["mean_auc"],
        }


    def compute_ichorcna_roc_metrics(
        ichor_df, metadata_df, sample_ids, cancer_type=None, score_column="ichorcna_tf"
    ):
        """ROC/PR/CM metrics for an aggregate score on one test fold.
        Thresholds are computed on the training set to avoid data leakage.
        """
        from sklearn.metrics import roc_curve, auc as _auc, precision_recall_curve, confusion_matrix as _cm

        train = ichor_df[~ichor_df["ID"].isin(sample_ids)].copy()
        test  = ichor_df[ ichor_df["ID"].isin(sample_ids)].copy()
        if cancer_type is not None:
            keep = ["Healthy", cancer_type]
            train = train[train["disease"].isin(keep)]
            test  = test[ test["disease"].isin(keep)]
        if len(test) == 0:
            return None

        def _labels(df):
            if cancer_type is None:
                return (df["disease"] != "Healthy").astype(int).values
            return (df["disease"] == cancer_type).astype(int).values

        y_tr, s_tr = _labels(train), train[score_column].values
        y_te, s_te = _labels(test),  test[score_column].values
        mask_tr = ~np.isnan(s_tr)
        mask_te = ~np.isnan(s_te)
        if not np.any(mask_tr) or not np.any(mask_te):
            return None
        y_tr, s_tr = y_tr[mask_tr], s_tr[mask_tr]
        y_te, s_te = y_te[mask_te], s_te[mask_te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            return None

        # Thresholds from training set
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, s_tr)
        thresholds = {"default_0.5": 0.5}
        youden_idx = np.argmax(tpr_tr - fpr_tr)
        thresholds["youden_optimal"] = float(thr_tr[youden_idx])
        spec_tr = 1 - fpr_tr
        idx_95 = np.where(spec_tr >= 0.95)[0]
        if len(idx_95):
            thresholds["95_percent_specificity"] = float(thr_tr[idx_95[-1]])

        # ROC and PR on test set
        fpr, tpr, _ = roc_curve(y_te, s_te)
        roc_auc = _auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_te, s_te)
        aupr = _auc(recall, precision)

        # Confusion matrices and classification reports per threshold
        confusion_matrices = {}
        classification_reports = {}
        for tname, tval in thresholds.items():
            y_pred = (s_te >= tval).astype(int)
            tn, fp, fn, tp = _cm(y_te, y_pred).ravel()
            confusion_matrices[tname] = {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            }
            prec_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_p  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_p   = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0
            prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec_n  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_n   = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0.0
            acc    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            classification_reports[tname] = {
                "0": {"precision": prec_n, "recall": rec_n, "f1-score": f1_n},
                "1": {"precision": prec_p, "recall": rec_p, "f1-score": f1_p},
                "accuracy": acc,
            }

        return {
            "fpr": fpr, "tpr": tpr, "auc": roc_auc,
            "precision": precision, "recall": recall, "aupr": aupr,
            "confusion_matrices": confusion_matrices,
            "classification_reports": classification_reports,
            "y_true": y_te, "y_pred_proba": s_te,
        }


    best_dd_config = find_best_datadriven_config(all_results_df, feature="griffin_diff")
    print(f"Best config: {best_dd_config}")

    return best_dd_config, compute_ichorcna_roc_metrics, ichor_results_df


@app.cell
def _(all_results_df, np, pd):
    # === Comprehensive Model Comparison: Find the Best Configuration ===

    def comprehensive_model_comparison(results_df):
        """
        Analyzes all combinations of features, classifiers, weighting functions, and gene counts
        to identify the best performing models.
        """

        print("="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("="*80)

        # Get all unique values
        features = sorted(results_df['feature'].unique())
        classifiers = sorted(results_df['classifier'].unique())
        weightings = sorted(results_df['weighting_function'].unique())
        gene_counts = sorted(results_df['top_genes'].unique())

        print(f"\nAnalyzing across:")
        print(f"  Features: {len(features)} ({', '.join(features)})")
        print(f"  Classifiers: {len(classifiers)} ({', '.join(classifiers)})")
        print(f"  Weighting functions: {len(weightings)} ({', '.join(weightings)})")
        print(f"  Gene counts: {len(gene_counts)} ({gene_counts})")

        # Aggregate performance metrics
        summary_data = []

        for feature in features:
            for classifier in classifiers:
                for weighting in weightings:
                    for n_genes in gene_counts:
                        subset = results_df[
                            (results_df['feature'] == feature) &
                            (results_df['classifier'] == classifier) &
                            (results_df['weighting_function'] == weighting) &
                            (results_df['top_genes'] == n_genes)
                        ]

                        if len(subset) == 0:
                            continue

                        # Calculate metrics
                        mean_full_auc = subset['auc_full_model'].mean()
                        std_full_auc = subset['auc_full_model'].std()

                        # Cancer-specific AUCs
                        breast_aucs = [row['auc_per_cancer_type'].get('Breast cancer', np.nan) 
                                       for _, row in subset.iterrows()]
                        crc_aucs = [row['auc_per_cancer_type'].get('Colorectal cancer', np.nan) 
                                    for _, row in subset.iterrows()]

                        mean_breast_auc = np.nanmean(breast_aucs)
                        std_breast_auc = np.nanstd(breast_aucs)
                        mean_crc_auc = np.nanmean(crc_aucs)
                        std_crc_auc = np.nanstd(crc_aucs)

                        # Minimum of the two cancer-specific AUCs (worst-case performance)
                        min_cancer_auc = min(mean_breast_auc, mean_crc_auc)

                        summary_data.append({
                            'feature': feature,
                            'classifier': classifier,
                            'weighting': weighting,
                            'n_genes': n_genes,
                            'mean_full_auc': mean_full_auc,
                            'std_full_auc': std_full_auc,
                            'mean_breast_auc': mean_breast_auc,
                            'std_breast_auc': std_breast_auc,
                            'mean_crc_auc': mean_crc_auc,
                            'std_crc_auc': std_crc_auc,
                            'min_cancer_auc': min_cancer_auc,
                            'n_folds': len(subset)
                        })

        summary_df = pd.DataFrame(summary_data)

        # Find best models by different criteria
        print("\n" + "="*80)
        print("TOP 10 MODELS BY DIFFERENT CRITERIA")
        print("="*80)

        criteria = [
            ('mean_full_auc', 'Overall AUC (Full Model)'),
            ('mean_breast_auc', 'Breast Cancer AUC'),
            ('mean_crc_auc', 'Colorectal Cancer AUC'),
            ('min_cancer_auc', 'Worst-Case Cancer AUC (min of Breast/CRC)')
        ]

        for metric, description in criteria:
            print(f"\n--- Top 10 by {description} ---")
            top_10 = summary_df.nlargest(10, metric)

            for idx, (i, row) in enumerate(top_10.iterrows(), 1):
                print(f"{idx:2d}. {row[metric]:.4f} | {row['feature']:<20} | {row['classifier']:<25} | "
                      f"{row['weighting']:<12} | {row['n_genes']:>5} genes")

        return summary_df

    # Run comprehensive analysis
    summary_df = comprehensive_model_comparison(all_results_df)
    summary_df.head()
    return (summary_df,)


@app.cell
def _(
    CLASSIFIER_COLORS,
    FEATURE_COLORS,
    WEIGHTING_COLORS,
    np,
    pd,
    plt,
    sns,
    summary_df,
):
    # === Visualizations to identify best models ===

    def visualize_best_models(summary_df):
        """
        Creates comprehensive visualizations to identify the best model configurations.
        Focus on how gene count affects performance across features, classifiers, and weightings.
        """

        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # Best overall AUC achieved (data-driven, full model) — referenced on line plots.
        BEST_AUC = float(summary_df["mean_full_auc"].max())

        # === PLOT 1: Heatmap - Feature vs Classifier (averaged across gene counts and weightings) ===
        ax1 = fig.add_subplot(gs[0, :2])

        # Name mappings for display
        classifier_name_map = {
            'KNN': 'KNN',
            'Logistic Regression': 'LR',
            'Multi-Layer Perceptron': 'MLP',
            'Random Forest': 'RF',
            'Support Vector Machine': 'SVM'
        }
        feature_name_map = {
            'fslr': 'FSLR',
            'fslr_coverage': 'FSLRC',
            'griffin_diff': 'GD',
            'rcov': 'RCOV',
            'mwh': 'MWH'
        }

        pivot_feat_clf = summary_df.groupby(['feature', 'classifier'])['mean_full_auc'].mean().unstack()
        # Rename columns and index
        pivot_feat_clf.columns = [classifier_name_map.get(c, c) for c in pivot_feat_clf.columns]
        pivot_feat_clf.index = [feature_name_map.get(f, f) for f in pivot_feat_clf.index]

        sns.heatmap(pivot_feat_clf, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax1, cbar_kws={'label': 'Mean AUC'})
        ax1.set_title('Average AUC by Feature and Classifier\n(averaged across gene counts and weightings)', 
                      fontsize=13, fontweight='bold')
        ax1.set_xlabel('Classifier', fontsize=11)
        ax1.set_ylabel('Feature', fontsize=11)

        # === PLOT 2: Best gene count distribution (ENHANCED) ===
        ax2 = fig.add_subplot(gs[0, 2])

        # For each feature-classifier-weighting combo, find the best gene count
        best_gene_counts = []
        gene_count_aucs = {}  # Store AUCs for each gene count

        for (feat, clf, weight), group in summary_df.groupby(['feature', 'classifier', 'weighting']):
            best_idx = group['mean_full_auc'].idxmax()
            best_gene_count = group.loc[best_idx, 'n_genes']
            best_auc = group.loc[best_idx, 'mean_full_auc']
            best_gene_counts.append(best_gene_count)

            # Store AUC for this gene count
            if best_gene_count not in gene_count_aucs:
                gene_count_aucs[best_gene_count] = []
            gene_count_aucs[best_gene_count].append(best_auc)

        # Get all possible gene counts from the data to ensure all are represented
        all_gene_counts = sorted(summary_df['n_genes'].unique())
        gene_count_freq = pd.Series(best_gene_counts).value_counts().reindex(all_gene_counts, fill_value=0).sort_index()

        # Calculate statistics for each gene count
        gene_count_stats = {}
        for gc in all_gene_counts:
            if gc in gene_count_aucs and len(gene_count_aucs[gc]) > 0:
                gene_count_stats[gc] = {
                    'min': np.min(gene_count_aucs[gc]),
                    'mean': np.mean(gene_count_aucs[gc]),
                    'max': np.max(gene_count_aucs[gc])
                }
            else:
                gene_count_stats[gc] = {'min': np.nan, 'mean': np.nan, 'max': np.nan}

        # Create bar plot with color intensity based on frequency
        bars = ax2.bar(range(len(gene_count_freq)), gene_count_freq.values, 
                       color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)

        # Color bars by frequency intensity
        max_freq = gene_count_freq.max()
        if max_freq > 0:
            for i, (bar, freq) in enumerate(zip(bars, gene_count_freq.values)):
                intensity = freq / max_freq
                bar.set_color(plt.cm.Blues(0.3 + 0.7 * intensity))

        ax2.set_xticks(range(len(gene_count_freq)))
        ax2.set_xticklabels(gene_count_freq.index, rotation=45, ha='right', fontsize=9)
        ax2.set_xlabel('Gene Count', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency (# of times optimal)', fontsize=11, fontweight='bold')
        ax2.set_title('Most Frequently Optimal Gene Counts', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add text annotations with statistics for top gene counts
        total_combinations = len(best_gene_counts)
        top_3_indices = gene_count_freq.nlargest(3).index

        for idx, gc in enumerate(all_gene_counts):
            freq = gene_count_freq[gc]
            if freq > 0 and gc in top_3_indices:
                stats = gene_count_stats[gc]
                # Add annotation above bar
                text = f"{freq}\n({freq/total_combinations*100:.0f}%)"
                ax2.text(idx, freq + max_freq * 0.02, text, 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Add summary text box
        n_features = len(summary_df['feature'].unique())
        n_classifiers = len(summary_df['classifier'].unique())
        n_weightings = len(summary_df['weighting'].unique())

        textstr = f'Total combinations: {total_combinations}\n'
        textstr += f'({n_features} features × {n_classifiers} classifiers × {n_weightings} weightings)\n\n'

        # Add top 3 gene counts info
        textstr += 'Top 3 optimal gene counts:\n'
        for i, gc in enumerate(top_3_indices, 1):
            freq = gene_count_freq[gc]
            pct = freq / total_combinations * 100
            stats = gene_count_stats[gc]
            textstr += f'{i}. {gc:,} genes: {freq}× ({pct:.0f}%)\n'
            if not np.isnan(stats['mean']):
                textstr += f'   AUC: {stats["min"]:.3f} - {stats["max"]:.3f} (μ={stats["mean"]:.3f})\n'

        # Place text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
        ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props,
                family='monospace')

        # === PLOT 3: Performance by weighting function ===
        ax3 = fig.add_subplot(gs[1, 0])

        weighting_perf = summary_df.groupby('weighting')['mean_full_auc'].agg(['mean', 'std'])
        weighting_perf = weighting_perf.sort_values('mean', ascending=False)

        # Use WEIGHTING_COLORS
        bar_colors = [WEIGHTING_COLORS.get(w, '#999999') for w in weighting_perf.index]
        bars = ax3.barh(range(len(weighting_perf)), weighting_perf['mean'], 
                         xerr=weighting_perf['std'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(weighting_perf)))
        ax3.set_yticklabels(weighting_perf.index)
        ax3.set_xlabel('Mean AUC', fontsize=11)
        ax3.set_title('Performance by Weighting Function', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim([0.5, 1.0])

        # === PLOT 4: Performance by classifier ===
        ax4 = fig.add_subplot(gs[1, 1])

        clf_perf = summary_df.groupby('classifier')['mean_full_auc'].agg(['mean', 'std'])
        clf_perf = clf_perf.sort_values('mean', ascending=False)

        # Use CLASSIFIER_COLORS
        bar_colors = [CLASSIFIER_COLORS.get(c, '#999999') for c in clf_perf.index]
        bars = ax4.barh(range(len(clf_perf)), clf_perf['mean'], 
                         xerr=clf_perf['std'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_yticks(range(len(clf_perf)))
        ax4.set_yticklabels(clf_perf.index)
        ax4.set_xlabel('Mean AUC', fontsize=11)
        ax4.set_title('Performance by Classifier', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        ax4.set_xlim([0.5, 1.0])

        # === PLOT 5: Performance by feature ===
        ax5 = fig.add_subplot(gs[1, 2])

        feat_perf = summary_df.groupby('feature')['mean_full_auc'].agg(['mean', 'std'])
        feat_perf = feat_perf.sort_values('mean', ascending=False)

        # Use FEATURE_COLORS
        bar_colors = [FEATURE_COLORS.get(f, '#999999') for f in feat_perf.index]
        bars = ax5.barh(range(len(feat_perf)), feat_perf['mean'], 
                         xerr=feat_perf['std'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax5.set_yticks(range(len(feat_perf)))
        ax5.set_yticklabels(feat_perf.index)
        ax5.set_xlabel('Mean AUC', fontsize=11)
        ax5.set_title('Performance by Feature', fontsize=12, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        ax5.set_xlim([0.5, 1.0])

        # === PLOT 6: AUC vs Gene Count by Classifier ===
        ax6 = fig.add_subplot(gs[2, 0])

        # Name mappings
        classifier_name_map = {
            'KNN': 'KNN',
            'K-Nearest Neighbors': 'KNN',
            'Logistic Regression': 'LR',
            'Multi-Layer Perceptron': 'MLP',
            'Random Forest': 'RF',
            'Support Vector Machine': 'SVM'
        }

        for classifier in sorted(summary_df['classifier'].unique()):
            subset = summary_df[summary_df['classifier'] == classifier]
            gene_auc = subset.groupby('n_genes').agg(
                mean=('mean_full_auc', 'mean'),
                best=('mean_full_auc', 'max'),
            )

            display_name = classifier_name_map.get(classifier, classifier)
            color = CLASSIFIER_COLORS.get(classifier, '#999999')
            # Best (max) AUC at each gene count — this is what touches BEST_AUC.
            ax6.plot(gene_auc.index, gene_auc['best'].values, marker='o',
                     label=display_name, linewidth=2.5, markersize=7, color=color)
            # Mean across the rest of the configurations as a lighter trace.
            ax6.plot(gene_auc.index, gene_auc['mean'].values,
                     linewidth=1.2, color=color, alpha=0.45, linestyle='--')

        ax6.axhline(BEST_AUC, color='black', linestyle=':', linewidth=1.4, alpha=0.75)
        ax6.text(ax6.get_xlim()[1], BEST_AUC, f' Best AUC = {BEST_AUC:.3f}',
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='black')
        ax6.set_xlabel('Number of Genes', fontsize=11)
        ax6.set_ylabel('Best AUC (solid) / Mean AUC (dashed)', fontsize=11)
        ax6.set_title('AUC vs Gene Count by Classifier', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9, loc='lower right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
        ax6.set_ylim([0.5, 1.0])

        # === PLOT 7: AUC vs Gene Count by Feature ===
        ax7 = fig.add_subplot(gs[2, 1])

        # Name mappings
        feature_name_map = {
            'fslr': 'FSLR',
            'fslr_coverage': 'FSLRC',
            'griffin_diff': 'GD',
            'rcov': 'RCOV',
            'mwh': 'MWH'
        }

        for feature in sorted(summary_df['feature'].unique()):
            subset = summary_df[summary_df['feature'] == feature]
            gene_auc = subset.groupby('n_genes').agg(
                mean=('mean_full_auc', 'mean'),
                best=('mean_full_auc', 'max'),
            )

            display_name = feature_name_map.get(feature, feature)
            color = FEATURE_COLORS.get(feature, '#999999')
            ax7.plot(gene_auc.index, gene_auc['best'].values, marker='s',
                     label=display_name, linewidth=2.5, markersize=7, color=color)
            ax7.plot(gene_auc.index, gene_auc['mean'].values,
                     linewidth=1.2, color=color, alpha=0.45, linestyle='--')

        ax7.axhline(BEST_AUC, color='black', linestyle=':', linewidth=1.4, alpha=0.75)
        ax7.text(ax7.get_xlim()[1], BEST_AUC, f' Best AUC = {BEST_AUC:.3f}',
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='black')
        ax7.set_xlabel('Number of Genes', fontsize=11)
        ax7.set_ylabel('Best AUC (solid) / Mean AUC (dashed)', fontsize=11)
        ax7.set_title('AUC vs Gene Count by Feature', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9, loc='lower right')
        ax7.grid(True, alpha=0.3)
        ax7.set_xscale('log')
        ax7.set_ylim([0.5, 1.0])

        # === PLOT 8: AUC vs Gene Count by Weighting Function ===
        ax8 = fig.add_subplot(gs[2, 2])

        # Name mappings
        weighting_name_map = {
            'fold_change': 'LOG-FOLD CHANGE',
            'mean_diff': 'ABS_MEAN_DIFF',
            'p_value': 'MWU',
            'random': 'RANDOM',
            'weighted': 'WEIGHTED'
        }

        for weighting in sorted(summary_df['weighting'].unique()):
            subset = summary_df[summary_df['weighting'] == weighting]
            gene_auc = subset.groupby('n_genes').agg(
                mean=('mean_full_auc', 'mean'),
                best=('mean_full_auc', 'max'),
            )

            display_name = weighting_name_map.get(weighting, weighting)
            color = WEIGHTING_COLORS.get(weighting, '#999999')
            ax8.plot(gene_auc.index, gene_auc['best'].values, marker='^',
                     label=display_name, linewidth=2.5, markersize=7, color=color)
            ax8.plot(gene_auc.index, gene_auc['mean'].values,
                     linewidth=1.2, color=color, alpha=0.45, linestyle='--')

        ax8.axhline(BEST_AUC, color='black', linestyle=':', linewidth=1.4, alpha=0.75)
        ax8.text(ax8.get_xlim()[1], BEST_AUC, f' Best AUC = {BEST_AUC:.3f}',
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='black')
        ax8.set_xlabel('Number of Genes', fontsize=11)
        ax8.set_ylabel('Best AUC (solid) / Mean AUC (dashed)', fontsize=11)
        ax8.set_title('AUC vs Gene Count by Weighting Function', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=9, loc='lower right')
        ax8.grid(True, alpha=0.3)
        ax8.set_xscale('log')
        ax8.set_ylim([0.5, 1.0])

        # === PLOT 9: Heatmap - Gene Count vs Classifier (MEAN - averaged across features and weightings) ===
        ax9 = fig.add_subplot(gs[3, :])

        pivot_genes_clf = summary_df.groupby(['n_genes', 'classifier'])['mean_full_auc'].mean().unstack()
        # Rename columns
        pivot_genes_clf.columns = [classifier_name_map.get(c, c) for c in pivot_genes_clf.columns]

        sns.heatmap(pivot_genes_clf, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax9, cbar_kws={'label': 'Mean AUC'})
        ax9.set_title('Mean AUC by Gene Count and Classifier', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Classifier', fontsize=11)
        ax9.set_ylabel('Number of Genes', fontsize=11)

        # === PLOT 10: Heatmap - Gene Count vs Classifier (MAX - maximum across features and weightings) ===
        ax10 = fig.add_subplot(gs[4, 0])

        pivot_genes_clf_max = summary_df.groupby(['n_genes', 'classifier'])['mean_full_auc'].max().unstack()
        # Rename columns
        pivot_genes_clf_max.columns = [classifier_name_map.get(c, c) for c in pivot_genes_clf_max.columns]

        sns.heatmap(pivot_genes_clf_max, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax10, cbar_kws={'label': 'Max AUC'})
        ax10.set_title('Max AUC by Gene Count and Classifier', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Classifier', fontsize=11)
        ax10.set_ylabel('Number of Genes', fontsize=11)

        # === PLOT 11: Heatmap - Gene Count vs Feature (MAX - maximum across classifiers and weightings) ===
        ax11 = fig.add_subplot(gs[4, 1])

        pivot_genes_feat_max = summary_df.groupby(['n_genes', 'feature'])['mean_full_auc'].max().unstack()
        # Rename columns
        pivot_genes_feat_max.columns = [feature_name_map.get(f, f) for f in pivot_genes_feat_max.columns]

        sns.heatmap(pivot_genes_feat_max, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax11, cbar_kws={'label': 'Max AUC'})
        ax11.set_title('Max AUC by Gene Count and Feature', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Feature', fontsize=11)
        ax11.set_ylabel('Number of Genes', fontsize=11)

        # === PLOT 12: Heatmap - Gene Count vs Weighting (MAX - maximum across classifiers and features) ===
        ax12 = fig.add_subplot(gs[4, 2])

        weighting_name_map = {
            'fold_change': 'LOG-FC',
            'mean_diff': 'ABS_MD',
            'p_value': 'MWU',
            'random': 'RANDOM',
            'weighted': 'WEIGHTED'
        }

        pivot_genes_weight_max = summary_df.groupby(['n_genes', 'weighting'])['mean_full_auc'].max().unstack()
        # Rename columns
        pivot_genes_weight_max.columns = [weighting_name_map.get(w, w) for w in pivot_genes_weight_max.columns]

        sns.heatmap(pivot_genes_weight_max, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    ax=ax12, cbar_kws={'label': 'Max AUC'})
        ax12.set_title('Max AUC by Gene Count and Weighting', fontsize=12, fontweight='bold')
        ax12.set_xlabel('Weighting Function', fontsize=11)
        ax12.set_ylabel('Number of Genes', fontsize=11)

        plt.suptitle('COMPREHENSIVE MODEL PERFORMANCE ANALYSIS', 
                     fontsize=16, fontweight='bold', y=0.995)

        plt.savefig('comprehensive_model_performance_analysis_brca_crc.png', dpi=1000, bbox_inches='tight')
        plt.show()


    # Generate visualizations
    visualize_best_models(summary_df)
    return


@app.cell
def _(np, pd, plt, summary_df):
    # === Standalone Plot: Most Frequently Optimal Gene Counts with AUC Statistics ===

    def plot_optimal_gene_counts_with_auc(summary_df):
        """
        Creates a combined visualization showing:
        1. Frequency bars: How often each gene count is optimal
        2. AUC line plot: Performance statistics (min, mean, max) for optimal configurations
        """

        # Calculate optimal gene counts for each combination
        best_gene_counts = []
        gene_count_aucs = {}  # Store AUCs for each gene count

        for (feat, clf, weight), group in summary_df.groupby(['feature', 'classifier', 'weighting']):
            best_idx = group['mean_full_auc'].idxmax()
            best_gene_count = group.loc[best_idx, 'n_genes']
            best_auc = group.loc[best_idx, 'mean_full_auc']
            best_gene_counts.append(best_gene_count)

            # Store AUC for this gene count
            if best_gene_count not in gene_count_aucs:
                gene_count_aucs[best_gene_count] = []
            gene_count_aucs[best_gene_count].append(best_auc)

        # Get all possible gene counts
        all_gene_counts = sorted(summary_df['n_genes'].unique())
        gene_count_freq = pd.Series(best_gene_counts).value_counts().reindex(all_gene_counts, fill_value=0).sort_index()

        # Calculate AUC statistics for each gene count
        auc_stats = []
        for gc in all_gene_counts:
            if gc in gene_count_aucs and len(gene_count_aucs[gc]) > 0:
                auc_stats.append({
                    'gene_count': gc,
                    'min': np.min(gene_count_aucs[gc]),
                    'mean': np.mean(gene_count_aucs[gc]),
                    'max': np.max(gene_count_aucs[gc]),
                    'std': np.std(gene_count_aucs[gc])
                })
            else:
                auc_stats.append({
                    'gene_count': gc,
                    'min': np.nan,
                    'mean': np.nan,
                    'max': np.nan,
                    'std': np.nan
                })

        auc_stats_df = pd.DataFrame(auc_stats)

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12,6))

        # === LEFT Y-AXIS: Frequency bars ===
        x_positions = np.arange(len(all_gene_counts))
        max_freq = gene_count_freq.max()

        # Color bars by frequency intensity
        bars = ax1.bar(x_positions, gene_count_freq.values, 
                       alpha=0.6, edgecolor='black', linewidth=1.5, 
                       width=0.7, label='Frequency (times optimal)')

        # Apply gradient coloring
        if max_freq > 0:
            for bar, freq in zip(bars, gene_count_freq.values):
                intensity = freq / max_freq
                bar.set_color(plt.cm.Blues(0.3 + 0.7 * intensity))

        ax1.set_xlabel('Number of Genes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (# of times optimal)', fontsize=14, fontweight='bold', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f'{int(gc):,}' for gc in all_gene_counts], rotation=45, ha='right', fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # === RIGHT Y-AXIS: AUC statistics ===
        ax2 = ax1.twinx()

        # Filter out gene counts with no data (NaN)
        valid_mask = ~auc_stats_df['mean'].isna()
        valid_positions = x_positions[valid_mask]
        valid_means = auc_stats_df.loc[valid_mask, 'mean'].values
        valid_mins = auc_stats_df.loc[valid_mask, 'min'].values
        valid_maxs = auc_stats_df.loc[valid_mask, 'max'].values

        # Plot mean AUC as a line
        ax2.plot(valid_positions, valid_means, 
                 color='darkred', linewidth=3, marker='o', markersize=10,
                 label='Mean AUC (when optimal)', zorder=5)

        # Add error bars showing min-max range
        ax2.errorbar(valid_positions, valid_means,
                     yerr=[valid_means - valid_mins, valid_maxs - valid_means],
                     fmt='none', ecolor='darkred', capsize=6, capthick=2, 
                     linewidth=2, alpha=0.6, zorder=4)

        # Add shaded region for min-max range
        ax2.fill_between(valid_positions, valid_mins, valid_maxs,
                         alpha=0.2, color='darkred', label='Min-Max AUC range')

        ax2.set_ylabel('AUC (when this gene count is optimal)', fontsize=14, fontweight='bold', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred', labelsize=11)
        ax2.set_ylim([0.5, 1.0])
        ax2.grid(axis='y', alpha=0.2, linestyle=':', color='darkred')

        # === Add annotations for top gene counts ===
        total_combinations = len(best_gene_counts)
        top_3_indices = gene_count_freq.nlargest(3).index

        for idx, gc in enumerate(all_gene_counts):
            freq = gene_count_freq[gc]
            if freq > 0 and gc in top_3_indices:
                # Frequency annotation
                pct = (freq / total_combinations) * 100
                ax1.text(idx, freq + max_freq * 0.02, 
                        f"{int(freq)}\n({pct:.0f}%)",
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='steelblue')

                # AUC annotation (if data exists)
                if gc in gene_count_aucs:
                    stats = auc_stats_df[auc_stats_df['gene_count'] == gc].iloc[0]
                    if not np.isnan(stats['mean']):
                        ax2.text(idx, stats['mean'] + 0.02,
                                f"μ={stats['mean']:.3f}",
                                ha='center', va='bottom', fontsize=9, fontweight='bold',
                                color='darkred', bbox=dict(boxstyle='round,pad=0.3', 
                                                           facecolor='white', 
                                                           edgecolor='darkred', 
                                                           alpha=0.8))

        # === Title and summary ===
        plt.title('Most Frequently Optimal Gene Counts with Performance Metrics',
                  fontsize=16, fontweight='bold', pad=20)

        # Add summary text box
        n_features = len(summary_df['feature'].unique())
        n_classifiers = len(summary_df['classifier'].unique())
        n_weightings = len(summary_df['weighting'].unique())

        textstr = f'Total combinations tested: {total_combinations}\n'
        textstr += f'({n_features} features × {n_classifiers} classifiers × {n_weightings} weightings)\n\n'
        textstr += 'Top 3 most frequently optimal:\n'

        for i, gc in enumerate(top_3_indices, 1):
            freq = gene_count_freq[gc]
            pct = (freq / total_combinations) * 100
            if gc in gene_count_aucs:
                stats = auc_stats_df[auc_stats_df['gene_count'] == gc].iloc[0]
                textstr += f'{i}. {int(gc):,} genes: {int(freq)}× ({pct:.0f}%)\n'
                if not np.isnan(stats['mean']):
                    textstr += f'   AUC range: {stats["min"]:.3f} - {stats["max"]:.3f}\n'
                    textstr += f'   Mean AUC: {stats["mean"]:.3f} ± {stats["std"]:.3f}\n'

        # props = dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, 
        #              edgecolor='black', linewidth=2)
        # ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
        #         verticalalignment='top', horizontalalignment='left', 
        #         bbox=props, family='monospace')

        print(textstr)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', fontsize=11, framealpha=0.95)

        plt.tight_layout()
        plt.show()

        # Print interpretation
        print("\n" + "="*80)
        print("INTERPRETATION: OPTIMAL GENE COUNTS")
        print("="*80)

        print("\n📊 What this plot shows:")
        print("   • BARS (left axis): How many times each gene count was optimal")
        print("   • LINE (right axis): Mean AUC when that gene count was the best choice")
        print("   • ERROR BARS: Range from minimum to maximum AUC")

        print("\n🔍 Key insights:")
        for i, gc in enumerate(top_3_indices, 1):
            freq = gene_count_freq[gc]
            pct = (freq / total_combinations) * 100
            if gc in gene_count_aucs:
                stats = auc_stats_df[auc_stats_df['gene_count'] == gc].iloc[0]
                print(f"\n   {i}. {int(gc):,} genes:")
                print(f"      • Optimal {int(freq)} times ({pct:.1f}% of all combinations)")
                if not np.isnan(stats['mean']):
                    print(f"      • When optimal, achieves AUC = {stats['mean']:.3f} ± {stats['std']:.3f}")
                    print(f"      • AUC range: {stats['min']:.3f} (worst) to {stats['max']:.3f} (best)")

        print("\n💡 Biological meaning:")
        print("   • High frequency = This gene count works well across many approaches")
        print("   • High AUC = Strong cancer detection performance")
        print("   • Narrow range = Consistent performance across different methods")
        print("   • Wide range = Performance varies depending on feature/classifier choice")

        print("\n" + "="*80 + "\n")

    # Generate the plot
    plot_optimal_gene_counts_with_auc(summary_df)
    return


@app.cell
def _(CLASSIFIER_COLORS, FEATURE_COLORS, plt):
    # === Section 3.3 Main Figure: Feature + Classifier panels ===
    # For each group we plot TWO traces per k:
    #   • solid, thick, saturated — the BEST (max) AUROC at that k
    #   • dashed, thin, 45% alpha — the MEAN AUROC across configs at that k
    # Same colour in both, so the gap between the two lines is itself a visual
    # cue for how much hyperparameter/weighting search is gaining you.

    def _best_and_mean_per_k(sub):
        agg = sub.groupby("n_genes").agg(
            mean=("mean_full_auc", "mean"),
            best=("mean_full_auc", "max"),
        )
        return agg.index.to_numpy(), agg["best"].to_numpy(), agg["mean"].to_numpy()


    def _style_axis(ax):
        ax.set_xscale("log")
        ax.set_ylim(0.55, 1.0)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="major", linestyle=":", linewidth=0.6,
                color="0.7", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")


    def _add_best_reference(ax, best_auc):
        ax.axhline(best_auc, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.8, zorder=1)
        x_left = ax.get_xlim()[0]
        ax.text(x_left, best_auc, f" Best AUROC = {best_auc:.3f} ",
                ha="left", va="bottom", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=4)


    def plot_section33_main(summary_df, best_auc=None,
                            figsize=(12, 4.8), save_path=None):
        """Two-panel main figure.
           Panel A — per feature: solid = best AUROC, dashed = mean AUROC.
           Panel B — per classifier: same convention.
        """
        if best_auc is None:
            best_auc = float(summary_df["mean_full_auc"].max())

        feature_name_map = {
            "fslr": "FSLR", "fslr_coverage": "FSLRC", "griffin_diff": "GD",
            "rcov": "RCOV", "mwh": "MWH",
        }
        classifier_name_map = {
            "KNN": "KNN", "Logistic Regression": "LR",
            "Multi-Layer Perceptron": "MLP", "Random Forest": "RF",
            "Support Vector Machine": "SVM",
        }

        fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

        # ---- Panel A — per feature ---------------------------------------------
        feats_sorted = (
            summary_df.groupby("feature")["mean_full_auc"].max()
            .sort_values(ascending=False).index.tolist()
        )
        for feat in feats_sorted:
            sub = summary_df[summary_df["feature"] == feat]
            k, best, mean = _best_and_mean_per_k(sub)
            color = FEATURE_COLORS.get(feat, "#999999")
            axA.plot(k, best, marker="o", markersize=5, linewidth=2.4,
                     color=color, zorder=3,
                     label=feature_name_map.get(feat, feat))
            axA.plot(k, mean, linestyle="--", linewidth=1.2,
                     color=color, alpha=0.45, zorder=2)

        _style_axis(axA)
        _add_best_reference(axA, best_auc)
        axA.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        axA.set_ylabel("Best AUROC (solid) / Mean AUROC (dashed)",
                       fontsize=10, fontweight="bold")
        axA.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                   title="Feature", title_fontsize=8)

        # ---- Panel B — per classifier ------------------------------------------
        clfs_sorted = (
            summary_df.groupby("classifier")["mean_full_auc"].max()
            .sort_values(ascending=False).index.tolist()
        )
        for clf in clfs_sorted:
            sub = summary_df[summary_df["classifier"] == clf]
            k, best, mean = _best_and_mean_per_k(sub)
            color = CLASSIFIER_COLORS.get(clf, "#999999")
            axB.plot(k, best, marker="o", markersize=5, linewidth=2.4,
                     color=color, zorder=3,
                     label=classifier_name_map.get(clf, clf))
            axB.plot(k, mean, linestyle="--", linewidth=1.2,
                     color=color, alpha=0.45, zorder=2)

        _style_axis(axB)
        _add_best_reference(axB, best_auc)
        axB.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        axB.set_ylabel("Best AUROC (solid) / Mean AUROC (dashed)",
                       fontsize=10, fontweight="bold")
        axB.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                   title="Classifier", title_fontsize=8)

        # ---- Footer making the nested-CV scheme explicit -----------------------
        n_outer = int(summary_df["n_folds"].max()) if "n_folds" in summary_df.columns else None
        nested_note = (
            f"Nested cross-validation: inner folds tune hyperparameters, "
            f"outer {n_outer}-fold evaluates"
            if n_outer else "Nested cross-validation"
        )
        fig.text(0.5, -0.015, nested_note, ha="center", va="top",
                 fontsize=8.5, style="italic", color="0.25")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return fig, (axA, axB)


    return


@app.cell
def _(
    APPROACH_COLORS,
    CLASSIFIER_COLORS,
    FEATURE_COLORS,
    all_results_df,
    best_dd_config,
    compute_ichorcna_roc_metrics,
    cris_metadata,
    ichor_results_df,
    interp1d,
    np,
    plt,
    summary_df,
):
    plt.close("all")

    # === Figure 4 -- 1x3 grid (A: features, B: classifiers, C: ROC curves) ===
    # Panel A -- AUROC vs n_genes, per feature            (solid = best, dashed = mean)
    # Panel B -- AUROC vs n_genes, per classifier         (solid = best, dashed = mean)
    # Panel C -- Full-model ROC: data-driven vs aggregate baselines

    FEATURE_NAME_MAP = {
        "fslr": "FSLR", "fslr_coverage": "FSLRC", "griffin_diff": "GD",
        "rcov": "RCOV", "mwh": "MWH",
    }
    CLASSIFIER_NAME_MAP = {
        "KNN": "KNN", "Logistic Regression": "LR",
        "Multi-Layer Perceptron": "MLP", "Random Forest": "RF",
        "Support Vector Machine": "SVM",
    }


    def _style_panel(ax, ylim=(0.55, 1.0)):
        ax.set_xscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="major", linestyle=":", linewidth=0.6,
                color="0.7", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")


    def _best_ref(ax, best_auc):
        ax.axhline(best_auc, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.8, zorder=1)
        x_left = ax.get_xlim()[0]
        ax.text(x_left, best_auc, f" Best AUROC = {best_auc:.3f} ",
                ha="left", va="bottom", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=4)


    def _panel_label(ax, letter):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-30, 8), textcoords="offset points",
                    fontsize=14, fontweight="bold",
                    va="bottom", ha="left")


    def _panel_feature(ax, summary_df, best_auc):
        feats_sorted = (
            summary_df.groupby("feature")["mean_full_auc"].max()
            .sort_values(ascending=False).index.tolist()
        )
        for feat in feats_sorted:
            sub = summary_df[summary_df["feature"] == feat]
            agg = sub.groupby("n_genes").agg(
                best=("mean_full_auc", "max"),
                mean=("mean_full_auc", "mean"),
            )
            color = FEATURE_COLORS.get(feat, "#999999")
            ax.plot(agg.index, agg["best"].values, marker="o", markersize=5,
                    linewidth=2.4, color=color, zorder=3,
                    label=FEATURE_NAME_MAP.get(feat, feat))
            ax.plot(agg.index, agg["mean"].values, linestyle="--",
                    linewidth=1.2, color=color, alpha=0.45, zorder=2)
        _style_panel(ax)
        _best_ref(ax, best_auc)
        ax.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        ax.set_ylabel("Best AUROC (solid) / Mean AUROC (dashed)",
                      fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                  title="Feature", title_fontsize=8)


    def _panel_classifier(ax, summary_df, best_auc):
        clfs_sorted = (
            summary_df.groupby("classifier")["mean_full_auc"].max()
            .sort_values(ascending=False).index.tolist()
        )
        for clf in clfs_sorted:
            sub = summary_df[summary_df["classifier"] == clf]
            agg = sub.groupby("n_genes").agg(
                best=("mean_full_auc", "max"),
                mean=("mean_full_auc", "mean"),
            )
            color = CLASSIFIER_COLORS.get(clf, "#999999")
            ax.plot(agg.index, agg["best"].values, marker="o", markersize=5,
                    linewidth=2.4, color=color, zorder=3,
                    label=CLASSIFIER_NAME_MAP.get(clf, clf))
            ax.plot(agg.index, agg["mean"].values, linestyle="--",
                    linewidth=1.2, color=color, alpha=0.45, zorder=2)
        _style_panel(ax)
        _best_ref(ax, best_auc)
        ax.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        ax.set_ylabel("Best AUROC (solid) / Mean AUROC (dashed)",
                      fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                  title="Classifier", title_fontsize=8)


    def _panel_roc(ax, all_results_df, ichor_results_df, cris_metadata, best_dd_config):
        """Full-model ROC curves: data-driven best config vs all 5 aggregate features."""
        base_fpr = np.linspace(0, 1, 101)

        dd = all_results_df[
            (all_results_df["feature"] == best_dd_config["feature"]) &
            (all_results_df["classifier"] == best_dd_config["classifier"]) &
            (all_results_df["weighting_function"] == best_dd_config["weighting"]) &
            (all_results_df["top_genes"] == best_dd_config["top_genes"])
        ]

        # Data-driven mean ROC (solid, thick) with std band
        tpr_list, auc_list = [], []
        for _, row in dd.iterrows():
            fn = interp1d(row["fpr_full_model"], row["tpr_full_model"],
                          bounds_error=False, fill_value=(0.0, 1.0))
            tpr_list.append(fn(base_fpr))
            auc_list.append(row["auc_full_model"])
        if tpr_list:
            mean_tpr = np.mean(tpr_list, axis=0)
            std_tpr  = np.std(tpr_list, axis=0)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
            color_dd = APPROACH_COLORS["Data-driven"]
            ax.plot(base_fpr, mean_tpr, linewidth=2.4, color=color_dd, zorder=3,
                    label=f"Data-driven  (AUC={np.mean(auc_list):.3f} +/- {np.std(auc_list):.3f})")
            ax.fill_between(base_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=color_dd, alpha=0.15, zorder=2)

        # Aggregate feature ROC curves (dashed) with std bands
        agg_map = {
            "aggregate_griffin_diff": ("GD",    FEATURE_COLORS.get("griffin_diff",    "#999999")),
            "aggregate_rcov":         ("RCOV",  FEATURE_COLORS.get("rcov",            "#999999")),
            "aggregate_mwh":          ("MWH",   FEATURE_COLORS.get("mwh",             "#999999")),
            "aggregate_fslr":         ("FSLR",  FEATURE_COLORS.get("fslr",            "#999999")),
            "aggregate_fslr_coverage":("FSLRC", FEATURE_COLORS.get("fslr_coverage",   "#999999")),
        }
        for col, (label, color) in agg_map.items():
            tpr_list, auc_list = [], []
            for _, row in dd.iterrows():
                m = compute_ichorcna_roc_metrics(
                    ichor_results_df, cris_metadata, row["test_sample_ids"],
                    cancer_type=None, score_column=col,
                )
                if m:
                    fn = interp1d(m["fpr"], m["tpr"],
                                  bounds_error=False, fill_value=(0.0, 1.0))
                    tpr_list.append(fn(base_fpr))
                    auc_list.append(m["auc"])
            if tpr_list:
                mean_tpr = np.mean(tpr_list, axis=0)
                std_tpr  = np.std(tpr_list, axis=0)
                mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
                ax.plot(base_fpr, mean_tpr, linewidth=1.8, linestyle="--",
                        color=color, alpha=0.85, zorder=2,
                        label=f"{label}  (AUC={np.mean(auc_list):.3f} ± {np.std(auc_list):.3f})")
                ax.fill_between(base_fpr,
                                np.clip(mean_tpr - std_tpr, 0, 1),
                                np.clip(mean_tpr + std_tpr, 0, 1),
                                color=color, alpha=0.12, zorder=1)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1.0, zorder=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92,
                  title="Approach", title_fontsize=8)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="major", linestyle=":", linewidth=0.6,
                color="0.7", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")


    def plot_figure4(summary_df, best_auc=None, figsize=(15, 5), save_path=None,
                     all_results_df=None, ichor_results_df=None,
                     cris_metadata=None, best_dd_config=None):
        """Manuscript Figure 4 -- 1x3 grid. A/B: AUROC vs genes; C: full-model ROC curves."""
        if best_auc is None:
            best_auc = float(summary_df["mean_full_auc"].max())

        fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=figsize)

        _panel_feature(axA, summary_df, best_auc);    _panel_label(axA, "A")
        _panel_classifier(axB, summary_df, best_auc); _panel_label(axB, "B")
        _panel_roc(axC, all_results_df, ichor_results_df, cris_metadata, best_dd_config)
        _panel_label(axC, "C")

        n_outer = int(summary_df["n_folds"].max()) if "n_folds" in summary_df.columns else None
        nested_note = (
            f"Nested cross-validation: inner folds tune hyperparameters, "
            f"outer {n_outer}-fold evaluates"
            if n_outer else "Nested cross-validation"
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.28)
        # fig.text(0.5, 0.02, nested_note, ha="center", va="bottom",
        #          fontsize=9, style="italic", color="0.25")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.show()
        return fig, (axA, axB, axC)


    plot_figure4(summary_df, save_path="figure4.png",
                 all_results_df=all_results_df, ichor_results_df=ichor_results_df,
                 cris_metadata=cris_metadata, best_dd_config=best_dd_config)

    return


@app.cell(hide_code=True)
def _(WEIGHTING_COLORS, plt, summary_df):
    # === Supplementary: weighting-function sweep ===

    weighting_name_map = {
        "fold_change": "log-FC",
        "mean_diff": "|Δμ|",
        "p_value": "MWU",
        "random": "random",
        "weighted": "weighted",
    }

    BEST_AUC = float(summary_df["mean_full_auc"].max())

    fig, ax = plt.subplots(figsize=(8, 5))

    for weighting in sorted(summary_df["weighting"].unique()):
        subset = summary_df[summary_df["weighting"] == weighting]
        gene_auc = subset.groupby("n_genes").agg(
            mean=("mean_full_auc", "mean"),
            best=("mean_full_auc", "max"),
        )
        color = WEIGHTING_COLORS.get(weighting, "#999999")
        label = weighting_name_map.get(weighting, weighting)
        ax.plot(gene_auc.index, gene_auc["best"].values,
                marker="^", markersize=6, linewidth=2.5, color=color,
                label=label, zorder=3)
        ax.plot(gene_auc.index, gene_auc["mean"].values,
                linestyle="--", linewidth=1.2, color=color, alpha=0.45, zorder=2)

    ax.axhline(BEST_AUC, color="black", linestyle=":", linewidth=1.4, alpha=0.75)
    ax.text(ax.get_xlim()[1], BEST_AUC, f" Best AUC = {BEST_AUC:.3f}",
            va="center", ha="left", fontsize=8.5, fontweight="bold", color="black")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Genes", fontsize=11)
    ax.set_ylabel("Best AUC (solid) / Mean AUC (dashed)", fontsize=11)
    ax.set_title("AUC vs Gene Count by Weighting Function", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    fig.savefig("suppl_weighting.png", dpi=300, bbox_inches="tight")
    plt.show()

    return


@app.cell
def _(FEATURE_COLORS, plt, summary_df):
    # === Anti-overfitting diagnostics ===
    # Pre-empts the reviewer reading "more features → better AUROC" as overfitting.
    #
    # Panel A — Cross-fold AUROC standard deviation vs n_genes for the best config
    #   at each k, per feature. Overfitting's fingerprint is *growing* fold-to-fold
    #   variance as k increases (the model chases noise that differs across folds).
    #   Flat or declining std at high k means the signal is stable across splits.
    #
    # Panel B — Signal-vs-random weighting gap vs n_genes. For each feature and k,
    #   we compare the best signal-weighted config against the best random-weighted
    #   config at the same k. If models were memorizing labels regardless of input,
    #   this gap would collapse. A persistent or growing gap means the biological
    #   gene scoring is adding real information on top of raw feature availability.

    def plot_overfitting_diagnostic(summary_df, figsize=(12, 4.5), save_path=None):
        feature_name_map = {
            "fslr": "FSLR", "fslr_coverage": "FSLRC", "griffin_diff": "GD",
            "rcov": "RCOV", "mwh": "MWH",
        }

        fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

        feats_sorted = (
            summary_df.groupby("feature")["mean_full_auc"].max()
            .sort_values(ascending=False).index.tolist()
        )

        # ---- Panel A — std of best config at each k, per feature ---------------
        for feat in feats_sorted:
            sub = summary_df[summary_df["feature"] == feat]
            best_per_k = sub.loc[sub.groupby("n_genes")["mean_full_auc"].idxmax()]
            color = FEATURE_COLORS.get(feat, "#999999")
            axA.plot(best_per_k["n_genes"], best_per_k["std_full_auc"],
                     marker="o", markersize=5, linewidth=2.0, color=color,
                     label=feature_name_map.get(feat, feat))
        axA.set_xscale("log")
        axA.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        axA.set_ylabel("Cross-fold AUROC std. dev.", fontsize=10, fontweight="bold")
        axA.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9,
                   title="Feature", title_fontsize=8)
        axA.tick_params(labelsize=9)
        for sp in axA.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")

        # ---- Panel B — signal-weighted minus random-weighted, per feature ------
        for feat in feats_sorted:
            sub = summary_df[summary_df["feature"] == feat]
            sig = sub[sub["weighting"] != "random"]
            rnd = sub[sub["weighting"] == "random"]
            if sig.empty or rnd.empty:
                continue
            sig_best = sig.groupby("n_genes")["mean_full_auc"].max()
            rnd_best = rnd.groupby("n_genes")["mean_full_auc"].max()
            common = sig_best.index.intersection(rnd_best.index)
            gap = sig_best.loc[common] - rnd_best.loc[common]
            color = FEATURE_COLORS.get(feat, "#999999")
            axB.plot(common, gap.values, marker="o", markersize=5, linewidth=2.0,
                     color=color, label=feature_name_map.get(feat, feat))
        axB.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        axB.set_xscale("log")
        axB.set_xlabel("Number of Genes", fontsize=10, fontweight="bold")
        axB.set_ylabel("AUROC gap  (signal − random)", fontsize=10, fontweight="bold")
        axB.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9,
                   title="Feature", title_fontsize=8)
        axB.tick_params(labelsize=9)
        for sp in axB.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        return
        # return fig, (axA, axB)


    plot_overfitting_diagnostic(summary_df, save_path="section33_overfit_diag.pdf")
    return


@app.cell
def _():
    import scipy

    test1 = [-3.5, -4.5]
    scipy.special.softmax(test1)
    return


if __name__ == "__main__":
    app.run()

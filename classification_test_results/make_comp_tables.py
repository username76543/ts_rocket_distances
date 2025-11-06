import os

import ast
import json
import numpy as np
from scipy.stats import rankdata, studentized_range
from math import sqrt
import matplotlib.pyplot as plt
import math

{"Trace": {"poi_ridge_acc": 1.0, "poi_rf_acc": 0.10096096992492676, "nn_acc": 0.82, "rocket_ridge_acc": 1.0, "minirocket_ridge_acc": 1.0, "multirocket_ridge_acc": 1.0, "quant_rf_acc": 1.0}, "ToeSegmentation1": {"poi_ridge_acc": 0.956140350877193, "poi_rf_acc": 0.04786396026611328, "nn_acc": 0.8114035087719298, "rocket_ridge_acc": 0.9385964912280702, "minirocket_ridge_acc": 0.956140350877193, "multirocket_ridge_acc": 0.9605263157894737, "quant_rf_acc": 0.7675438596491229}, "Coffee": {"poi_ridge_acc": 1.0, "poi_rf_acc": 0.08775448799133301, "nn_acc": 0.9642857142857143, "rocket_ridge_acc": 1.0, "minirocket_ridge_acc": 1.0, "multirocket_ridge_acc": 1.0, "quant_rf_acc": 1.0}, "DodgerLoopWeekend": {"poi_ridge_acc": 0.9682539682539683, "poi_rf_acc": 0.08735489845275879, "nn_acc": 0.9841269841269841, "rocket_ridge_acc": 0.9682539682539683, "minirocket_ridge_acc": 0.9841269841269841, "multirocket_ridge_acc": 0.9841269841269841, "quant_rf_acc": 0.9841269841269841}, "DodgerLoopGame": {"poi_ridge_acc": 0.7401574803149606, "poi_rf_acc": 0.046472787857055664, "nn_acc": 0.7559055118110236, "rocket_ridge_acc": 0.8976377952755905, "minirocket_ridge_acc": 0.8503937007874016, "multirocket_ridge_acc": 0.8976377952755905, "quant_rf_acc": 0.7952755905511811}, "DodgerLoopDay": {"poi_ridge_acc": 0.5324675324675324, "poi_rf_acc": 0.05724763870239258, "nn_acc": 0.4675324675324675, "rocket_ridge_acc": 0.5324675324675324, "minirocket_ridge_acc": 0.6493506493506493, "multirocket_ridge_acc": 0.5584415584415584, "quant_rf_acc": 0.6233766233766234}, "CricketZ": {"poi_ridge_acc": 0.8384615384615385, "poi_rf_acc": 0.20163393020629883, "nn_acc": 0.7410256410256411, "rocket_ridge_acc": 0.7871794871794872, "minirocket_ridge_acc": 0.7897435897435897, "multirocket_ridge_acc": 0.8102564102564103, "quant_rf_acc": 0.717948717948718}, "CricketY": {"poi_ridge_acc": 0.8256410256410256, "poi_rf_acc": 0.19809246063232422, "nn_acc": 0.7794871794871795, "rocket_ridge_acc": 0.7846153846153846, "minirocket_ridge_acc": 0.7923076923076923, "multirocket_ridge_acc": 0.8025641025641026, "quant_rf_acc": 0.735897435897436}, "CricketX": {"poi_ridge_acc": 0.8076923076923077, "poi_rf_acc": 0.1978468894958496, "nn_acc": 0.7512820512820513, "rocket_ridge_acc": 0.782051282051282, "minirocket_ridge_acc": 0.7897435897435897, "multirocket_ridge_acc": 0.7871794871794872, "quant_rf_acc": 0.7}, "FreezerRegularTrain": {"poi_ridge_acc": 0.9947368421052631, "poi_rf_acc": 0.07769346237182617, "nn_acc": 0.9140350877192982, "rocket_ridge_acc": 0.9947368421052631, "minirocket_ridge_acc": 0.9992982456140351, "multirocket_ridge_acc": 0.992280701754386, "quant_rf_acc": 0.9996491228070176}, "FreezerSmallTrain": {"poi_ridge_acc": 0.7670175438596492, "poi_rf_acc": 0.060745954513549805, "nn_acc": 0.7035087719298245, "rocket_ridge_acc": 0.7463157894736843, "minirocket_ridge_acc": 0.9403508771929825, "multirocket_ridge_acc": 0.8589473684210527, "quant_rf_acc": 0.9898245614035087}}


def rank_floats(values):
    """
    Ranks a list of floats such that:
      - The highest value gets rank 0
      - The second highest gets rank 1, etc.
      - Tied values receive the same rank.
    Returns a list of ranks corresponding to the input order.
    """
    # Get the unique values sorted descending
    sorted_unique = sorted(set(values), reverse=True)
    
    # Create a mapping of value -> rank
    rank_map = {val: rank for rank, val in enumerate(sorted_unique)}
    
    # Map each original value to its rank
    return [rank_map[v] for v in values]


float_list = [2.5, 3.1, 2.5, 4.0, 3.1]
ranks = rank_floats(float_list)
print(ranks)  # Output: [1, 2, 1, 4, 2]

def scatter_grid(pairs, titles=None, max_per_fig=9, save_prefix="scatter_page"):
    """
    Plot multiple x–y scatterplots in grid layout, adding y=x line.
    Automatically splits into multiple figures if there are more than max_per_fig.

    Args:
        pairs (list of tuples): Each element is a (x, y) pair of equal-length lists or arrays.
        titles (list of str, optional): Titles for each scatterplot.
        max_per_fig (int, optional): Maximum number of subplots per figure (default: 9).
        save_prefix (str, optional): Prefix for saved figure filenames (default: 'scatter_page').
    """
    n = len(pairs)
    titles = titles or [f"Pair {i+1}" for i in range(n)]

    # Determine how many figures are needed
    n_figs = math.ceil(n / max_per_fig)

    for fig_idx in range(n_figs):
        start = fig_idx * max_per_fig
        end = min(start + max_per_fig, n)
        subset = pairs[start:end]
        subset_titles = titles[start:end]

        num_subplots = len(subset)
        cols = math.ceil(math.sqrt(num_subplots))
        rows = math.ceil(num_subplots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten() if num_subplots > 1 else np.array([axes])

        for i, (x, y) in enumerate(subset):
            ax = axes[i]
            ax.scatter(x, y, alpha=0.7)
            ax.set_title(subset_titles[i])
            ax.set_xlabel(subset_titles[i].split('/')[0])
            ax.set_ylabel(subset_titles[i].split('/')[1])

            # Draw y=x line
            min_val = min(min(x), min(y))
            max_val = max(max(x), max(y))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)

        # Hide unused subplots
        for ax in axes[num_subplots:]:
            ax.axis('off')

        plt.tight_layout()

        # Save and show
        filename = f"{save_prefix}_{fig_idx+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filename}")

        #plt.show()


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def nemenyi_cd(num_methods, num_datasets, alpha=0.05):
    """
    Compute Critical Difference (CD) for the Nemenyi post-hoc test
    based on Demšar (2006).
    """
    q_alpha = {
        0.10: [0, 1.65, 1.96, 2.10, 2.22, 2.33, 2.39, 2.44, 2.49, 2.52, 2.56],
        0.05: [0, 1.96, 2.24, 2.34, 2.43, 2.52, 2.56, 2.61, 2.65, 2.68, 2.72],
        0.01: [0, 2.57, 2.91, 3.05, 3.17, 3.27, 3.33, 3.39, 3.43, 3.47, 3.50],
    }
    q = q_alpha.get(alpha, q_alpha[0.05])[min(num_methods, 10)]
    return q * sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))


def visualize_ranking_results_with_cd(results, num_datasets, alpha=0.05, save_path=None):
    """
    Visualize ranking results using:
      1. A Critical Difference (CD) diagram (single line layout)
      2. A bar chart showing frequency of best performance

    Parameters
    ----------
    results : dict
        {
            'approach_name': {
                'mean_rank': float,
                'rank_counts': {0: int, 1: int, ...}
            },
            ...
        }
    num_datasets : int
        Number of datasets used for ranking.
    alpha : float
        Significance level for CD computation.
    save_path : str or None
        Optional path to save the figure.
    """
    # Extract data
    methods = list(results.keys())
    mean_ranks = np.array([results[m]['mean_rank'] for m in methods])
    best_counts = np.array([results[m]['rank_counts'].get(0, 0) for m in methods])

    # Sort by mean rank (ascending = better)
    order = np.argsort(mean_ranks)
    methods = [methods[i] for i in order]
    mean_ranks = mean_ranks[order]
    best_counts = best_counts[order]

    num_methods = len(methods)
    cd = nemenyi_cd(num_methods, num_datasets, alpha)

    # ---- Create figure ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(wspace=0.4)

    # --- (1) Critical Difference Diagram ---
    ax = axs[0]
    min_rank = np.floor(mean_ranks.min() - 0.5)
    max_rank = np.ceil(mean_ranks.max() + 0.5)

    # Draw number line
    ax.hlines(0, min_rank, max_rank, color='black', lw=1.2)
    ax.set_xlim(max_rank, min_rank)  # invert x-axis (lower rank = better)
    ax.set_ylim(-1.2, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_title(f"Critical Difference Diagram (α={alpha})")

    # Plot points + labels (above the line)
    for r, method in zip(mean_ranks, methods):
        ax.vlines(r, 0, 0.15, color='gray', lw=1)
        ax.plot(r, 0.15, 'o', color='C0', markersize=8)
        ax.text(r, 0.25, f"{method}\n({r:.2f})",
                ha='center', va='bottom', fontsize=9, rotation=45)

    # CD reference bar (top)
    ax.hlines(1.4, mean_ranks[0], mean_ranks[0] + cd, lw=2, color='black')
    ax.text(mean_ranks[0] + cd / 2, 1.5, f"CD = {cd:.2f}", ha='center', va='bottom', fontsize=9)

    # Find groups within CD
    groups = []
    i = 0
    while i < num_methods:
        group = [i]
        j = i + 1
        while j < num_methods and abs(mean_ranks[j] - mean_ranks[i]) <= cd:
            group.append(j)
            j += 1
        if len(group) > 1:
            groups.append(group)
        i += 1

    # Draw CD group bars BELOW the line
    y_group = -0.2
    for group in groups:
        x_start = mean_ranks[group[0]]
        x_end = mean_ranks[group[-1]]
        ax.hlines(y_group, x_start, x_end, lw=3, color='black')
        y_group -= 0.15  # stagger multiple groups slightly

    # --- (2) Bar chart of best counts ---
    ax2 = axs[1]
    ax2.barh(methods, best_counts, color='C1')
    for y, val in enumerate(best_counts):
        ax2.text(val + 0.1, y, str(val), va='center', fontsize=9)
    ax2.set_xlabel("Times ranked #0 (best)")
    ax2.set_title("Frequency of Best Performance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved CD diagram to: {save_path}")
    plt.show()


'''def nemenyi_cd(num_methods, num_datasets, alpha=0.05):
    """
    Compute Critical Difference for Nemenyi post-hoc test.
    Based on Demsar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets.
    """
    q_alpha = {
        0.10: [0, 1.65, 1.96, 2.10, 2.22, 2.33, 2.39, 2.44, 2.49, 2.52, 2.56],
        0.05: [0, 1.96, 2.24, 2.34, 2.43, 2.52, 2.56, 2.61, 2.65, 2.68, 2.72],
        0.01: [0, 2.57, 2.91, 3.05, 3.17, 3.27, 3.33, 3.39, 3.43, 3.47, 3.50]
    }
    q = q_alpha[alpha][min(num_methods, 10)]
    return q * sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))


def visualize_ranking_results_with_cd(results, num_datasets, alpha=0.05):
    """
    Visualize ranking results using:
      1. A Critical Difference (CD) diagram on a single number line
      2. A bar chart showing how many times each approach was best

    Parameters
    ----------
    results : dict
        {
            'approach_name': {
                'mean_rank': float,
                'rank_counts': {0: int, 1: int, ...}
            },
            ...
        }
    num_datasets : int
        Number of datasets over which ranks were computed.
    alpha : float
        Significance level (default = 0.05)
    """
    methods = list(results.keys())
    mean_ranks = np.array([results[m]['mean_rank'] for m in methods])
    best_counts = np.array([results[m]['rank_counts'].get(0, 0) for m in methods])

    # Sort by mean rank (lower is better)
    order = np.argsort(mean_ranks)
    methods = [methods[i] for i in order]
    mean_ranks = mean_ranks[order]
    best_counts = best_counts[order]

    num_methods = len(methods)
    cd = nemenyi_cd(num_methods, num_datasets, alpha)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(wspace=0.4)

    # --- (1) CD diagram on a single number line ---
    ax = axs[0]
    min_rank = np.floor(mean_ranks.min() - 0.5)
    max_rank = np.ceil(mean_ranks.max() + 0.5)

    # Base number line
    ax.hlines(0, min_rank, max_rank, color='black', lw=1.2)
    ax.set_xlim(max_rank, min_rank)  # invert X: lower rank = better
    ax.set_ylim(-1, 2)

    # Reference lines & points
    for r, method in zip(mean_ranks, methods):
        ax.vlines(r, 0, 0.15, color='gray', lw=1)
        ax.plot(r, 0.15, 'o', color='C0', markersize=8)
        ax.text(r, 0.25, f"{method}\n({r:.2f})", ha='center', va='bottom', fontsize=9, rotation=45)

    # CD bar at top
    ax.hlines(1.2, mean_ranks[0], mean_ranks[0] + cd, lw=2, color='black')
    ax.text(mean_ranks[0] + cd / 2, 1.3, f"CD = {cd:.2f}", ha='center', va='bottom', fontsize=9)

    # Find and draw non-significant groups
    groups = []
    i = 0
    while i < num_methods:
        group = [i]
        j = i + 1
        while j < num_methods and abs(mean_ranks[j] - mean_ranks[i]) <= cd:
            group.append(j)
            j += 1
        if len(group) > 1:
            groups.append(group)
        i += 1

    y_group = 0.6
    for group in groups:
        x_start = mean_ranks[group[0]]
        x_end = mean_ranks[group[-1]]
        ax.hlines(y_group, x_start, x_end, lw=3, color='black')
        y_group += 0.15

    ax.set_yticks([])
    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_title(f"Critical Difference Diagram (α={alpha})")

    # --- (2) Bar chart of best counts ---
    ax2 = axs[1]
    ax2.barh(methods, best_counts, color='C1')
    for y, val in enumerate(best_counts):
        ax2.text(val + 0.1, y, str(val), va='center')
    ax2.set_xlabel("Times ranked #0 (best)")
    ax2.set_title("Frequency of Best Performance")

    plt.tight_layout()
    plt.show()'''


            
'''def nemenyi_cd(num_methods, num_datasets, alpha=0.05):
    """
    Compute the Critical Difference for the Nemenyi test.
    """
    q_alpha = studentized_range.ppf(1 - alpha/2, num_methods, np.inf) / sqrt(2)
    cd = q_alpha * sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))
    return cd

def visualize_ranking_results_with_cd(results, num_datasets, alpha=0.05):
    """
    Visualize ranking results using:
      1. A Critical Difference (CD) diagram with significance bars
      2. A bar chart showing how many times each approach was best
    
    Parameters
    ----------
    results : dict
        {
            'approach_name': {
                'mean_rank': float,
                'rank_counts': {0: int, 1: int, ...}
            },
            ...
        }
    num_datasets : int
        Number of datasets over which ranks were computed.
    alpha : float
        Significance level (default = 0.05)
    """
    # Extract info
    methods = list(results.keys())
    mean_ranks = np.array([results[m]['mean_rank'] for m in methods])
    best_counts = np.array([results[m]['rank_counts'].get(0, 0) for m in methods])

    # Sort by mean rank (ascending: lower is better)
    order = np.argsort(mean_ranks)
    methods = [methods[i] for i in order]
    mean_ranks = mean_ranks[order]
    best_counts = best_counts[order]

    num_methods = len(methods)
    cd = nemenyi_cd(num_methods, num_datasets, alpha)

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(wspace=0.3)

    # --- (1) Critical Difference Diagram ---
    ax = axs[0]
    y_positions = np.arange(num_methods)

    # Base lines
    ax.hlines(y_positions, np.min(mean_ranks)-0.5, np.max(mean_ranks)+0.5, color='lightgray', linestyles='dotted')
    ax.plot(mean_ranks, y_positions, 'o', color='C0', markersize=9)

    # Text labels
    for y, method, rank in zip(y_positions, methods, mean_ranks):
        ax.text(rank + 0.05, y, f"{method} ({rank:.2f})", va='center', ha='left', fontsize=9)

    # Draw CD bar
    y_cd = -1
    ax.hlines(y_cd, mean_ranks[0], mean_ranks[0] + cd, lw=2, color='black')
    ax.text(mean_ranks[0] + cd / 2, y_cd - 0.2, f"CD = {cd:.2f}", ha='center', va='top', fontsize=9)

    # Find groups that are not significantly different
    groups = []
    i = 0
    while i < num_methods:
        group = [i]
        j = i + 1
        while j < num_methods and abs(mean_ranks[j] - mean_ranks[i]) <= cd:
            group.append(j)
            j += 1
        if len(group) > 1:
            groups.append(group)
        i += 1

    # Draw significance bars
    y_sig = num_methods - 0.5
    for group in groups:
        ax.hlines(y_sig, mean_ranks[group[0]], mean_ranks[group[-1]], lw=3, color='black')
        y_sig -= 0.3

    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_yticks([])
    ax.invert_xaxis()
    ax.set_title(f"Critical Difference Diagram (α={alpha})")

    # --- (2) Bar chart of best counts ---
    ax2 = axs[1]
    ax2.barh(methods, best_counts, color='C1')
    for y, val in enumerate(best_counts):
        ax2.text(val + 0.1, y, str(val), va='center')

    ax2.set_xlabel("Times ranked #0 (best)")
    ax2.set_title("Frequency of Best Performance")

    plt.tight_layout()
    plt.show()'''



dir_list = os.listdir()  
print(dir_list)
all_test_accuracies = {}
all_test_times = {}
for filename in dir_list:
    if 'msm_' in filename:
        print(filename)
        with open(filename, 'r') as file:
            data = file.read()
        file.close()
        print(data)
        data = eval(data)
        for test_set in data.keys():
            print(test_set)
            if "time" in filename:
                all_test_times[test_set] = data[test_set]
            if 'acc' in filename:
                all_test_accuracies[test_set] = data[test_set]
                
#print(all_test_accuracies)
print(all_test_times)
print(all_test_accuracies.keys())
print(all_test_times.keys())
all_classifiers = {}

for test_set in all_test_accuracies.keys():
    print(test_set)
    for transform_type in all_test_accuracies[test_set].keys():
        print(all_test_accuracies[test_set][transform_type].keys())
        for classifier_type in all_test_accuracies[test_set][transform_type].keys():
            all_classifiers[classifier_type] = []
       
classifier_list = list(all_classifiers.keys())
print(classifier_list)

classifier_acc_compare_xy = {}
for i in range(0, len(classifier_list)):
    classifier_acc_compare_xy[classifier_list[i]] = {}
    for j in range(i+1, len(classifier_list)):
        classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]] = [[],[]]
        for test_set in all_test_accuracies.keys():
            print(test_set)
            print(all_test_accuracies[test_set])
            x_val = 0
            y_val = 0
            for transform_type in all_test_accuracies[test_set].keys():
                print(all_test_accuracies[test_set][transform_type].keys())
                for classifier_type in all_test_accuracies[test_set][transform_type].keys():
                    if classifier_type == classifier_list[i]:
                        x_val = 1-all_test_accuracies[test_set][transform_type][classifier_type]['zero_one']
                    if classifier_type == classifier_list[j]:
                        y_val = 1-all_test_accuracies[test_set][transform_type][classifier_type]['zero_one']
            print(classifier_list[i], x_val)
            print(classifier_list[j], y_val)
            classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]][0].append(x_val)
            classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]][1].append(y_val)
print(classifier_acc_compare_xy)
x_labels = []
y_labels = []
titles = []
x_y_pairs = []
for x_key in classifier_acc_compare_xy.keys():
    print(x_key)
    for y_key in classifier_acc_compare_xy[x_key].keys():
        x_labels.append(x_key)
        y_labels.append(y_key)
        titles.append(x_key + "/" + y_key)
        x_y_pairs.append(classifier_acc_compare_xy[x_key][y_key])
for i in range(0, len(x_labels)):
    print(x_labels[i], y_labels[i], x_y_pairs[i])
    print(x_key, classifier_acc_compare_xy[x_key].keys())
    
scatter_grid(x_y_pairs, titles)
all_classifier_ranks = {}

for key in all_classifiers.keys():
    all_classifier_ranks[key] = []

for test_set in all_test_accuracies.keys():
    element_acc_pairs = []
    accuracies = []
    for transform_type in all_test_accuracies[test_set].keys():
        print(all_test_accuracies[test_set][transform_type].keys())
        for classifier_type in all_test_accuracies[test_set][transform_type].keys():
            element_acc_pairs.append((classifier_type, 1-all_test_accuracies[test_set][transform_type][classifier_type]['zero_one']))
            accuracies.append(1-all_test_accuracies[test_set][transform_type][classifier_type]['zero_one'])
    element_acc_pairs = np.array(element_acc_pairs)
    print(accuracies)
    sorted_indices = rank_floats(accuracies)

    print(element_acc_pairs)
    print(sorted_indices)
    for i in range(0, len(element_acc_pairs)):
        print(element_acc_pairs[i], sorted_indices[i])
        all_classifier_ranks[element_acc_pairs[i][0]].append(sorted_indices[i])
print(all_classifier_ranks)

for_visualizer = {}
for key in all_classifier_ranks.keys():
    print(key, sum(all_classifier_ranks[key])/len(all_classifier_ranks[key]))
    mean_rank = sum(all_classifier_ranks[key])/len(all_classifier_ranks[key])
    rank_dict = {}
    for i in range(0, len(all_classifier_ranks)):
        rank_dict[i] = all_classifier_ranks[key].count(i)
    print(rank_dict)
    for_visualizer[key] = {'mean_rank': mean_rank, 'rank_counts':rank_dict}

visualize_ranking_results_with_cd(for_visualizer, num_datasets=len(all_test_accuracies.keys()))

classifier_acc_compare_xy = {}


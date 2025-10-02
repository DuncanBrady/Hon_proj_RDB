#File containing functions for visualising dists of data, including overall dists and zero-inflated dists
import numpy as np
import matplotlib.pyplot as plt


def plot_overall_hist(data, labels={"title": "Overall Data Distribution", "xlabel": "Value", "ylabel": "Count"}, num_bins=100):
    # Plots the distribution of the entire datasets values
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    median_val = np.median(data)

    # create figure and axiss
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # plot histogram on first subplot
    counts, bins, patches = ax1.hist(data.flatten(), bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title(labels.get("title"), fontsize=16)
    ax1.set_xlabel(labels.get("xlabel"), fontsize=14)
    ax1.set_ylabel(labels.get("ylabel"), fontsize=14)
    ax1.set_yscale('log')  # log scale for y-axis
    ax1.grid(alpha=0.75) 
    # add vertical lines for mean and median
    ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
    # add vertical lines for min and max
    ax1.axvline(min_val, color='orange', linestyle='dashed', linewidth=1, label=f'Min: {min_val:.2f}')
    ax1.axvline(max_val, color='purple', linestyle='dashed', linewidth=1, label=f'Max: {max_val:.2f}')
    ax1.legend()

    ## Add annotations for mean, median, min, max
    fig.status_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
    
    return fig

def plot_zero_inflated_dist(data):
    pass   

def plot_non_zero_dist(data):
    # isolate the non-zero values and plot their distribution
    non_zero_data = data[data != 0]
    plot_overall_dist(non_zero_data)


def plot_bin_dist(data, labels):
    pass   

def plot_density_dist(data, labels):
    pass

def violin_plot_dist(data, labels):
    pass   
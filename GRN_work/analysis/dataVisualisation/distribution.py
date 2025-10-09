#File containing functions for visualising dists of data, including overall dists and zero-inflated dists
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(data, labels={"title": "Overall Data Distribution", "xlabel": "Value", "ylabel": "Count"}, num_bins=100, log_scale=False):
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
    if log_scale:
        ax1.set_yscale('log')  # log scale for y-axis
    ax1.grid(alpha=0.75) 
    # add vertical lines for mean and median
    ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')
    # add vertical lines for min and max
    ax1.axvline(min_val, color='orange', linestyle='dashed', linewidth=1, label=f'Min: {min_val:.2f}')
    ax1.axvline(max_val, color='purple', linestyle='dashed', linewidth=1, label=f'Max: {max_val:.2f}')
    ax1.legend()

    # Add annotations for mean, median, min, and max and save the plot
    textstr = '\n'.join((
        f'Mean: {mean_val:.2f}',
        f'Median: {median_val:.2f}',
        f'Min: {min_val:.2f}',
        f'Max: {max_val:.2f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.75, 0.95, textstr, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)            
    plt.tight_layout()
    return fig

def plot_zero_inflated_dist(data, labels = {"title": "Percentage of Zero Values", "xlabel": "Value", "ylabel": "Count"}):
    non_zero_data  = data[data != 0]
    zero_count = data.size - non_zero_data.size
    zero_percentage = (zero_count / data.size) * 100
    non_zero_percentage = 100 - zero_percentage
    labels = ["Non-zero values", "Zero values"]
    sizes = [non_zero_percentage, zero_percentage]
    fig = plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title("Proportion of Zero vs Non-Zero Values in Dataset", fontsize=16)
    return fig

def plot_non_zero_dist(data):
    non_zero_data  = data[data != 0]
    labels = {"title": "Non-Zero Data Distribution", "xlabel": "Value", "ylabel": "Count"}
    non_zero_plot = plot_hist(non_zero_data, labels=labels)
    return non_zero_plot


def plot_bin_dist(data, labels, bin_count = None):  
    if bin_count is None:
        print("No bin count provided, please provide a bin count")
        bin_count = int(input("Enter bin count: "))
        attempts = 0  
        while not isinstance(bin_count, int) or bin_count <= 0 and attempts < 3:
            print("Invalid input. Please enter a positive integer for bin count.")
            bin_count = int(input("Enter bin count: ")) 
            attempts += 1
        if attempts == 3:
            print("Invalid bin count, moving on with default of 7")
            bin_count = 7
    hist_plot = plot_hist(data, labels=labels, num_bins=bin_count)
    return hist_plot
    pass   


def plot_density_dist(data, labels):
    pass

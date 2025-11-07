# Script with command line options to visualise a given dataset and save the plots
# Functions are designed to also be called by other files.
import numpy as np
import argparse
import os
import distribution


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise dataset distributions and save plots.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plots.')
    parser.add_argument('--plot_type', type=str, choices=['overall', 'zero_inflated', 'bin', 'density', 'violin'], required=True, help='Type of plot to generate.')
    return parser.parse_args()

def load_data(data_path, transpose=False):
    data = np.load(data_path)
    if transpose:
        data = data.T
    return data

def visualise_data(data, methods = ['overall'], overlays = []):
    return None 

def save_plots(output_dir, plot, plot_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, plot_name)
    plot.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def overlay_plot(data, methods):
    return None

def main():
    args = parse_args()
    data = load_data()
    distribution.plot_hist(None)
    return 0

if __name__ == "__main__":
    main()
import numpy as np

def term_freq_bin(data, num_bins):
      # Separate zero and non-zero values
    zero_mask = np.nonzero(data)
    non_zero_data = data

    if len(non_zero_data) == 0:
        return [data]
    else:
        print(non_zero_data.shape, non_zero_data)

    # Calculate percentiles for binning non-zero values
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(non_zero_data, percentiles)

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    print("The unique bin edges are:", bin_edges)

    #Using the bin edges, mask the original data values into bin number
    binned_data = np.digitize(non_zero_data, bin_edges)
    return binned_data

def k_means_bin(data, num_bins):
    # Get the total number of genes
    num_genes = data.shape[0]
    # Randomly select initial centroids
    centroids = data[np.random.choice(num_genes, num_bins, replace=False)]
    # Assign genes to the nearest centroid
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    # Create bins based on labels
    bins = [data[labels == i] for i in range(num_bins)]
    return bins


# bins = [0,1,2,3,4]
# test_data = np.random.randint(0, 20, size=10)
# print(test_data)
# np.histogram(test_data,bins)
# # Get histogram data
# hist_counts, bin_edges = np.histogram(test_data, bins)

# # Plot the histogram
# plt.figure(figsize=(8, 5))
# plt.bar(range(len(hist_counts)), hist_counts, align='center', alpha=0.7)
# plt.xticks(range(len(hist_counts)), [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)])
# plt.xlabel('Bin Range')
# plt.ylabel('Count')
# plt.title('Histogram of Test Data')
# plt.grid(axis='y', alpha=0.3)
# plt.show()
 

 # Print the shape of the data to understand its structure
# print("Data shape:", data.shape)

# # Calculate min and max expression values across the dataset
# min_expression = np.min(data)
# max_expression = np.max(data)
# print(f"Min expression value: {min_expression}")
# print(f"Max expression value: {max_expression}")

# # Calculate max expression value for each cell (assuming rows are cells)
# max_per_cell = np.max(data, axis=1)

# # Calculate variance of max expression values between cells
# variance_of_max_expressions = np.var(max_per_cell)
# print(f"Variance of max expression values between cells: {variance_of_max_expressions}")

# # Calculate mean of max expression values between cells
# mean_of_max_expressions = np.mean(max_per_cell)
# print(f"Average of max expression values between cells: {mean_of_max_expressions}")

# # Calculate average variance across all genes
# gene_variances = np.var(data, axis=0)
# avg_gene_variance = np.mean(gene_variances)
# print(f"Average variance of gene expression across cells: {avg_gene_variance}")
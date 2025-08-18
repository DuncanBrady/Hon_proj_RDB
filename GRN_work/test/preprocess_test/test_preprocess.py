import pytest
import numpy as np
from GRN_work.src.preprocess.binning import term_freq_bin


class TestTermFreqBin:
    def test_one_dimensional_array(self):
        data = np.array([1, 2, 3, 4, 5])
        n_bins = 3
        result = term_freq_bin(data, n_bins)
        print(result)
        expected = np.array([1, 1, 2, 3, 3])
        print("Original Data:", data)
        print("Expected:", expected)
        print("Result:", result)
        assert np.array_equal(result, expected)

    def test_two_dimensional_array(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        n_bins = 3
        result = term_freq_bin(data, n_bins)
        expected = np.array([[1, 1, 2], [2, 3, 3]])
        print("Original Data:", data)
        print("Expected:", expected)
        print("Result:", result)
        assert np.array_equal(result, expected)

    def test_float_array(self):
        data = np.array([0.0, 0.5, 1.0, 2.5, 3.1, 4.8, 5.9])
        n_bins = 3
        result = term_freq_bin(data, n_bins)
        expected = np.array([0, 1, 1, 2, 2, 3, 3])
        print("Original Data:", data)
        print("Expected:", expected)
        print("Result:", result)
        assert np.array_equal(result, expected)
    
    def test_zero_array(self):
        data = np.zeros_like(10, shape=(2, 10))
        n_bins = 3
        result = term_freq_bin(data, n_bins)
        expected = np.zeros_like(data)
        print("Original Data:", data)
        print("Expected:", expected)
        print("Result:", result)
        assert np.array_equal(result, expected)
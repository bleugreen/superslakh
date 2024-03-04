import numpy as np
import torch


def create_log_filter(num_bins, num_bands, device='cuda'):
    # Generate logarithmically spaced bins
    log_bins = np.logspace(np.log10(1), np.log10(num_bins), num=num_bands+1, base=10) - 1
    log_bins = np.floor(log_bins).astype(int)

    # Ensure all bins are unique and cover the entire range
    log_bins[-1] = num_bins  # Ensure the last bin correctly ends at num_bins
    if len(np.unique(log_bins)) < len(log_bins):
        # Handle potential duplicates after flooring by adjusting bins
        for i in range(1, len(log_bins)):
            if log_bins[i] <= log_bins[i-1]:
                log_bins[i] = log_bins[i-1] + 1

    filter_matrix = torch.zeros(num_bands, num_bins, device=device)
    for i in range(num_bands):
        start_bin = log_bins[i]
        end_bin = log_bins[i+1] if i < num_bands - 1 else num_bins
        filter_matrix[i, start_bin:end_bin] = 1 / max(1, (end_bin - start_bin))

    return filter_matrix


def apply_log_filter(stft_output, filter_matrix):
    """
    Apply the logarithmic filter matrix to the Short-Time Fourier Transform (STFT) output.

    This function applies a precomputed logarithmic filter matrix to the STFT output of an audio signal
    to reduce its dimensionality and to capture the energy in logarithmically spaced frequency bands.

    Parameters
    ----------
    stft_output : torch.Tensor
        A tensor representing the STFT output with shape (batch_size, num_bins, num_frames), where
        num_bins is the number of frequency bins and num_frames is the number of time frames.
    filter_matrix : torch.Tensor
        A tensor representing the logarithmic filter matrix with shape (num_bands, num_bins), where
        num_bands is the number of logarithmically spaced frequency bands.

    Returns
    -------
    torch.Tensor
        A tensor representing the filtered STFT output with shape (batch_size, num_bands, num_frames).
        Each band contains the aggregated energy from the corresponding set of frequency bins.
    """
    stft_output_transposed = stft_output.transpose(1, 2)
    filtered_output_transposed = torch.matmul(stft_output_transposed, filter_matrix.T)
    filtered_output = filtered_output_transposed.transpose(1, 2)
    return filtered_output
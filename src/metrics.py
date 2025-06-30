import numpy as np
from scipy.stats import wasserstein_distance
from scipy.integrate import cumulative_trapezoid

def calculate_rel_wasserstein_dist(t_vec, distr1, distr2):
    """
    Calculate the relative Wasserstein distance between two distributions.
    Normalizing is done by the expeceted time of the second distribution.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Relative Wasserstein distance.
    """
    
    # Normalize the distributions
    distr1_norm = distr1 / np.trapezoid(distr1, t_vec)
    distr2_norm = distr2 / np.trapezoid(distr2, t_vec)
    
    # Calculate the Wasserstein distance
    w_dist = wasserstein_distance(t_vec, t_vec, u_weights=distr1_norm, v_weights=distr2_norm)

    expected_t = np.trapezoid(t_vec * distr2_norm, t_vec)
    rel_w_dist = w_dist / expected_t if expected_t > 0 else np.nan
    
    return rel_w_dist

def calculate_rel_wasserstein2_dist(t_vec, distr1, distr2):
    """
    Calculate the relative 2nd-order Wasserstein distance (W2) between two distributions.
    Normalized by the expected tMRCA of distr2.

    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).

    Returns:
    float: Relative W2 distance.
    """
    # Normalize to get proper PDFs
    distr1_norm = distr1 / np.trapezoid(distr1, t_vec)
    distr2_norm = distr2 / np.trapezoid(distr2, t_vec)

    # Numerically approximate CDFs
    cdf1 = np.cumsum(distr1_norm)
    cdf1 /= cdf1[-1]
    cdf2 = np.cumsum(distr2_norm)
    cdf2 /= cdf2[-1]

    # Build inverse CDFs (quantile functions)
    q_levels = np.linspace(0, 1, len(t_vec))
    q1 = np.interp(q_levels, cdf1, t_vec)
    q2 = np.interp(q_levels, cdf2, t_vec)

    # Evaluate quantile differences
    w2_squared = np.mean((q1 - q2) ** 2)
    w2 = np.sqrt(w2_squared)

    # Normalize by mean of second distribution
    expected_t = np.trapezoid(t_vec * distr2_norm, t_vec)
    rel_w2 = w2 / expected_t if expected_t > 0 else np.nan

    return rel_w2


def calculate_rel_mode_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the relative mode shift between two distributions.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Relative mode shift.
    """
    
    mode1 = t_vec[np.argmax(distr1)]
    mode2 = t_vec[np.argmax(distr2)]
    
    rel_mode_shift = (mode1 - mode2) / mode2 if mode2 > 0 else np.nan
    if abs_value:
        rel_mode_shift = np.abs(rel_mode_shift)
    
    return rel_mode_shift

def calculate_mode_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the mode shift between two distributions.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Mode shift.
    """
    
    mode1 = t_vec[np.argmax(distr1)]
    mode2 = t_vec[np.argmax(distr2)]
    
    mode_shift = mode1 - mode2
    if abs_value:
        mode_shift = np.abs(mode_shift)
    
    return mode_shift

def calculate_rel_mean_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the relative mean shift between two distributions.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Relative mean shift.
    """
    
    mean1 = np.trapezoid(t_vec * distr1, t_vec) / np.trapezoid(distr1, t_vec) #normalize to get real probability distribtion
    mean2 = np.trapezoid(t_vec * distr2, t_vec) / np.trapezoid(distr2, t_vec)
    
    rel_mean_shift = (mean1 - mean2) / mean2 if mean2 > 0 else np.nan

    if abs_value:
        rel_mean_shift = np.abs(rel_mean_shift)
    
    return rel_mean_shift

def calculate_rel_median_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the relative median shift between two distributions.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Relative median shift.
    """

    # Normalize the distributions
    distr1 = distr1 / np.trapezoid(distr1, t_vec)
    distr2 = distr2 / np.trapezoid(distr2, t_vec)

    # Compute CDFs using cumulative trapezoid integration
    cdf1 = cumulative_trapezoid(distr1, t_vec, initial=0)
    cdf2 = cumulative_trapezoid(distr2, t_vec, initial=0)

    # Find medians where CDF crosses 0.5
    median1 = np.interp(0.5, cdf1, t_vec)
    median2 = np.interp(0.5, cdf2, t_vec)

    rel_median_shift = (median1 - median2) / median2 if median2 > 0 else np.nan

    if abs_value:
        rel_median_shift = np.abs(rel_median_shift)

    return rel_median_shift

def calculate_median_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the median shift between two distributions.
    
    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    
    Returns:
    float: Median shift.
    """

    # Normalize the distributions
    distr1 = distr1 / np.trapezoid(distr1, t_vec)
    distr2 = distr2 / np.trapezoid(distr2, t_vec)

    # Compute CDFs using cumulative trapezoid integration
    cdf1 = cumulative_trapezoid(distr1, t_vec, initial=0)
    cdf2 = cumulative_trapezoid(distr2, t_vec, initial=0)

    # Find medians where CDF crosses 0.5
    median1 = np.interp(0.5, cdf1, t_vec)
    median2 = np.interp(0.5, cdf2, t_vec)

    median_shift = median1 - median2

    if abs_value:
        median_shift = np.abs(median_shift)

    return median_shift

def calculate_rel_mean_shift_const(N, mu, L, abs_value=True):
    """
    Calculate the relative mean shift between two distributions, assuming constant population size.
    
    Parameters:
    N (int): Effective population size.
    mu (float): Mutation rate per base pair per generation.
    L (int): Length of the genome in base pairs.
    abs_value (bool): Whether to return the absolute value of the relative mean shift.
    
    
    Returns:
    float: Relative mean shift.
    """
    
    rel_mean_shift = -1 / (2 * N * mu * L + 1)
    if abs_value:
        rel_mean_shift = np.abs(rel_mean_shift)
    
    return rel_mean_shift

def calculate_coalescent_information_ratio_at_MAP(N, mu, L, posterior, t_vec, beta = None, N_low = None, t_bottleneck_start = None, t_bottleneck_end = None, population_model='constant', reverse_scale = False):
    """
    Calculate the coalescent information ratio (Omega) at the MAP estimate of the posterior.
    
    Parameters:
    N (int): Effective population size for constant population size, N_present for exponential growth and N_high for bottleneck.
    mu (float): Mutation rate per base pair per generation.
    L (int): Length of the genome in base pairs.
    beta (float, optional): Growth rate of the population for exponential growth models.
    N_low (int, optional): Population size during the bottleneck for bottleneck models.
    t_bottleneck_start (float, optional): Start time of the bottleneck backwards in time.
    t_bottleneck_end (float, optional): End time of the bottleneck backwards in time.
    population_model (str): Type of population model ('constant', 'exponential', 'bottleneck').
    posteriors (np.array): DataFrame containing posterior distributions with columns 't_MRCA' and 'posterior'.
    
    Returns:
    float: Coalescent information ratio at the MAP estimate.
    """
    
    # Get the MAP estimate
    t_map_estimate = t_vec[np.argmax(posterior)]

    if population_model == 'constant':
        omega = np.sqrt(1 / (1 + t_map_estimate / (2 * mu * L * N**2)))
    elif population_model == 'exponential':
        if beta is None:
            raise ValueError("beta must be provided for exponential growth models.")
        omega = np.sqrt((2 * mu* L / t_map_estimate) / ((2 * mu* L / t_map_estimate)+ beta**2 + 1/N**2))
    elif population_model == 'bottleneck':
        if N_low is None or t_bottleneck_start is None or t_bottleneck_end is None:
            raise ValueError("N_low, t_bottleneck_start, and t_bottleneck_end must be provided for bottleneck models.")
        if t_map_estimate < t_bottleneck_start:
            N_e = N
        elif t_bottleneck_start <= t_map_estimate <= t_bottleneck_end:
            N_e = N_low
        else:
            N_e = N
        omega = np.sqrt(1 / (1 + t_map_estimate / (2 * mu * L * N_e**2)))

    if reverse_scale:
        omega = 1 - omega
    
    return omega
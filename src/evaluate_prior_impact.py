import numpy as np
import pandas as pd
from src.prob_functions import likelihood_tMRCA_mutations, coalescent_prior, coalescent_prior_expN_present, coalescent_prior_bottleneck
from src.metrics import calculate_rel_mean_shift_const, calculate_rel_median_shift, calculate_median_shift, calculate_rel_mean_shift, calculate_rel_mode_shift, calculate_rel_wasserstein_dist, calculate_rel_wasserstein2_dist, calculate_coalescent_information_ratio_at_MAP, calculate_mode_shift


# possibly adapt such that t_max is calculated authomatically, e.g. time point when probability is below a certain threshold
def create_prior_influence_metric_df_const(N_values, alpha_values, base_mu, base_k, L, t_max):
    """
    Create a DataFrame to store prior influence metrics for different N and alpha values.
    
    Parameters:
    N_values (list): List of effective population sizes.
    alpha_values (list): List of alpha values that scale the variance of the likelihood function.
    base_mu (float): Base mutation rate.
    base_k (int): Base number of mutations.
    L (int): Length of the genome in base pairs.
    t_max (float): Maximum time point to consider.
    
    Returns:
    pd.DataFrame: DataFrame containing prior influence metrics.
    """
    
    data = []
    
    for N in N_values:
        for alpha in alpha_values:
            mu = base_mu * alpha
            k_mut = int(base_k * alpha) if int(base_k * alpha) > 0 else 1
            
            t_vec = np.linspace(0, t_max, t_max*10)
            likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_vec])
            priors = np.array([coalescent_prior(t, N) for t in t_vec])
            posteriors = likelihoods * priors
            
            rel_mean_shift = calculate_rel_mean_shift_const(L, mu, N, abs_value=True)
            rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
            rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
            omega_reverse = calculate_coalescent_information_ratio_at_MAP(N, mu, L, posteriors, t_vec, population_model='constant', reverse_scale=True)
            mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=True)
            
            data.append({
                'N': N,
                'alpha': alpha,
                'rel_mean_shift': rel_mean_shift,
                'rel_mode_shift': rel_mode_shift,
                'rel_median_shift': rel_median_shift,
                'mode_shift': mode_shift,
                'median_shift': median_shift,
                'rel_wasserstein_dist': rel_wasserstein_dist,
                'calculate_rel_wasserstein2_dist': rel_wasserstein_dist2,
                'reverse_coalescent_information_ratio_at_MAP': omega_reverse,
            })
    
    df = pd.DataFrame(data)
    
    return df



def create_prior_influence_metric_df_const_pathogen(pathogen_params):
    """
    Create a DataFrame to store prior influence metrics for different pathogens and different N values.
    
    Parameters:
    pathogen_params (dict): Dictionary containing pathogen parameters including N values, mutation rate, number of mutations, genome length, and maximum time.
    
    Returns:
    pd.DataFrame: DataFrame containing prior influence metrics.
    """
    
    data = []

    for name, params in pathogen_params.items():
        for N in params['N_values']:
            
            t_vec = np.linspace(0, params['t_max'], params['t_max']*10)
            likelihoods = np.array([likelihood_tMRCA_mutations(params['k'], params['mu'], t, params['L']) for t in t_vec])
            priors = np.array([coalescent_prior(t, N) for t in t_vec])
            posteriors = likelihoods * priors
            
            rel_mean_shift = calculate_rel_mean_shift_const(params['L'], params['mu'], N, abs_value=True)
            rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
            rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
            omega_reverse = calculate_coalescent_information_ratio_at_MAP(N, params['mu'], params['L'], posteriors, t_vec, population_model='constant', reverse_scale=True)
            mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=True)

            data.append({
                'pathogen': name,
                'N': N,
                'rel_mean_shift': rel_mean_shift,
                'rel_mode_shift': rel_mode_shift,
                'rel_median_shift': rel_median_shift,
                'mode_shift': mode_shift,
                'median_shift': median_shift,
                'rel_wasserstein_dist': rel_wasserstein_dist,
                'rel_wasserstein2_dist': rel_wasserstein_dist2,
                'reverse_coalescent_information_ratio_at_MAP': omega_reverse,
            })
    
    df = pd.DataFrame(data)
    
    return df

def create_prior_influence_metric_df_exp_pathogen(pathogen_params, N, beta_vec):
    """
    Create a DataFrame to store prior influence metrics for different pathogens and different exponential growth rate.
    
    Parameters:
    pathogen_params (dict): Dictionary containing pathogen parameters including mutation rate, number of mutations, genome length, and maximum time.
    N (int): Effective population size at present.
    beta_vec (float vec): Growth rate of the population (positive = growth, negative = decline).
    
    Returns:
    pd.DataFrame: DataFrame containing prior influence metrics.
    """
    
    data = []

    for name, params in pathogen_params.items():
        for beta in beta_vec:
            
            t_vec = np.linspace(0, params['t_max'], params['t_max']*10)
            likelihoods = np.array([likelihood_tMRCA_mutations(params['k'], params['mu'], t, params['L']) for t in t_vec])
            priors = np.array([coalescent_prior_expN_present(t, N, beta) for t in t_vec])
            posteriors = likelihoods * priors
            
            rel_mean_shift = calculate_rel_mean_shift(t_vec, posteriors, likelihoods, abs_value=False)
            rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
            rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
            omega_reverse = calculate_coalescent_information_ratio_at_MAP(N, params['mu'], params['L'], posteriors, t_vec, beta=beta, population_model='exponential', reverse_scale=True)
            mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=True)

            data.append({
                'pathogen': name,
                'N': N,
                'beta': beta,
                'rel_mean_shift': rel_mean_shift,
                'rel_mode_shift': rel_mode_shift,
                'rel_median_shift': rel_median_shift,
                'mode_shift': mode_shift,
                'median_shift': median_shift,
                'rel_wasserstein_dist': rel_wasserstein_dist,
                'rel_wasserstein2_dist': rel_wasserstein_dist2,
                'reverse_coalescent_information_ratio_at_MAP': omega_reverse,
            })
    
    df = pd.DataFrame(data)
    
    return df

def create_prior_influence_metric_df_bottleneck_pathogen(pathogen_params, N_high, N_low, t_bottleneck_start, t_bottleneck_end_vec):
    """
    Create a DataFrame to store prior influence metrics for different pathogens and different exponential growth rate.
    
    Parameters:
    pathogen_params (dict): Dictionary containing pathogen parameters including mutation rate, number of mutations, genome length, and maximum time.
    N (int): Effective population size at present.
    beta_vec (float vec): Growth rate of the population (positive = growth, negative = decline).
    
    Returns:
    pd.DataFrame: DataFrame containing prior influence metrics.
    """
    
    data = []

    for name, params in pathogen_params.items():
        for t_bottleneck_end in t_bottleneck_end_vec:
            
            t_vec = np.linspace(0, params['t_max'], params['t_max']*10)
            likelihoods = np.array([likelihood_tMRCA_mutations(params['k'], params['mu'], t, params['L']) for t in t_vec])
            priors = np.array([coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end) for t in t_vec])
            posteriors = likelihoods * priors
            
            rel_mean_shift = calculate_rel_mean_shift(t_vec, posteriors, likelihoods, abs_value=False)
            rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
            rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
            omega_reverse = calculate_coalescent_information_ratio_at_MAP(N_high, params['mu'], params['L'], posteriors, t_vec, N_low=N_low,
                                                                            t_bottleneck_start=t_bottleneck_start,
                                                                            t_bottleneck_end=t_bottleneck_end,
                                                                            population_model='bottleneck',
                                                                            reverse_scale=True)
            mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=True)
            median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=True)

            data.append({
                'pathogen': name,
                'N_high': N_high,
                'N_low': N_low,
                't_bottleneck_start': t_bottleneck_start,
                't_bottleneck_end': t_bottleneck_end,
                'rel_mean_shift': rel_mean_shift,
                'rel_mode_shift': rel_mode_shift,
                'rel_median_shift': rel_median_shift,
                'mode_shift': mode_shift,
                'median_shift': median_shift,
                'rel_wasserstein_dist': rel_wasserstein_dist,
                'rel_wasserstein2_dist': rel_wasserstein_dist2,
                'reverse_coalescent_information_ratio_at_MAP': omega_reverse,
            })
    
    df = pd.DataFrame(data)
    
    return df




def create_prior_influence_metric_df_exp(N_values, alpha_values, beta_values, base_mu, base_k, L, t_max=None):
    """
    Create a DataFrame to store prior influence metrics for an exponentially growing population.

    Parameters:
    N_values (list): List of effective population sizes at present.
    alpha_values (list): List of alpha values scaling the variance of the likelihood.
    beta_values (list): Growth rate of the population (positive = growth), backward in time population size is declining.
    base_mu (float): Base mutation rate.
    base_k (int): Base number of mutations.
    L (int): Genome length.
    t_max (float or None): Optional max time.

    Returns:
    pd.DataFrame: DataFrame of prior influence metrics.
    """

    data = []

    t_vec = np.linspace(0, t_max, int(t_max * 10))

    for alpha in alpha_values:
        mu = base_mu * alpha
        k_mut = max(int(base_k * alpha), 1)  # ensure at least 1 mutation
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_vec])
        for N in N_values:
            for beta in beta_values:
                priors = np.array([coalescent_prior_expN_present(t, N, beta) for t in t_vec])
                posteriors = likelihoods * priors
                # potentially normalize for proper probability interpretation, but not necessary since metrics normalize as well.
                # Calculate metrics
                rel_mean_shift = calculate_rel_mean_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
                rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
                reverse_omega = calculate_coalescent_information_ratio_at_MAP(N, mu, L, posteriors, t_vec, beta=beta,
                                                                            population_model='exponential',
                                                                            reverse_scale=True)

                data.append({
                    'N_present': N,
                    'alpha': alpha,
                    'beta': beta,
                    'rel_mean_shift': rel_mean_shift,
                    'rel_mode_shift': rel_mode_shift,
                    'rel_median_shift': rel_median_shift,
                    'rel_wasserstein_dist': rel_wasserstein_dist,
                    'calculate_rel_wasserstein2_dist': rel_wasserstein_dist2,
                    'r_coalescent_information_ratio_at_MAP': reverse_omega,
                })

    return pd.DataFrame(data)

def create_prior_influence_metric_df_pathogen_exp(N_beta_dict, mu, k, L, t_max):
    """
    Create a DataFrame to store prior influence metrics for an exponentially growing population.

    Parameters:
    N_values (list): List of effective population sizes at present.
    alpha_values (list): List of alpha values scaling the variance of the likelihood.
    beta_values (list): Growth rate of the population (positive = growth), backward in time population size is declining.
    base_mu (float): Base mutation rate.
    base_k (int): Base number of mutations.
    L (int): Genome length.
    t_max (float or None): Optional max time.

    Returns:
    pd.DataFrame: DataFrame of prior influence metrics.
    """

    data = []

    t_vec = np.linspace(0, t_max, int(t_max * 10))

    for N, beta_vec in N_beta_dict.items():
        for beta in beta_vec:
            likelihoods = np.array([likelihood_tMRCA_mutations(k, mu, t, L) for t in t_vec])
            priors = np.array([coalescent_prior_expN_present(t, N, beta) for t in t_vec])
            posteriors = likelihoods * priors

            rel_mean_shift = calculate_rel_mean_shift(t_vec, posteriors, likelihoods, abs_value=False)
            rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=False)
            rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
            rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
            reverse_omega = calculate_coalescent_information_ratio_at_MAP(N, mu, L, posteriors, t_vec, beta=beta,
                                                                        population_model='exponential',
                                                                        reverse_scale=True)
            mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=False)
            median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=False)

            data.append({
                'N_present': N,
                'beta': beta,
                'rel_mean_shift': rel_mean_shift,
                'rel_mode_shift': rel_mode_shift,
                'rel_median_shift': rel_median_shift,
                'rel_wasserstein_dist': rel_wasserstein_dist,
                'calculate_rel_wasserstein2_dist': rel_wasserstein_dist2,
                'r_coalescent_information_ratio_at_MAP': reverse_omega,
                'mode_shift': mode_shift,
                'median_shift': median_shift,
            })

    return pd.DataFrame(data)

def create_prior_influence_metric_df_bottleneck(N_high, N_low_values, t_bottleneck_end_vec, t_bottleneck_start, alpha_values, base_mu, base_k, L, t_max):
    """
    Create a DataFrame to store prior influence metrics for a constant population size with bottleneck.
    For simplicity, fix the constant population size before and after the bottleneck as well as the start time of the bottleneck.
    Bottleneck depth and duration are varied across the grid.

    Parameters:
    N_high (int): Effective population size before the bottleneck.
    N_low_values (list): List of effective population sizes during the bottleneck.
    t_bottleneck_end_vec (list): List of end times for the bottleneck backwards in time.
    t_bottleneck_start (int): Start time of the bottleneck backwards in time.
    alpha_values (list): List of alpha values scaling the variance of the likelihood.
    base_mu (float): Base mutation rate.
    base_k (int): Base number of mutations.
    L (int): Genome length.
    t_max (float): Max time to consider for tMRCA.

    Returns:
    pd.DataFrame: DataFrame of prior influence metrics.
    """

    data = []

    t_vec = np.linspace(0, t_max, int(t_max * 10))

    for alpha in alpha_values:
        mu = base_mu * alpha
        k_mut = max(int(base_k * alpha), 1)  # ensure at least 1 mutation
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_vec])
        for N_low in N_low_values:
            for t_bottleneck_end in t_bottleneck_end_vec:
                priors = np.array([coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end) for t in t_vec])
                posteriors = likelihoods * priors
                # potentially normalize for proper probability interpretation, but not necessary since metrics normalize as well.
                # Calculate metrics
                rel_mean_shift = calculate_rel_mean_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_median_shift = calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_mode_shift = calculate_rel_mode_shift(t_vec, posteriors, likelihoods, abs_value=False)
                rel_wasserstein_dist = calculate_rel_wasserstein_dist(t_vec, posteriors, likelihoods)
                rel_wasserstein_dist2 = calculate_rel_wasserstein2_dist(t_vec, posteriors, likelihoods)
                reverse_omega = calculate_coalescent_information_ratio_at_MAP(N_high, mu, L, posteriors, t_vec, N_low=N_low,
                                                                            t_bottleneck_start=t_bottleneck_start,
                                                                            t_bottleneck_end=t_bottleneck_end,
                                                                            population_model='bottleneck',
                                                                            reverse_scale=True)
                mode_shift = calculate_mode_shift(t_vec, posteriors, likelihoods, abs_value=False)
                median_shift = calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=False)

                data.append({
                    'N_high': N_high,
                    'N_low': N_low,
                    't_bottleneck_end': t_bottleneck_end,
                    't_bottleneck_start': t_bottleneck_start,
                    'alpha': alpha,
                    'rel_mean_shift': rel_mean_shift,
                    'rel_mode_shift': rel_mode_shift,
                    'mode_shift': mode_shift,
                    'median_shift': median_shift,
                    'rel_median_shift': rel_median_shift,
                    'rel_wasserstein_dist': rel_wasserstein_dist,
                    'rel_wasserstein2_dist': rel_wasserstein_dist2,
                    'r_coalescent_information_ratio_at_MAP': reverse_omega,
                })

    return pd.DataFrame(data)


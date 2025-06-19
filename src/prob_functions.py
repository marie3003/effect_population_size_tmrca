from math import exp, factorial

def likelihood_tMRCA_mutations(k_mut, mu, t_MRCA, L = 100):
    """
    Calculate the likelihood of observing k_mut mutations given a tMRCA and mutation rate.
    We only look at the tMRCA of two sequences sampled at the same time point.
    
    Parameters:
    k_mut (int): Number of mutations observed.
    mu (float): Mutation rate per base pair per generation.
    t_MRCA (float): calendar time to the most recent common ancestor, assume a generation time of 1.
    L (int): Length of the genome in base pairs (default is 100).
    
    Returns:
    float: Likelihood of observing k_mut mutations.
    """
   

    # Calculate the expected number of mutations
    expected_mutations = 2 * mu * L * t_MRCA
    
    # Calculate the likelihood using Poisson distribution
    likelihood = (expected_mutations ** k_mut) * exp(-expected_mutations) / factorial(k_mut)
    
    return likelihood


def coalescent_prior(t_MRCA, N):
    """
    Calculate the prior probability of a tMRCA given a population size N. For now assuming a constant population size.
    Assume fixed generation time g = 1
    
    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N (int): Effective population size.
    
    Returns:
    float: Prior probability of the tMRCA.
    """
    
    if t_MRCA < 0:
        return 0.0
    
    prior = 1 / N * exp(-t_MRCA / N)
    
    return prior


def coalescent_prior_expN(t_MRCA, N_0, beta, t_present):
    """
    Calculate the prior probability of a tMRCA given an exponentially growing population size N.
    
    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N (int): Effective population size.
    
    Returns:
    float: Prior probability of the tMRCA.
    """
    
    if t_MRCA < 0:
        return 0.0
    
    exp1 = exp(-beta * (t_present - t_MRCA))
    
    # Second exponential term
    inner_term = (exp(beta * t_MRCA) - 1)
    exp2 = exp(-(1 / (N_0 * beta)) * exp(-beta * t_present) * inner_term)
    
    prior = (1 / N_0) * exp1 * exp2
    
    return prior

def coalescent_prior_expN_present(t_MRCA, N_present, beta):
    """
    Calculate the prior probability of a tMRCA given an exponentially growing population size N.
    Scale time in a way such that the present time is t = 0.
    
    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N_present (int): Effective population size at the present time.
    beta (float): Growth rate of the population.
    
    Returns:
    float: Prior probability of the tMRCA.
    """
    
    if t_MRCA < 0:
        return 0.0
    
    exp1 = exp(beta * t_MRCA)
    
    # Second exponential term
    inner_term = (exp(beta * t_MRCA) - 1)
    exp2 = exp(-(1 / (N_present * beta)) * inner_term)
    
    prior = (1 / N_present) * exp1 * exp2
    
    return prior

def coalescent_prior_bottleneck(t_MRCA, N_high, N_low, t_bottleneck_start, t_bottleneck_end):
    """
    Calculate the prior probability of a tMRCA given a constant population size with a bottleneck.
    
    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N_high (int): Population size before the bottleneck.
    N_low (int): Population size during the bottleneck.
    t_bottleneck_start (float): Start time of the bottleneck backwards in time.
    t_bottleneck_end (float): End time of the bottleneck backwards in time.
    
    Returns:
    float: Prior probability of the tMRCA.
    """
    
    if t_MRCA < 0:
        return 0.0
    
    if t_MRCA < t_bottleneck_start:
        # Before the bottleneck
        prior = (1 / N_high) * exp(-t_MRCA / N_high)
    elif t_bottleneck_start <= t_MRCA <= t_bottleneck_end:
        # During the bottleneck
        prior = (1 / N_low) * exp(-(t_bottleneck_start / N_high + (t_MRCA - t_bottleneck_start) / N_low))
    else:
        # After the bottleneck
        prior = (1 / N_high) * exp(-(t_bottleneck_start / N_high + (t_bottleneck_end - t_bottleneck_start) / N_low + (t_MRCA - t_bottleneck_end) / N_high))
    
    return prior
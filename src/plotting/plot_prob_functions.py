import matplotlib.pyplot as plt
import numpy as np
import inspect

from src.prob_functions import likelihood_tMRCA_mutations, coalescent_prior, coalescent_prior_expN_present, coalescent_prior_expN, coalescent_prior_bottleneck
from src.metrics import calculate_coalescent_information_ratio_at_MAP, calculate_rel_mean_shift, calculate_rel_mean_shift_const, calculate_rel_mode_shift, calculate_rel_median_shift, calculate_rel_wasserstein_dist, calculate_rel_wasserstein2_dist, calculate_mode_shift


def plot_tMRCA_constN(N_values, alpha_values, base_mu, base_k, t_max, L, time_scale="days", metrics=None, title=None):

    t_values = np.linspace(0, t_max, t_max * 10)

    colors = ["#a6444f", "#397398", "#80557e"]  # Prior, Likelihood, Posterior

    fig, axs = plt.subplots(len(N_values), len(alpha_values), figsize=(20, 12), sharex=True, sharey=True)

    if len(N_values) == 1 and len(alpha_values) == 1:
        axs = np.array([[axs]])
    elif len(N_values) == 1:
        axs = np.expand_dims(axs, axis=0)
    elif len(alpha_values) == 1:
        axs = np.expand_dims(axs, axis=1)

    for row, N in enumerate(N_values):
        for col, alpha in enumerate(alpha_values):
            mu = base_mu * alpha
            k_mut = max(int(base_k * alpha), 1)

            likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
            priors = np.array([coalescent_prior(t, N) for t in t_values])
            posteriors = likelihoods * priors

            # Normalize to proper distributions
            #likelihoods /= np.trapezoid(likelihoods, t_values) if np.trapezoid(likelihoods, t_values) > 0 else 1
            #priors /= np.trapezoid(priors, t_values) if np.trapezoid(priors, t_values) > 0 else 1
            #posteriors /= np.trapezoid(posteriors, t_values) if np.trapezoid(posteriors, t_values) > 0 else 1

            # Normalize
            likelihoods /= np.max(likelihoods) if np.max(likelihoods) > 0 else 1
            priors /= np.max(priors) if np.max(priors) > 0 else 1
            posteriors /= np.max(posteriors) if np.max(posteriors) > 0 else 1

            ax = axs[row, col]
            ax.plot(t_values, priors, linestyle="--", color=colors[0], label="Prior")
            ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], label="Likelihood")
            ax.plot(t_values, posteriors, linewidth=2, color=colors[2], label="Posterior")
            ax.grid(True)

            if row == len(N_values) - 1:
                ax.set_xlabel(f"tMRCA [{time_scale}]")
            if col == 0:
                ax.set_ylabel(f"N = {int(N)}")
            if row == 0:
                ax.set_title(f"α = {alpha}")

            ax.legend(loc="upper right", fontsize=9)

            # Optional metrics
            metric_lines = []
            if metrics:
                available_args = {
                    "t_vec": t_values,
                    "distr1": posteriors,
                    "distr2": likelihoods,
                    "N": N,
                    "mu": mu,
                    "L": L,
                    "posterior": posteriors,
                }

                for metric_fn, label, extra_kwargs in metrics:
                    try:
                        # Introspect function signature
                        sig = inspect.signature(metric_fn)
                        accepted_args = sig.parameters.keys()

                        # Filter only needed arguments
                        filtered_args = {k: v for k, v in available_args.items() if k in accepted_args}
                        if extra_kwargs:
                            filtered_args.update(extra_kwargs)

                        val = metric_fn(**filtered_args)

                        if isinstance(val, (float, int)):
                            metric_lines.append(f"{label} = {val:.2f}")
                        elif isinstance(val, str):
                            metric_lines.append(f"{label}: {val}")
                        elif isinstance(val, list):
                            metric_lines.extend(val)
                    except Exception as e:
                        metric_lines.append(f"{label}: error")

            if metric_lines:
                ax.text(
                    0.05, 0.95,
                    "\n".join(metric_lines),
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
                )
    if title == None:
        title = "Prior, Likelihood, and Posterior of tMRCA at varying N and α"
    plt.suptitle(title, fontsize=16)
    plt.show()


def plot_tMRCA_expN_present(mu, L, k_mut, N_present, beta_vec, max_tmrca, N = None, evaluation = 'likelihood', plot_comparison_to_const = True):
    """
    Plot the prior, likelihood, and posterior of tMRCA for different exponential growth rates (β).
    
    Parameters:
    mu (float): Mutation rate per base per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N (int): Effective population size.
    t_present (int): Present time point for exponential growth.
    N_0 (int): Initial population size for exponential growth.
    beta_vec (list): List of exponential growth rates.
    """
    

    t_values = np.linspace(0, max_tmrca, max_tmrca)

    colors = [
        "#a6444f",  # prior constant
        "#397398",  # likelihood
        "#80557e",  # posterior constant
        "#b5d2f2",  # prior expN
        "#d991b4",  # posterior expN
    ]

    # Set up subplots
    fig, axs = plt.subplots(1, len(beta_vec), figsize=(20, 5), sharey=True)

    for col, beta in enumerate(beta_vec):
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
        priors_expN = np.array([coalescent_prior_expN_present(t, N_present, beta) for t in t_values])
        posteriors_expN = likelihoods * priors_expN

        # Normalize all for visualization
        likelihoods /= np.max(likelihoods)
        priors_expN /= np.max(priors_expN)
        posteriors_expN /= np.max(posteriors_expN)

        ax = axs[col]
        ax.plot(t_values, likelihoods, label="Likelihood", linestyle=":", color=colors[1], alpha = 0.8)
        ax.plot(t_values, priors_expN, label="Prior (exp N)", linestyle="-.", color=colors[3], alpha = 0.8)
        ax.plot(t_values, posteriors_expN, label="Posterior (exp N)", linewidth=2, color=colors[4], alpha = 0.8)

        if plot_comparison_to_const:
            priors = np.array([coalescent_prior(t, N) for t in t_values])
            posteriors = likelihoods * priors
            #normalize
            priors /= np.max(priors)
            posteriors /= np.max(posteriors)
            #plot
            ax.plot(t_values, priors, label="Prior (const N)", linestyle="--", color=colors[0], alpha = 0.8)
            ax.plot(t_values, posteriors, label="Posterior (const N)", linewidth=2, color=colors[2], alpha = 0.8)
        
        ax.set_title(f"β = {beta}")

        ax.set_xlabel("tMRCA (generations)")
        if col == 0:
            ax.set_ylabel("Probability (normalized)")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

        # Compares posterior based on exponential growth prior to likelihood (uniform prior)

        if evaluation == 'likelihood':
            rel_w_dist = calculate_rel_wasserstein_dist(t_values, posteriors_expN, likelihoods)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
        elif evaluation == 'posterior_const_prior':
            rel_w_dist = calculate_rel_wasserstein_dist(t_values, posteriors_expN, posteriors)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, posteriors, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, posteriors, abs_value=False)
        ax.text(
            0.05, 0.95,
            f"W. dist. = {rel_w_dist:.2f}\nRel. mean shift = {rel_mean_shift:.2f}\nRel. mode shift = {rel_mode_shift:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
        )

    plt.suptitle(f"Prior, Likelihood, and Posterior of tMRCA for Different Exponential Growth Rates (β) with N_present = {int(N_present)}")
    plt.show()

def plot_tMRCA_expN(mu, L, k_mut, t_present, N_0, beta_vec, max_tmrca, varying_N = False, N = None, N_vec = None, evaluation = 'likelihood'):
    """
    Plot the prior, likelihood, and posterior of tMRCA for different exponential growth rates (β).
    
    Parameters:
    mu (float): Mutation rate per base per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N (int): Effective population size.
    t_present (int): Present time point for exponential growth.
    N_0 (int): Initial population size for exponential growth.
    beta_vec (list): List of exponential growth rates.
    """
    

    t_values = np.linspace(0, max_tmrca, max_tmrca)

    colors = [
        "#a6444f",  # prior constant
        "#397398",  # likelihood
        "#80557e",  # posterior constant
        "#b5d2f2",  # prior expN
        "#d991b4",  # posterior expN
    ]

    # Set up subplots
    fig, axs = plt.subplots(1, len(beta_vec), figsize=(20, 5), sharey=True)

    for col, beta in enumerate(beta_vec):
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
        if varying_N:
            N_col = N_vec[col]
        else:
            N_col = N
        priors = np.array([coalescent_prior(t, N_col) for t in t_values])
        priors_expN = np.array([coalescent_prior_expN(t, N_0, beta, t_present) for t in t_values])
        
        posteriors = likelihoods * priors
        posteriors_expN = likelihoods * priors_expN

        # Normalize all for visualization
        likelihoods /= np.max(likelihoods)
        priors /= np.max(priors)
        posteriors /= np.max(posteriors)
        priors_expN /= np.max(priors_expN)
        posteriors_expN /= np.max(posteriors_expN)

        ax = axs[col]
        ax.plot(t_values, priors, label="Prior (const N)", linestyle="--", color=colors[0], alpha = 0.8)
        ax.plot(t_values, likelihoods, label="Likelihood", linestyle=":", color=colors[1], alpha = 0.8)
        ax.plot(t_values, posteriors, label="Posterior (const N)", linewidth=2, color=colors[2], alpha = 0.8)
        ax.plot(t_values, priors_expN, label="Prior (exp N)", linestyle="-.", color=colors[3], alpha = 0.8)
        ax.plot(t_values, posteriors_expN, label="Posterior (exp N)", linewidth=2, color=colors[4], alpha = 0.8)
        
        ax.set_title(f"β = {beta}, N = {N_col:.2f}" if varying_N else f"β = {beta}")

        ax.set_xlabel("tMRCA (generations)")
        if col == 0:
            ax.set_ylabel("Probability (normalized)")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

        # Compares posterior based on exponential growth prior to likelihood (uniform prior)

        if evaluation == 'likelihood':
            rel_w_dist = calculate_rel_wasserstein_dist(t_values, posteriors_expN, likelihoods)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
        elif evaluation == 'posterior_const_prior':
            rel_w_dist = calculate_rel_wasserstein_dist(t_values, posteriors_expN, posteriors)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, posteriors, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, posteriors, abs_value=False)
        ax.text(
            0.05, 0.95,
            f"W. dist. = {rel_w_dist:.2f}\nRel. mean shift = {rel_mean_shift:.2f}\nRel. mode shift = {rel_mode_shift:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
        )

    N_text = "N_0" if not varying_N else "N_present"
    plt.suptitle(f"Prior, Likelihood, and Posterior of tMRCA for Different Exponential Growth Rates (β) with N = {N_text}, Wasserstein dist. compared to {evaluation}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_tMRCA_bottleneck(mu, L, k_mut, N_high, N_low_vec, t_bottleneck_start, t_bottleneck_end_vec, t_max):
    """
    Plot the prior, likelihood, and posterior of tMRCA for varying bottleneck depths and durations.
    
    Parameters:
    mu (float): Mutation rate per base pair per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N_high (int): Population size before the bottleneck.
    N_low_vec (list): List of population sizes during the bottleneck.
    t_bottleneck_start (int): Start time of the bottleneck backwards in time.
    t_bottleneck_end_vec (list): List of end times for the bottleneck backwards in time.
    t_max (int): Maximum tMRCA to consider.
    """
    colors = ["#a6444f", "#397398", "#80557e", "#b5d2f2", "#d991b4"]

    t_values = np.linspace(0, t_max, t_max)
    
    fig, axs = plt.subplots(len(N_low_vec), len(t_bottleneck_end_vec), figsize=(20, 12), sharex=True, sharey=True)

    for row, N_low in enumerate(N_low_vec):
        for col, t_bottleneck_end in enumerate(t_bottleneck_end_vec):
            
            likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
            priors = np.array([coalescent_prior(t, N_high) for t in t_values])
            priors_bottleneck = np.array([
                coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end)
                for t in t_values
            ])
            posteriors = likelihoods * priors
            posteriors_bottleneck = likelihoods * priors_bottleneck

            # Normalize for visualization
            likelihoods /= np.max(likelihoods)
            priors /= np.max(priors)
            posteriors /= np.max(posteriors)
            priors_bottleneck /= np.max(priors_bottleneck)
            posteriors_bottleneck /= np.max(posteriors_bottleneck)

            ax = axs[row, col]
            line1, = ax.plot(t_values, priors, linestyle="--", color=colors[0], label="Prior (Const N)")
            line2, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], label="Likelihood")
            line3, = ax.plot(t_values, posteriors, color=colors[2], label="Posterior (Const N)")
            line4, = ax.plot(t_values, priors_bottleneck, linestyle="--", color=colors[3], label="Prior (Bottleneck)")
            line5, = ax.plot(t_values, posteriors_bottleneck, color=colors[4], label="Posterior (Bottleneck)")
            ax.grid(True)


            if row == len(N_low_vec) - 1:
                ax.set_xlabel("tMRCA (generations)")
            if col == 0:
                ax.set_ylabel(f"N_low = {N_low}")
            if row == 0:
                ax.set_title(f"t_end = {t_bottleneck_end}")

            rel_w_dist = calculate_rel_wasserstein_dist(t_values, posteriors_bottleneck, likelihoods)
            rel_w2_dist = calculate_rel_wasserstein2_dist(t_values, posteriors_bottleneck, likelihoods)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_bottleneck, likelihoods, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_bottleneck, likelihoods, abs_value=False)

            ax.text(
                0.05, 0.95,
                f"W. dist. = {rel_w_dist:.2f}\nW2. dist. = {rel_w2_dist:.2f} \nRel. mean shift = {rel_mean_shift:.2f}\nRel. mode shift = {rel_mode_shift:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
            )

            #if row == 0 and col == 0:
            #    ax.legend(fontsize=9, loc="upper right")

    # Final formatting
    fig.legend(
        handles=[line1, line2, line3, line4, line5],
        loc='upper right',
        bbox_to_anchor=(1, 0.98),
        fontsize=10
    )
    plt.suptitle("tMRCA Prior, Likelihood, and Posterior\nVarying Bottleneck Depth (rows) and Duration (columns)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

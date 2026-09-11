import matplotlib.pyplot as plt
import numpy as np
import inspect

from src.prob_functions import likelihood_tMRCA_mutations, coalescent_prior, coalescent_prior_expN_present, coalescent_prior_expN, coalescent_prior_bottleneck
from src.metrics import calculate_coalescent_information_ratio_at_MAP, calculate_rel_mean_shift, calculate_rel_mean_shift_const, calculate_rel_mode_shift, calculate_rel_median_shift, calculate_wasserstein_dist, calculate_wasserstein2_dist, calculate_mode_shift, calculate_median_shift

import inspect

def plot_tMRCA_constN(N_values, alpha_values, base_mu, base_k, t_max, L, time_scale="days", metrics=None, title=None, figsize=None, x_max=None, scale_mode="max", save_path=None, y_range=None, column_labels=None):
    """
    Parameters:
    ...
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of t_max
        (which sets the range used for metric calculation). Defaults to t_max if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    column_labels (list, optional): Custom column titles, one per entry in alpha_values (e.g. the
        mutational signal each alpha corresponds to, like "0.05 mutations per genome per year").
        Defaults to "α = {alpha}" if not set.
    """

    if column_labels is not None and len(column_labels) != len(alpha_values):
        raise ValueError("column_labels must have the same length as alpha_values.")

    t_values = np.linspace(0, t_max, t_max * 10)
    x_max = t_max if x_max is None else x_max

    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    colors = ["#a6444f", "#397398", "#80557e"]  # Prior, Likelihood, Posterior

    fig, axs = plt.subplots(len(N_values), len(alpha_values), figsize=(20, 12) if figsize is None else figsize, sharex=False, sharey=False)

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

            if scale_mode == "area":
                # Normalize to proper probability distributions (integrate to 1)
                likelihoods /= np.trapezoid(likelihoods, t_values) if np.trapezoid(likelihoods, t_values) > 0 else 1
                priors /= np.trapezoid(priors, t_values) if np.trapezoid(priors, t_values) > 0 else 1
                posteriors /= np.trapezoid(posteriors, t_values) if np.trapezoid(posteriors, t_values) > 0 else 1
            else:
                # Normalize by max, so curves are comparable in shape but not real densities
                likelihoods /= np.max(likelihoods) if np.max(likelihoods) > 0 else 1
                priors /= np.max(priors) if np.max(priors) > 0 else 1
                posteriors /= np.max(posteriors) if np.max(posteriors) > 0 else 1

            ax = axs[row, col]
            line_prior, = ax.plot(t_values, priors, linestyle="--", color=colors[0], label="Prior", linewidth=4)
            line_likelihood, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], label="Likelihood", linewidth=4)
            line_posterior, = ax.plot(t_values, posteriors, linewidth=4, color=colors[2], label="Posterior")
            ax.grid(True)
            ax.set_xlim(0, x_max)
            if y_range is not None:
                ax.set_ylim(*y_range)


            ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=22)
            y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
            ax.set_ylabel(y_label, fontsize=22)
            if col == 0:
                ax.annotate(
                    f"N = {int(N)}",
                    xy=(-0.28, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center", fontsize=24,
                )
            if row == 0:
                col_title = f"α = {alpha}" if column_labels is None else column_labels[col]
                ax.set_title(col_title, fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)

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
                    fontsize=16,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
                )
    if title == None:
        title = "Prior, Likelihood, and Posterior of tMRCA at varying N and α"
    fig.suptitle(title, fontsize=28)
    fig.legend(
        [line_prior, line_likelihood, line_posterior],
        ["Prior", "Likelihood", "Posterior"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=18,
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.2, w_pad=2.0)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_tMRCA_expN_present(mu, L, k_mut, N_present, beta_vec, max_tmrca, N = None, plot_comparison_to_const = True, time_scale = "days", x_max=None, scale_mode="max", save_path=None, metrics=None, y_range=None, figsize=None):
    """
    Plot the prior, likelihood, and posterior of tMRCA for different exponential growth rates (β).

    Parameters:
    mu (float): Mutation rate per base per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N (int): Effective population size for comparison to constant population size prior.
    t_present (int): Present time point for exponential growth.
    N_0 (int): Initial population size for exponential growth.
    beta_vec (list): List of exponential growth rates.
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of max_tmrca
        (which sets the range used for metric calculation). Defaults to max_tmrca if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    metrics (list of (callable, str, dict), optional): List of (metric_fn, label, extra_kwargs)
        tuples to compute and annotate in each panel, same convention as in plot_tMRCA_constN.
        Defaults to Wasserstein distance, reverse coalescent information ratio, and median shift
        if not set.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    """


    t_values = np.linspace(0, max_tmrca, max_tmrca)
    x_max = max_tmrca if x_max is None else x_max

    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    colors = [
        "#b7b5b5",  # prior constant
        "#397398",  # likelihood
        "#6c6c6c",  # posterior constant
        "#a6444f",  # prior expN
        "#80557e",  # posterior expN
    ]

    # Set up subplots
    fig, axs = plt.subplots(1, len(beta_vec), figsize=(20, 5) if figsize is None else figsize, sharey=False)
    if len(beta_vec) == 1:
        axs = [axs]

    for col, beta in enumerate(beta_vec):
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
        priors_expN = np.array([coalescent_prior_expN_present(t, N_present, beta) for t in t_values])
        posteriors_expN = likelihoods * priors_expN

        # Normalize all for visualization
        if scale_mode == "area":
            likelihoods /= np.trapezoid(likelihoods, t_values) if np.trapezoid(likelihoods, t_values) > 0 else 1
            priors_expN /= np.trapezoid(priors_expN, t_values) if np.trapezoid(priors_expN, t_values) > 0 else 1
            posteriors_expN /= np.trapezoid(posteriors_expN, t_values) if np.trapezoid(posteriors_expN, t_values) > 0 else 1
        else:
            likelihoods /= np.max(likelihoods)
            priors_expN /= np.max(priors_expN)
            posteriors_expN /= np.max(posteriors_expN)

        ax = axs[col]
        legend_handles = []
        legend_labels = []
        if plot_comparison_to_const:
            priors = np.array([coalescent_prior(t, N) for t in t_values])
            posteriors = likelihoods * priors
            #normalize
            if scale_mode == "area":
                priors /= np.trapezoid(priors, t_values) if np.trapezoid(priors, t_values) > 0 else 1
                posteriors /= np.trapezoid(posteriors, t_values) if np.trapezoid(posteriors, t_values) > 0 else 1
            else:
                priors /= np.max(priors)
                posteriors /= np.max(posteriors)
            #plot
            line, = ax.plot(t_values, priors, label="Prior (const N)", linestyle="--", color=colors[0], alpha = 0.8, linewidth=4)
            legend_handles.append(line); legend_labels.append("Prior (const N)")
            line, = ax.plot(t_values, posteriors, label="Posterior (const N)", linewidth=4, color=colors[2], alpha = 0.8)
            legend_handles.append(line); legend_labels.append("Posterior (const N)")

        line, = ax.plot(t_values, likelihoods, label="Likelihood", linestyle=":", color=colors[1], alpha = 0.8, linewidth=4)
        legend_handles.append(line); legend_labels.append("Likelihood")
        line, = ax.plot(t_values, priors_expN, label="Exp. growth Prior", linestyle="-.", color=colors[3], alpha = 0.8, linewidth=4)
        legend_handles.append(line); legend_labels.append("Exp. growth Prior")
        line, = ax.plot(t_values, posteriors_expN, label="Posterior", linewidth=4, color=colors[4], alpha = 0.8)
        legend_handles.append(line); legend_labels.append("Posterior")

        ax.set_title(f"β = {beta}", fontsize=24)

        ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=22)
        ax.set_xlim(0, x_max)
        if y_range is not None:
            ax.set_ylim(*y_range)
        if col == 0:
            y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
            ax.set_ylabel(y_label, fontsize=22)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Compares posterior based on exponential growth prior to likelihood (uniform prior)

        panel_metrics = metrics
        if panel_metrics is None:
            panel_metrics = [
                (calculate_wasserstein_dist, "W. dist.", {}),
                (calculate_coalescent_information_ratio_at_MAP, "1 - Ω", {'population_model': 'exponential', 'reverse_scale': True}),
                (calculate_median_shift, "Median shift", {'abs_value': False}),
            ]

        available_args = {
            "t_vec": t_values,
            "distr1": posteriors_expN,
            "distr2": likelihoods,
            "N": N_present,
            "mu": mu,
            "L": L,
            "beta": beta,
            "posterior": posteriors_expN,
        }

        metric_lines = []
        for metric_fn, label, extra_kwargs in panel_metrics:
            try:
                sig = inspect.signature(metric_fn)
                accepted_args = sig.parameters.keys()
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
                fontsize=16,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
            )

    #fig.suptitle(f"Prior, Likelihood, and Posterior of tMRCA for Different Exponential Growth Rates (β) with N_present = {int(N_present)}", fontsize=28)
    fig.legend(
        legend_handles, legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=18,
        ncol=len(legend_labels),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=1.2, w_pad=2.0)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_tMRCA_expN(mu, L, k_mut, t_present, N_0, beta_vec, max_tmrca, varying_N = False, N = None, N_vec = None, evaluation = 'likelihood', time_scale = "days", save_path=None):
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
        "#b7b5b5",  # prior constant
        "#397398",  # likelihood
        "#6c6c6c",  # posterior constant
        "#a6444f",  # prior expN
        "#80557e",  # posterior expN
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

        ax.set_xlabel(f"tMRCA [{time_scale}]")
        if col == 0:
            ax.set_ylabel("Probability (normalized)")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

        # Compares posterior based on exponential growth prior to likelihood (uniform prior)

        if evaluation == 'likelihood':
            w_dist = calculate_wasserstein_dist(t_values, posteriors_expN, likelihoods)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, likelihoods, abs_value=False)
        elif evaluation == 'posterior_const_prior':
            w_dist = calculate_wasserstein_dist(t_values, posteriors_expN, posteriors)
            rel_mean_shift = calculate_rel_mean_shift(t_values, posteriors_expN, posteriors, abs_value=False)
            rel_mode_shift = calculate_rel_mode_shift(t_values, posteriors_expN, posteriors, abs_value=False)
        ax.text(
            0.05, 0.95,
            f"W. dist. = {w_dist:.2f}\nRel. mean shift = {rel_mean_shift:.2f}\nRel. mode shift = {rel_mode_shift:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
        )

    N_text = "N_0" if not varying_N else "N_present"
    plt.suptitle(f"Prior, Likelihood, and Posterior of tMRCA for Different Exponential Growth Rates (β) with N = {N_text}, Wasserstein dist. compared to {evaluation}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_tMRCA_bottleneck(mu, L, k_mut, N_high, N_low_vec, t_bottleneck_start, t_bottleneck_end_vec, t_max, time_scale="days", figsize = None, x_max=None, scale_mode="max", save_path=None, max_cols=None, title=None, tile_labels=None, metrics=None, y_range=None):
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
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of t_max
        (which sets the range used for metric calculation). Defaults to t_max if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    max_cols (int, optional): Maximum number of subplots per row. If set, all (N_low,
        t_bottleneck_end) combinations are flattened and wrapped into a grid with at most
        max_cols columns per row, instead of the default N_low_vec x t_bottleneck_end_vec grid.
    title (str, optional): Figure suptitle. Pass "" for no title. Defaults to
        "tMRCA Prior, Likelihood, and Posterior\\nat varying bottleneck duration" if not set.
    tile_labels (list of str, optional): Custom per-subplot titles, overriding the default
        "t_end = ..." title. Without max_cols: one label per entry in t_bottleneck_end_vec
        (used as the shared column title). With max_cols: one label per (N_low,
        t_bottleneck_end) panel, in the same flattened order as the wrapped grid.
    metrics (list of (callable, str, dict), optional): List of (metric_fn, label, extra_kwargs)
        tuples to compute and annotate in each panel, same convention as in plot_tMRCA_constN.
        Defaults to Wasserstein distance, reverse coalescent information ratio, mode shift, and
        median shift if not set.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    """
    colors = [
        "#b7b5b5",  # prior constant
        "#397398",  # likelihood
        "#6c6c6c",  # posterior constant
        "#a6444f",  # prior bottleneck
        "#80557e",  # posterior bottleneck
    ]

    t_values = np.linspace(0, t_max, t_max)
    x_max = t_max if x_max is None else x_max

    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    panels = [(N_low, t_bottleneck_end) for N_low in N_low_vec for t_bottleneck_end in t_bottleneck_end_vec]

    if max_cols is not None:
        ncols = max_cols
        nrows = -(-len(panels) // max_cols)  # ceil division
    else:
        nrows, ncols = len(N_low_vec), len(t_bottleneck_end_vec)

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows) if figsize is None else figsize, squeeze=False)

    multiple_N_low = len(N_low_vec) > 1

    for idx, (N_low, t_bottleneck_end) in enumerate(panels):
        if max_cols is not None:
            row, col = divmod(idx, max_cols)
        else:
            row, col = divmod(idx, len(t_bottleneck_end_vec))
        if True:

            likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
            priors = np.array([coalescent_prior(t, N_high) for t in t_values])
            priors_bottleneck = np.array([
                coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end)
                for t in t_values
            ])
            posteriors = likelihoods * priors
            posteriors_bottleneck = likelihoods * priors_bottleneck

            # Normalize for visualization
            if scale_mode == "area":
                likelihoods /= np.trapezoid(likelihoods, t_values) if np.trapezoid(likelihoods, t_values) > 0 else 1
                priors /= np.trapezoid(priors, t_values) if np.trapezoid(priors, t_values) > 0 else 1
                posteriors /= np.trapezoid(posteriors, t_values) if np.trapezoid(posteriors, t_values) > 0 else 1
                priors_bottleneck /= np.trapezoid(priors_bottleneck, t_values) if np.trapezoid(priors_bottleneck, t_values) > 0 else 1
                posteriors_bottleneck /= np.trapezoid(posteriors_bottleneck, t_values) if np.trapezoid(posteriors_bottleneck, t_values) > 0 else 1
            else:
                likelihoods /= np.max(likelihoods)
                priors /= np.max(priors)
                posteriors /= np.max(posteriors)
                priors_bottleneck /= np.max(priors_bottleneck)
                posteriors_bottleneck /= np.max(posteriors_bottleneck)

            ax = axs[row, col]
            #line1, = ax.plot(t_values, priors, linestyle="--", color=colors[0], label="Prior (Const N)", linewidth=4)
            line2, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], label="Likelihood", linewidth=3)
            #line3, = ax.plot(t_values, posteriors, color=colors[2], label="Posterior (Const N)", linewidth=3)
            line4, = ax.plot(t_values, priors_bottleneck, linestyle="--", color=colors[3], label="Bottleneck Prior", linewidth=3)
            line5, = ax.plot(t_values, posteriors_bottleneck, color=colors[4], label="Posterior", linewidth=3)
            ax.grid(True)
            ax.set_xlim(0, x_max)
            if y_range is not None:
                ax.set_ylim(*y_range)
            ax.tick_params(axis='both', which='major', labelsize=15)


            ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=14)
            if max_cols is None:
                if col == 0:
                    y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
                    ax.set_ylabel(f"N_low = {N_low}\n{y_label}", fontsize = 14)
                if row == 0:
                    panel_title = f"t_end = {t_bottleneck_end}" if tile_labels is None else tile_labels[col]
                    ax.set_title(panel_title, fontsize=15)
            else:
                y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
                ax.set_ylabel(y_label, fontsize=14)
                if tile_labels is not None:
                    panel_title = tile_labels[idx]
                else:
                    panel_title = f"t_end = {t_bottleneck_end}"
                    if multiple_N_low:
                        panel_title = f"N_low = {N_low}, {panel_title}"
                ax.set_title(panel_title, fontsize=15)

            panel_metrics = metrics
            if panel_metrics is None:
                panel_metrics = [
                    (calculate_wasserstein_dist, "W. dist.", {}),
                    (calculate_coalescent_information_ratio_at_MAP, "1 - Ω", {'N_low': N_low, 't_bottleneck_start': t_bottleneck_start, 't_bottleneck_end': t_bottleneck_end, 'population_model': 'bottleneck', 'reverse_scale': True}),
                    (calculate_mode_shift, "Mode shift", {'abs_value': False}),
                    (calculate_median_shift, "Median shift", {'abs_value': False}),
                ]

            available_args = {
                "t_vec": t_values,
                "distr1": posteriors_bottleneck,
                "distr2": likelihoods,
                "N": N_high,
                "mu": mu,
                "L": L,
                "N_low": N_low,
                "t_bottleneck_start": t_bottleneck_start,
                "t_bottleneck_end": t_bottleneck_end,
                "posterior": posteriors_bottleneck,
            }

            metric_lines = []
            for metric_fn, label, extra_kwargs in panel_metrics:
                try:
                    sig = inspect.signature(metric_fn)
                    accepted_args = sig.parameters.keys()
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
                    0.97, 0.95,
                    "\n".join(metric_lines),
                    transform=ax.transAxes,
                    fontsize=13,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
                )

            #if row == 0 and col == 0:
            #    ax.legend(fontsize=9, loc="upper right")

    # Hide any unused trailing axes when panels don't fill the grid evenly
    for idx in range(len(panels), nrows * ncols):
        if max_cols is not None:
            row, col = divmod(idx, max_cols)
        else:
            row, col = divmod(idx, len(t_bottleneck_end_vec))
        axs[row, col].axis("off")

    # Final formatting
    fig.legend(
        handles=[line2, line4, line5],
        loc='upper right',
        bbox_to_anchor=(1, 1.0),
        fontsize=15,
        ncol=3,
    )
    if title is None:
        title = "tMRCA Prior, Likelihood, and Posterior\nat varying bottleneck duration"
    if title:
        fig.suptitle(title, fontsize=16)
        rect_top = 0.92
    else:
        rect_top = 0.94
    plt.tight_layout(rect=[0, 0, 1, rect_top])
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

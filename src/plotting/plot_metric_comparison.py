import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import numpy as np

############### Constant population size ################

def plot_error_metrics_const(evaluation_df, title, save_path=None):

    error_metrics = evaluation_df.columns[2:]
    num_metrics = len(error_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 6), sharex=True)

    # Plot each metric
    for i, metric in enumerate(error_metrics):
        ax = axes[i]
        sns.lineplot(data=evaluation_df, x="N", y=metric, hue="alpha", marker="o", ax=ax)
        ax.set_title(metric, fontsize=16)
        ax.set_xscale("log")
        ax.set_xlabel("N", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.legend(title="alpha", fontsize=12, title_fontsize='13')

    fig.suptitle(title, fontsize=20)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_metrics_by_alpha_const(evaluation_df, title, save_path=None):

    colors = [
        "#a6444f",  # reddish
        "#397398",  # dark blue
        "#80557e",  # purple
        "#b5d2f2",  # light blue
        "#d991b4",  # pink
        "#57a8b8",  # teal
        "#7394c2",  # mid blue
        "#7a7a7a"   # gray
    ]

    # Extract relevant columns
    error_metrics = evaluation_df.columns[2:]  # skip N and alpha
    alphas = sorted(evaluation_df["alpha"].unique())
    num_alphas = len(alphas)

    # Set up subplots
    fig, axes = plt.subplots(1, num_alphas, figsize=(7 * num_alphas, 6), sharey=True)

    # Plot for each alpha
    for i, alpha_val in enumerate(alphas):
        ax = axes[i]
        df_alpha = evaluation_df[evaluation_df["alpha"] == alpha_val]

        # Plot each error metric as a separate line
        for j, metric in enumerate(error_metrics):
            ax.plot(df_alpha["N"], df_alpha[metric], marker='o', label=metric, color = colors[j % len(colors)])

        ax.set_xscale("log")
        ax.set_title(f"alpha = {alpha_val}", fontsize=16)
        ax.set_xlabel("N", fontsize=14)
        if i == 0:
            ax.set_ylabel("Error", fontsize=14)
        ax.legend(loc='lower left', bbox_to_anchor=(0.18, 0.65), fontsize=10, title ="Metrics", title_fontsize='12')

    fig.suptitle(title, fontsize=20)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

################ Exponentially growing population size ################

def plot_error_metrics_exp(exp_evaluation_df, title, save_path=None):

    # Custom color palette
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Extract info
    error_metrics = exp_evaluation_df.columns[3:]  # Skip N_present, alpha, beta
    alphas = sorted(exp_evaluation_df["alpha"].unique())
    betas = sorted(exp_evaluation_df["beta"].unique())
    num_metrics = len(error_metrics)
    num_alphas = len(alphas)

    fig, axes = plt.subplots(num_alphas, num_metrics, figsize=(6 * num_metrics, 5 * num_alphas))

    # Ensure axes is always 2D
    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_metrics == 1:
        axes = np.expand_dims(axes, axis=1)


    for i, alpha_val in enumerate(alphas):
        df_alpha = exp_evaluation_df[exp_evaluation_df["alpha"] == alpha_val]
        for j, metric in enumerate(error_metrics):
            ax = axes[i, j]

            # Plot each beta separately using the specified color
            for k, beta_val in enumerate(betas):
                df_beta = df_alpha[df_alpha["beta"] == beta_val]
                ax.plot(df_beta["N_present"], df_beta[metric], marker='o',
                        label=f"β = {beta_val}", color=colors[k % len(colors)])

            # Plot zero-reference line
            ax.axhline(0, color='black', linestyle='--', linewidth=1)

            ax.set_xscale("log")
            ax.set_xlabel("N_present", fontsize=12)

            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}\n\n{metric}", fontsize=12)
            else:
                ax.set_ylabel(f"{metric}")
            if i == 0:
                ax.set_title(metric, fontsize=14)

            # Add legend to every plot
            ax.legend(title="Growth rate", fontsize=9, title_fontsize=10, loc="upper left")

    fig.suptitle(title, fontsize=22)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_error_metrics_exp_all_metrics_per_subplot(exp_evaluation_df, title, save_path=None):

    # Define custom colors for the error metrics
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Extract info
    error_metrics = exp_evaluation_df.columns[3:]  # Skip N_present, alpha, beta
    alphas = sorted(exp_evaluation_df["alpha"].unique())
    betas = sorted(exp_evaluation_df["beta"].unique())
    num_metrics = len(error_metrics)
    num_betas = len(betas)
    num_alphas = len(alphas)

    # Create subplots: rows = alpha, cols = beta
    fig, axes = plt.subplots(num_alphas, num_betas, figsize=(6 * num_betas, 5 * num_alphas))

    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_betas == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, alpha_val in enumerate(alphas):
        df_alpha = exp_evaluation_df[exp_evaluation_df["alpha"] == alpha_val]
        for j, beta_val in enumerate(betas):
            ax = axes[i, j]
            ax.grid(True)
            df_beta = df_alpha[df_alpha["beta"] == beta_val]

            # Plot each metric with its own color
            for k, metric in enumerate(error_metrics):
                ax.plot(df_beta["N_present"], df_beta[metric], marker='o',
                        label=metric, color=colors[k % len(colors)])

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xscale("log")

            if i == num_alphas - 1:
                ax.set_xlabel("N_present", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}", fontsize=12)
            if i == 0:
                ax.set_title(f"β = {beta_val}", fontsize=14)

            # Legend for error metrics
            ax.legend(title="Metric", fontsize=9, title_fontsize=10, loc="upper left")

    fig.suptitle(title, fontsize=22)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_error_heatmaps(exp_evaluation_df, alpha_fixed, title, save_path=None):

    # Filter by fixed alpha
    df = exp_evaluation_df[exp_evaluation_df["alpha"] == alpha_fixed].copy()

    if df.empty:
        raise ValueError(f"No data found for alpha = {alpha_fixed}")

    # Custom diverging colormap: red (negative) – purple (zero) – blue (positive)
    red_purple_blue = LinearSegmentedColormap.from_list(
        name="RedPurpleBlue",
        colors= ["#f4a6a6", "#d3a0c8", "#c2c2f0", "#a0c4f4", "#add8e6"],  # red → purple → blue
        N=256
    )


    metrics = df.columns[3:]  # Skip N_present, alpha, beta
    num_metrics = len(metrics)

    # Determine global min and max across all metrics for consistent color scale
    vmin = df[metrics].min().min()
    vmax = df[metrics].max().max()

    # Create subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharey=True)

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        pivot_table = df.pivot(index="beta", columns="N_present", values=metric)

        sns.heatmap(
            pivot_table,
            ax=axes[i],
            cmap= red_purple_blue,#"RdPu",  # red-purple-blue scheme
            annot=True,
            fmt=".3f",
            cbar_kws={'label': metric},
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax
        )

        axes[i].set_title(metric, fontsize=14)
        axes[i].set_xlabel("N_present")
        if i == 0:
            axes[i].set_ylabel("Growth rate (β)")

    fig.suptitle(f"{title} (alpha = {alpha_fixed})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_error_metrics_by_beta(exp_evaluation_df, title, neg = False, save_path=None):

    # Define custom colors for the error metrics
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Extract info
    error_metrics = exp_evaluation_df.columns[3:]  # Skip N_present, alpha, beta
    alphas = sorted(exp_evaluation_df["alpha"].unique())
    Ns = sorted(exp_evaluation_df["N_present"].unique())
    num_N = len(Ns)
    num_alphas = len(alphas)

    # Create subplots: rows = alpha, cols = N
    fig, axes = plt.subplots(num_alphas, num_N, figsize=(6 * num_N, 5 * num_alphas))

    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_N == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, alpha_val in enumerate(alphas):
        df_alpha = exp_evaluation_df.copy()[exp_evaluation_df["alpha"] == alpha_val]
        for j, N_val in enumerate(Ns):
            ax = axes[i, j]
            ax.grid(True)
            df_N = df_alpha.copy()[df_alpha["N_present"] == N_val]
            df_N.sort_values(by="beta", inplace=True)  # Sort by beta for consistent plotting

            # Plot each error metric
            for k, metric in enumerate(error_metrics):
                if neg == False:
                    ax.plot(df_N["beta"], df_N[metric], marker='o',
                        label=metric, color=colors[k % len(colors)])
                else:
                    ax.plot(-df_N["beta"], df_N[metric], marker='o',
                        label=metric, color=colors[k % len(colors)])

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            if neg == False:
                ax.set_xlabel("beta", fontsize=12)
            else:
                ax.set_xlabel("-beta", fontsize=12)
            ax.set_xscale("log")

            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}", fontsize=12)
            if i == 0:
                ax.set_title(f"N = {int(N_val)}", fontsize=14)

            ax.legend(title="Metric", fontsize=9, title_fontsize=10, loc="upper left")

    fig.suptitle(title, fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

################ Bottleneck population size ################

def plot_bottleneck_heatmaps(bottleneck_evaluation_df, title, save_path=None):

    bottleneck_evaluation_df = bottleneck_evaluation_df.copy()

    alphas = sorted(bottleneck_evaluation_df["alpha"].unique())
    metrics = bottleneck_evaluation_df.columns[5:]
    bottleneck_evaluation_df['bottleneck_duration'] = bottleneck_evaluation_df['t_bottleneck_end'] - bottleneck_evaluation_df['t_bottleneck_start']
    num_metrics = len(metrics)
    num_alphas = len(alphas)

    # Create pastel diverging colormap
    pastel_red_purple_blue = LinearSegmentedColormap.from_list(
        "PastelRedPurpleBlue",
        ["#f4a6a6", "#d3a0c8", "#c2c2f0", "#a0c4f4", "#add8e6"],
        N=256
    )

    # Global vmin/vmax across all plots for consistent color scale
    vmin = bottleneck_evaluation_df[metrics].min().min()
    vmax = bottleneck_evaluation_df[metrics].max().max()

    # Create subplots (rows = alpha, cols = metrics)
    fig, axes = plt.subplots(num_alphas, num_metrics, figsize=(6 * num_metrics, 5 * num_alphas))

    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_metrics == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, alpha_val in enumerate(alphas):
        df_alpha = bottleneck_evaluation_df[bottleneck_evaluation_df["alpha"] == alpha_val]

        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            pivot_table = df_alpha.pivot(index="bottleneck_duration", columns="N_low", values=metric)

            sns.heatmap(
                pivot_table,
                ax=ax,
                cmap=pastel_red_purple_blue,
                annot=True,
                fmt=".3f",
                #cbar_kws={'label': metric},
                linewidths=0.5,
                vmin=vmin,
                vmax=vmax
            )

            ax.set_xlabel("N_low")
            if i == 0:
                ax.set_title(metric, fontsize=14)
            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}\nBottleneck duration", fontsize=12)

    fig.suptitle(title, fontsize=24)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_bottleneck_error_metrics_by_depth(bottleneck_df, title, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Define custom colors for the error metrics
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Ensure bottleneck duration is present
    df = bottleneck_df.copy()
    if "bottleneck_duration" not in df.columns:
        df["bottleneck_duration"] = df["t_bottleneck_end"] - df["t_bottleneck_start"]

    error_metrics = df.columns.difference([
        "N_high", "N_low", "t_bottleneck_start", "t_bottleneck_end",
        "alpha", "bottleneck_duration"
    ])

    alphas = sorted(df["alpha"].unique())
    durations = sorted(df["bottleneck_duration"].unique())
    num_alphas = len(alphas)
    num_durations = len(durations)

    # Create subplots: rows = alpha, cols = bottleneck duration
    fig, axes = plt.subplots(num_alphas, num_durations, figsize=(6 * num_durations, 5 * num_alphas), sharex=True)

    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_durations == 1:
        axes = np.expand_dims(axes, axis=1)

    handles = []
    labels = []

    for i, alpha_val in enumerate(alphas):
        df_alpha = df[df["alpha"] == alpha_val]
        for j, duration in enumerate(durations):
            ax = axes[i, j]
            ax.grid(True)

            df_duration = df_alpha[df_alpha["bottleneck_duration"] == duration]

            for k, metric in enumerate(error_metrics):
                line, = ax.plot(df_duration["N_low"], df_duration[metric], marker='o',
                                label=metric, color=colors[k % len(colors)])
                if i == 0 and j == 0:  # collect legend from the first subplot only
                    handles.append(line)
                    labels.append(metric)

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel("N_low", fontsize=12)

            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}", fontsize=12)
            if i == 0:
                ax.set_title(f"Duration = {duration}", fontsize=14)

    # Global legend
    fig.legend(handles, labels, title="Metric", fontsize=14, title_fontsize=16,
               loc='upper right', bbox_to_anchor=(0.9, 1), borderaxespad=0)

    fig.suptitle(title, fontsize=22)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_bottleneck_error_metrics_by_duration(bottleneck_df, title, save_path=None):

    # Define custom colors for the error metrics
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Copy and compute bottleneck duration if needed
    df = bottleneck_df.copy()
    if "bottleneck_duration" not in df.columns:
        df["bottleneck_duration"] = df["t_bottleneck_end"] - df["t_bottleneck_start"]

    # Extract relevant info
    error_metrics = df.columns.difference([
        "N_high", "N_low", "t_bottleneck_start", "t_bottleneck_end",
        "alpha", "bottleneck_duration"
    ])
    alphas = sorted(df["alpha"].unique())
    depths = sorted(df["N_low"].unique())
    num_alphas = len(alphas)
    num_depths = len(depths)

    # Create subplot grid: rows = alpha, cols = N_low (depth)
    fig, axes = plt.subplots(num_alphas, num_depths, figsize=(6 * num_depths, 5 * num_alphas), sharex=True)

    if num_alphas == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_depths == 1:
        axes = np.expand_dims(axes, axis=1)

    handles = []
    labels = []

    for i, alpha_val in enumerate(alphas):
        df_alpha = df[df["alpha"] == alpha_val]
        for j, depth in enumerate(depths):
            ax = axes[i, j]
            ax.grid(True)

            df_depth = df_alpha[df_alpha["N_low"] == depth]

            for k, metric in enumerate(error_metrics):
                line, = ax.plot(df_depth["bottleneck_duration"], df_depth[metric], marker='o',
                                label=metric, color=colors[k % len(colors)])
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(metric)

            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel("Bottleneck duration", fontsize=12)

            if j == 0:
                ax.set_ylabel(f"alpha = {alpha_val}", fontsize=12)
            if i == 0:
                ax.set_title(f"N_low = {depth}", fontsize=14)

    # Global legend outside subplots
    fig.legend(handles, labels, title="Metric", fontsize=14, title_fontsize=16,
               loc='upper right', bbox_to_anchor=(0.9, 1), borderaxespad=0)

    fig.suptitle(title, fontsize=22)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


################ Pathogen comparison ################

def plot_metrics_by_pathogen_const(evaluation_df, error_metrics, title, save_path=None):
    """
    Plot all prior influence error metrics over N for each pathogen in separate subplots (columns).
    
    Parameters:
    - evaluation_df (pd.DataFrame): DataFrame with columns ['pathogen', 'N', ...metrics...]
    - error_metrics (list): List with all error metrics to plot.
    - title (str): Title of the full plot.
    """

    colors = [
        "#a6444f",  # reddish
        "#397398",  # dark blue
        "#80557e",  # purple
        "#b5d2f2",  # light blue
        "#d991b4",  # pink
        "#57a8b8",  # teal
        "#7394c2",  # mid blue
        "#7a7a7a"   # gray
    ]

    # Extract all metric columns (skip first two: pathogen, N)
    pathogens = sorted(evaluation_df["pathogen"].unique())
    num_pathogens = len(pathogens)

    # Set up subplots
    fig, axes = plt.subplots(1, num_pathogens, figsize=(7 * num_pathogens, 6), sharey = True)

    # Ensure axes is iterable even with 1 pathogen
    if num_pathogens == 1:
        axes = [axes]

    for i, pathogen_name in enumerate(pathogens):
        ax = axes[i]
        df_pathogen = evaluation_df[evaluation_df["pathogen"] == pathogen_name]

        # Plot each error metric
        for j, metric in enumerate(error_metrics):
            ax.plot(df_pathogen["N"], df_pathogen[metric],
                    marker='o', label=metric, color=colors[j % len(colors)])

        ax.set_xscale("log")
        ax.set_title(f"{pathogen_name}", fontsize=16)
        ax.set_xlabel("N", fontsize=14)
        if i == 0:
            ax.set_ylabel("Error metric value", fontsize=14)

        ax.legend(loc='upper right',
                  fontsize=10, title="Metrics", title_fontsize='12')
        ax.grid(True)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_error_metrics_by_beta_pathogen(exp_evaluation_df, error_metrics, title, neg = False, save_path=None):
    """
    Plot prior influence metrics over beta for different N_present values in separate subplots.

    Parameters:
    - exp_evaluation_df (pd.DataFrame): DataFrame with columns ['N_present', 'beta', ...metrics...]
    - error_metrics (list): Error metrics to show in plot.
    - title (str): Title for the entire plot figure.
    """

    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#d991b4",  # pink
        "#b5d2f2",  # light blue
        "#7394c2",  # mid blue
        "#397398",  # dark blue
        "#57a8b8",  # teal
        "#7a7a7a"   # gray
    ]

    # Extract metric columns (skip N_present and beta)
    Ns = sorted(exp_evaluation_df["N_present"].unique())
    num_N = len(Ns)

    # Create horizontal subplots (one per N)
    fig, axes = plt.subplots(1, num_N, figsize=(6 * num_N, 5), sharey=True)

    if num_N == 1:
        axes = [axes]

    for j, N_val in enumerate(Ns):
        ax = axes[j]
        df_N = exp_evaluation_df[exp_evaluation_df["N_present"] == N_val].copy()
        df_N.sort_values(by="beta", inplace=True)

        for k, metric in enumerate(error_metrics):
            if neg:
                ax.plot(-df_N["beta"], df_N[metric], marker='o',
                        label=metric, color=colors[k % len(colors)])
            else:
                ax.plot(df_N["beta"], df_N[metric], marker='o',
                    label=metric, color=colors[k % len(colors)])

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        if neg:
            ax.set_xlabel("-Beta", fontsize=14)
        else:
            ax.set_xlabel("Beta", fontsize=14)
        ax.set_xscale("log")
        ax.tick_params(labelleft=True)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True)
        ax.set_title(f"N = {int(N_val)}", fontsize=16)
        if j == 0:
            ax.set_ylabel("Error Metric Value", fontsize=14)
        ax.legend(title="Metric", fontsize=10, title_fontsize=11, loc="upper left")

    fig.suptitle(title, fontsize=22)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_mode_shift_pathogen_summary(df, title, save_path=None):
    # Define custom colors for each pathogen
    colors = [
        "#a6444f",  # reddish
        "#80557e",  # purple
        "#57a8b8",  # teal
        "#7394c2",  # mid blue
        "#7a7a7a"   # gray
    ]

    # Copy and prepare DataFrame
    plot_df = df.copy()
    pathogens = sorted(plot_df["pathogen"].unique())

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 6))

    handles = []
    labels = []

    for i, pathogen in enumerate(pathogens):
        df_path = plot_df[plot_df["pathogen"] == pathogen]
        df_path = df_path.sort_values("N")  # Ensure sorted by N

        ## shift day time scale to years
        if pathogen == "Influenza":
            df_path["mode_shift"] = df_path["mode_shift"] / 365

        line, = ax.plot(
            df_path["N"],
            df_path["mode_shift"],
            marker="o",
            label=pathogen,
            color=colors[i % len(colors)]
        )
        handles.append(line)
        labels.append(pathogen)

    ax.set_xscale("log")
    ax.set_xlabel("Population size N (log scale)", fontsize=12)
    ax.set_ylabel("Mode shift [years]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', ls='--', linewidth=0.5)

    # Add legend
    ax.legend(handles, labels, title="Pathogen", fontsize=10, title_fontsize=11)

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def _plot_metric_summary_const(ax, df_const, metric, colors, y_label, legend_title=None):
    df_const = df_const.copy()
    Ns_const = sorted(df_const["N"].unique())

    handles, labels = [], []
    for i, N in enumerate(Ns_const):
        df_N = df_const[df_const["N"] == N].sort_values("alpha")

        line, = ax.plot(
            df_N["alpha"],
            df_N[metric],
            marker="o",
            label=f"N = {N}",
            color=colors[i % len(colors)]
        )
        handles.append(line)
        labels.append(f"N = {N}")

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Constant population size", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(handles, labels, title="N" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def _plot_metric_summary_exp(ax, df_exp, metric, colors, y_label, legend_title=None):
    exp_df = df_exp.copy()
    Ns_exp = sorted(exp_df["N_present"].unique())

    for i, N in enumerate(Ns_exp):
        df_N = exp_df[exp_df["N_present"] == N].sort_values("beta")
        ax.plot(
            df_N["beta"],
            df_N[metric],
            marker="o",
            label=f"N = {N}",
            color=colors[i % len(colors)]
        )

    ax.set_xscale("log")
    ax.set_xlabel("Growth rate β (log scale)", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Exponential growth", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(title="N_present" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def _plot_metric_summary_declining(ax, df_declining, metric, colors, y_label, legend_title=None):
    declining_df = df_declining.copy()
    Ns_decline = sorted(declining_df["N_present"].unique())

    for i, N in enumerate(Ns_decline):
        df_N = declining_df[declining_df["N_present"] == N].copy()
        df_N["neg_beta"] = -df_N["beta"]
        df_N = df_N.sort_values("neg_beta")

        ax.plot(
            df_N["neg_beta"],
            df_N[metric],
            marker="o",
            label=f"N = {N}",
            color=colors[i % len(colors)]
        )

    ax.set_xscale("log")
    ax.set_xlabel("−β (log scale)", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Exponential decline", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(title="N_present" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def _plot_metric_summary_bottleneck(ax, df_bottleneck, metric, colors, y_label, legend_title=None):
    bottleneck_df = df_bottleneck.copy()
    bottleneck_df["bottleneck_duration"] = bottleneck_df["t_bottleneck_end"] - bottleneck_df["t_bottleneck_start"]
    N_lows = sorted(bottleneck_df["N_low"].unique())

    for i, N_low in enumerate(N_lows):
        df_Nlow = bottleneck_df[bottleneck_df["N_low"] == N_low].sort_values("t_bottleneck_end")
        ax.plot(
            df_Nlow["bottleneck_duration"],
            df_Nlow[metric],
            marker="o",
            label=f"N_low = {N_low}",
            color=colors[i % len(colors)]
        )

    ax.set_xlabel("Bottleneck duration [years]", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Population bottleneck", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(title="N_low" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def plot_metric_summary(title, df_const=None, df_exp=None, df_declining=None, df_bottleneck=None, metric="mode_shift", y_label=None, save_path=None, legend_title=None):
    # Define custom colors: shades of purple/pink, starting from the same purple used in
    # plot_pathogen_error_summary (so the matching line has the same color), getting lighter
    colors = [
        "#80557e", "#d991b4", "#f2c4d8", "#fbe6f0",
    ]

    # Exponential panel: purple needs to be the 2nd line, so add a darker shade before it
    # and drop the lightest shade
    exp_colors = [
        "#4b3866", "#80557e", "#d991b4", "#f2c4d8",
    ]

    y_label = f"{metric} [years]" if y_label is None else y_label

    # Only include tiles for the DataFrames that were provided
    panels = []
    if df_const is not None:
        panels.append((df_const, _plot_metric_summary_const, colors))
    if df_exp is not None:
        panels.append((df_exp, _plot_metric_summary_exp, exp_colors))
    if df_declining is not None:
        panels.append((df_declining, _plot_metric_summary_declining, colors))
    if df_bottleneck is not None:
        panels.append((df_bottleneck, _plot_metric_summary_bottleneck, colors))

    if not panels:
        raise ValueError("At least one of df_const, df_exp, df_declining, df_bottleneck must be provided.")

    # Create subplots
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (df, plot_fn, panel_colors) in zip(axes, panels):
        plot_fn(ax, df, metric, panel_colors, y_label, legend_title)
        ax.tick_params(axis='both', which='major', labelsize=18)

    # Set overall title
    #fig.suptitle(title, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

def plot_pathogen_error_summary(title, df_const=None, df_growth=None, df_decline=None, df_bottleneck=None, error_col="mode_shift", y_label=None, save_path=None, legend_position="in"):
    """
    legend_position (str): Where to place the legend. "in" (default) puts a legend in every
        subplot. "below" puts a single shared legend below the whole figure. "none" hides the
        legend entirely.
    """
    if legend_position not in ("in", "below", "none"):
        raise ValueError(f"legend_position must be 'in', 'below', or 'none', got {legend_position!r}")

    colors = [
        "#a6444f", "#80557e", "#7394c2", "#397398",
    ]

    y_label = error_col if y_label is None else y_label

    # Configs for each subplot, only included if the corresponding df is provided
    plot_configs = []

    if df_const is not None:
        plot_configs.append({
            "df": df_const,
            "x_col": "N",
            "x_label": "Population size N (log scale)",
            "title": "Constant population size",
            "xscale": "log",
        })

    if df_growth is not None:
        plot_configs.append({
            "df": df_growth,
            "x_col": "beta",
            "x_label": "Growth rate β (log scale)",
            "title": "Exponential growth",
            "xscale": "log",
        })

    if df_decline is not None:
        plot_configs.append({
            "df": df_decline.assign(neg_beta=lambda d: -d["beta"]),
            "x_col": "neg_beta",
            "x_label": "−β (log scale)",
            "title": "Exponential decline",
            "xscale": "log",
        })

    if df_bottleneck is not None:
        plot_configs.append({
            "df": df_bottleneck.assign(bottleneck_duration=lambda d: d["t_bottleneck_end"] - d["t_bottleneck_start"]),
            "x_col": "bottleneck_duration",
            "x_label": "Bottleneck duration [years]",
            "title": "Population bottleneck",
            "xscale": "linear",
        })

    if not plot_configs:
        raise ValueError("At least one of df_const, df_growth, df_decline, df_bottleneck must be provided.")

    pathogens = sorted(plot_configs[0]["df"]["pathogen"].unique())

    # Create subplots with shared y-axis
    fig, axes = plt.subplots(1, len(plot_configs), figsize=(7.5 * len(plot_configs), 6), sharey=True)
    if len(plot_configs) == 1:
        axes = [axes]

    handles, labels = [], []
    for ax, config in zip(axes, plot_configs):
        df = config["df"]
        x_col = config["x_col"]

        for i, pathogen in enumerate(pathogens):
            df_p = df[df["pathogen"] == pathogen].sort_values(x_col)
            line, = ax.plot(df_p[x_col], df_p[error_col], marker="o", color=colors[i % len(colors)], label=pathogen)
            if ax is axes[0]:
                handles.append(line)
                labels.append(pathogen)

        ax.set_xscale(config["xscale"])
        ax.set_xlabel(config["x_label"], fontsize=22)
        ax.set_title(config["title"], fontsize=24)
        ax.grid(True, which='both', ls='--', linewidth=0.5)
        ax.axhline(0, color="black", linestyle="dotted", linewidth=1)
        ax.tick_params(labelleft=True)  # ensure y-axis ticks are shown for each
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylabel(y_label, fontsize=22)
        if legend_position == "in":
            ax.legend(title="Mutations per genome per year", fontsize=17, title_fontsize=18)

    if legend_position == "below":
        fig.legend(
            handles, labels, title="Mutations per genome per year",
            loc="upper center", bbox_to_anchor=(0.5, 0.0),
            ncol=1, fontsize=18, title_fontsize=20
        )

    fig.suptitle(title, fontsize=28)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

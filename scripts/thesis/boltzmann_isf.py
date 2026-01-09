from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axis import Axis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod, get_default_isf_times
from coherent_rates.isf import (
    get_analytical_isf,
    get_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

CAM_DARK_BLUE = "#133844"
CAM_WARM_BLUE = "#00BDB6"
CAM_SLATE_1 = "#ECEEF1"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Utopia"],
        "text.latex.preamble": r"\usepackage{fourier}" + "\n" + r"\usepackage{amsmath}",
        "font.size": 11,
    },
)


def get_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


def format_axis_scientific(ax: Axis) -> None:
    formatter = ticker.ScalarFormatter(useMathText=True)
    # 2. Force scientific notation
    # (0, 0) tells it to use scientific notation for all numbers regardless of size
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.set_major_formatter(formatter)


def plot_periodic_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=100,
    )

    times = get_default_isf_times(system=system, config=config)
    isf = get_boltzmann_isf(system, config, times, n_repeats=5000)

    fig, ax = plt.subplots(
        figsize=get_fig_size(),
        layout="constrained",
    )

    fit = GaussianMethod().get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = GaussianMethod.get_fitted_data(fit, isf["basis"])

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="center right",
        fontsize=9,
        bbox_to_anchor=(1.0, 0.6),
    )
    legend.get_frame().set_alpha(0)

    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower left",
        borderpad=1.0,
        bbox_to_anchor=(0.1, 0, 1, 1),
        bbox_transform=ax.transAxes,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)

    inset_ax.set_facecolor((0, 0, 0, 0))
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9, labelpad=-1)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.periodic.pdf")


def plot_free_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=100,
    )

    times = get_default_isf_times(system=system, config=config)
    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    fig, ax = plt.subplots(
        figsize=get_fig_size(),
        layout="constrained",
    )
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)
    fig, ax, line = plot_value_list_against_time(
        analytical_isf,
        ax=ax,
        measure="abs",
    )
    line.set_label("Analytical")
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)

    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower right",
        borderpad=1.0,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)
    _, _, inset_line = plot_value_list_against_time(
        analytical_isf,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_DARK_BLUE)
    inset_line.set_linestyle("--")
    inset_ax.set_facecolor(CAM_SLATE_1)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.free.pdf")


if __name__ == "__main__":
    plot_free_isf()
    plot_periodic_isf()

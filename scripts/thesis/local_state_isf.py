from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod
from coherent_rates.isf import get_boltzmann_isf, get_local_boltzmann_isf
from coherent_rates.state import (
    ThermalLocalizationStrategy,
)
from coherent_rates.system import (
    LITHIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    format_axis_scientific,
    get_thesis_figure,
)


def plot_periodic_isf() -> None:
    system = LITHIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(66,),
        truncation=50,
        temperature=155,
    )

    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=100)

    fig, ax = get_thesis_figure()

    method = GaussianMethod(measure="abs")
    method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)

    strategy = ThermalLocalizationStrategy(
        system=system,
        config=config,
        sigma_0=(system.lattice_constant / 18,),
    )
    isf = get_local_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=1000,
        strategy=strategy,
    )
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Localized")
    line.set_color(CAM_BLUE.warm)

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
    inset_line.set_color(CAM_BLUE.warm)

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
    fig.savefig("scripts/thesis/local_state_isf.pdf")


if __name__ == "__main__":
    plot_periodic_isf()

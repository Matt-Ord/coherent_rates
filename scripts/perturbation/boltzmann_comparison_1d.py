import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.constants import (
    Boltzmann,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_analytical_isf,
    get_boltzmann_isf,
    get_scattered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_SLATE_1,
    format_axis_scientific,
    get_thesis_figure,
)


def plot_periodic_comparison() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(50,),
        truncation=50,
        temperature=155,
    )
    times = EvenlySpacedTimeBasis(1000, 1, 0, 1.5e-11)
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201
    isf = get_weak_boltzmann_isf(system, config, times)
    fig, ax = get_thesis_figure()
    method = GaussianMethod(measure="abs")
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("First Order")
    line.set_color(CAM_CHERRY.base)

    isf_se = get_weak_boltzmann_isf(system, config, times, second_order=True)
    fig, ax, line = plot_value_list_against_time(isf_se, measure="abs", ax=ax)
    line.set_label("Second Order")
    line.set_color(CAM_CHERRY.dark)

    isf_full = get_boltzmann_isf(system, config, times)
    fig, ax, line = plot_value_list_against_time(isf_full, measure="abs", ax=ax)
    line.set_label("Complete Simulation")
    line.set_color(CAM_BLUE.warm)

    ax.set_ylim(0.9 * np.abs(fitted_data["data"][-1]), 1)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/boltzmann_comparison_1d.pdf")


def plot_free_comparison() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(2,),
        truncation=25,
        temperature=155,
    )

    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201
    print("Decay after 1e-10s")  # noqa: T201
    decayed_isf = np.exp(
        -(Boltzmann * config.temperature * (1e-10 * delta_k) ** 2) / (2 * system.mass),
    )
    print(f"I: {decayed_isf:0.3e}")  # noqa: T201
    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    isf = get_weak_boltzmann_isf(
        system,
        config,
        times,
        second_order=True,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    fig, ax = get_thesis_figure()
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)
    fig, ax, line = plot_value_list_against_time(
        analytical_isf,
        ax=ax,
        measure="abs",
    )
    line.set_label("Analytical")
    line.set_color(CAM_BLUE.dark)
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

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower right",
        borderpad=1.0,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_BLUE.warm)
    _, _, inset_line = plot_value_list_against_time(
        analytical_isf,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_BLUE.dark)
    inset_line.set_linestyle("--")
    inset_ax.set_ylim(0, 1.1 * np.max(np.angle(analytical_isf["data"])))
    inset_ax.set_xlim(ax.get_xlim())
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

    fig.savefig("scripts/perturbation/boltzmann_comparison_1d.free.pdf")


if __name__ == "__main__":
    plot_free_comparison()
    plot_periodic_comparison()

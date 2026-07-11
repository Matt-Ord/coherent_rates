import numpy as np
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import (
    PeriodicSystemConfig,
)
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
    get_scattered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    format_axis_scientific,
    get_thesis_figure,
)


def plot_periodic_weak_isf() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    fig, ax = get_thesis_figure()

    isf = get_weak_boltzmann_isf(system, config, times)
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("First Order")
    line.set_color(CAM_CHERRY.base)

    isf_so = get_weak_boltzmann_isf(system, config, times, second_order=True)
    fig, ax, line = plot_value_list_against_time(isf_so, measure="abs", ax=ax)
    line.set_label("Second Order")
    line.set_color(CAM_CHERRY.dark)

    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Complete Simulation")
    line.set_color(CAM_BLUE.warm)

    ax.set_ylim(0.9 * np.abs(isf["data"][-1]), 1.0)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(frameon=False, loc="lower left", fontsize=9)
    legend.get_frame().set_alpha(0)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/boltzmann_comparison_2d.pdf")


if __name__ == "__main__":
    plot_periodic_weak_isf()

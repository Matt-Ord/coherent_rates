from typing import Any

import numpy as np
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod, get_scattered_momentum
from coherent_rates.isf import get_boltzmann_isf, get_local_boltzmann_isf
from coherent_rates.state import (
    ThermalLocalizationStrategy,
)
from coherent_rates.system import (
    LITHIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    get_thesis_figure,
)


# Gererated by
# scripts/ballistic/ballistic_demo.py
def _load_isf() -> tuple[
    np.ndarray[Any, np.dtype[np.floating]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    data = np.load("scripts/thesis/isf_serialized.npz")
    return data["isf"], data["times"]


def plot_periodic_isf() -> None:
    system = LITHIUM_COPPER_BRIDGE_SYSTEM_1D
    print(f"system.barrier_energy: {system.barrier_energy:.2e}")  # noqa: T201

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(66,),
        truncation=50,
        temperature=155,
    )

    print(  # noqa: T201
        f"delta k: {get_scattered_momentum(system, config, [config.direction])[0]:.2e}",
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
    line.set_color(CAM_BLUE.dark)

    classical_isf, times_classical = _load_isf()
    (line,) = ax.plot(times_classical, classical_isf, "--")
    line.set_label("Classical")
    line.set_color(CAM_CHERRY.dark)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")
    ax.set_ylim(0.9, 1.0)
    ax.set_xlim(0, 0.5e-12)

    legend = ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/local_state_isf.pdf")


if __name__ == "__main__":
    plot_periodic_isf()

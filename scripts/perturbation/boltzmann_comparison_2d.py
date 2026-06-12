from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
from surface_potential_analysis.basis.time_basis_like import BasisWithTimeLike
from surface_potential_analysis.state_vector.eigenstate_list import StatisticalValueList
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.util.decorators import cached
from surface_potential_analysis.util.plot import plot_data_1d

from coherent_rates.config import (
    PeriodicSystemConfig,
)
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_band_resolved_boltzmann_isf,
    get_boltzmann_isf,
    get_scattered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    System,
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


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def _get_split_band_isf_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    *,
    n_repeats: int = 10,
) -> Path:
    return Path(
        f"data/{hash((system, config))}.{hash(times)}.{n_repeats}.boltzmann.isf.split",
    )


@cached(_get_split_band_isf_path)
def get_split_band_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 10,
) -> StatisticalValueList[TupleBasisLike[BasisLike[Any, Any], _BT0]]:
    return get_band_resolved_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )


def get_average_band_energy(
    system: System,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    basis = hamiltonian["basis"][0].wavefunctions["basis"]
    n_bands = basis[0][0].n
    return np.average(hamiltonian["data"].reshape(n_bands, -1), axis=-1)


def plot_split_band_isf() -> None:
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

    isf = get_split_band_isf(system, config, times, n_repeats=20)
    band_energies = get_average_band_energy(system, config)

    stacked = isf["data"].reshape(isf["basis"].shape)
    fig, ax, line = plot_data_1d(
        np.sum(
            stacked[(band_energies < (SODIUM_COPPER_SYSTEM_2D.barrier_energy / 9))],
            axis=0,
        ),
        isf["basis"][1].times,
        measure="abs",
        ax=ax,
    )
    line.set_label(r"$E < E_{\mathrm{bridge}}$")
    line.set_color(CAM_CHERRY.base)

    fig, ax, line = plot_data_1d(
        np.sum(
            stacked[(band_energies < (SODIUM_COPPER_SYSTEM_2D.barrier_energy))],
            axis=0,
        ),
        isf["basis"][1].times,
        measure="abs",
        ax=ax,
    )
    line.set_label(r"$E < E_{\mathrm{top}}$")
    line.set_color(CAM_CHERRY.dark)

    fig, ax, line = plot_data_1d(
        np.sum(
            stacked[(band_energies < (3 * Boltzmann * config.temperature))],
            axis=0,
        ),
        isf["basis"][1].times,
        measure="abs",
        ax=ax,
    )
    line.set_label(r"$E < 3 K_b T$")
    line.set_color(CAM_BLUE.warm)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(frameon=False, loc="lower left", fontsize=9)
    legend.get_frame().set_alpha(0)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/boltzmann_comparison_2d.split_band.pdf")


if __name__ == "__main__":
    plot_periodic_weak_isf()
    plot_split_band_isf()

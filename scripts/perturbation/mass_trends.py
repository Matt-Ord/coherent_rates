import itertools
from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.util.decorators import cached, disabled_timing

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
    get_free_particle_isf,
    get_free_particle_time,
)
from coherent_rates.isf import (
    get_momentum_threshold_effective_mass,
    get_occupation_threshold_effective_mass,
    get_ordered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, System
from coherent_rates.util import CAM_BLUE, CAM_CHERRY, get_thesis_figure


def _assess_isf_validity() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(1,),
        truncation=75,
        temperature=155,
        offset=(0.01,),  # Breaks some of the symmetry
    )

    fig, _axes = plt.subplots(
        layout="constrained",
        nrows=5,
        ncols=5,
        figsize=(15, 15),
    )

    barrier_ratios = np.linspace(0, 4, 5)
    mass_ratios = np.linspace(1, 20, 5)

    m_0 = (2 * np.pi * hbar) ** 2 / (
        2 * Boltzmann * config.temperature * system.lattice_constant**2
    )
    v_0 = Boltzmann * config.temperature

    with disabled_timing():
        for (barrier_ratio, mass_ratio), ax in zip(
            itertools.product(
                barrier_ratios,
                mass_ratios,
            ),
            _axes.ravel(),
            strict=True,
        ):
            system = system.with_mass(m_0 * mass_ratio)
            system = system.with_barrier_energy(v_0 * barrier_ratio)

            get_ordered_momentum.load_or_call_cached(system, config)

            times = GaussianMethod(measure="abs").get_fit_times(
                system=system,
                config=config,
            )

            isf = get_weak_boltzmann_isf(system, config, times)
            _, _, _line = plot_value_list_against_time(isf, measure="abs", ax=ax)

            total_occupation, effective_mass = get_momentum_threshold_effective_mass(
                system,
                config,
            )
            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system,
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_CHERRY.warm,
                linestyle="--",
                label="Effective Mass",
            )
            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system.with_mass(effective_mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_CHERRY.dark,
                linestyle="--",
                label="Effective Mass",
            )
            ax.set_ylim(((1 - 1.1 * total_occupation), 1))
            ax.set_title(
                f"Barrier: {barrier_ratio:.2f}, Mass: {mass_ratio:.2f}",
                fontsize=8,
            )

            min_isf = np.min(np.abs(isf["data"]))
            total_occupation, effective_mass = get_occupation_threshold_effective_mass(
                system,
                config,
                threshold=1 - min_isf,
            )
            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system.with_mass(effective_mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_BLUE.dark,
                linestyle=":",
                label="Effective Mass",
            )

            def loss_function(threshold_guess: float) -> float:
                if threshold_guess <= 0:
                    return float("inf")
                tot_occ, eff_mass = get_momentum_threshold_effective_mass(
                    system,  # noqa: B023
                    config,
                    threshold=threshold_guess,
                )
                fit_system = system.with_mass(eff_mass)  # noqa: B023

                free_time = get_free_particle_time(fit_system, config)
                t_cutoff = np.sqrt(2) * free_time
                time_mask = times.times <= t_cutoff  # noqa: B023

                predicted_isf = get_free_particle_isf(
                    fit_system,
                    config,
                    times.times[time_mask],  # noqa: B023
                    offset=1 - tot_occ,
                )
                return float(
                    np.mean((np.abs(isf["data"][time_mask]) - predicted_isf) ** 2),  # noqa: B023
                )

            optimization_result = scipy.optimize.brute(
                loss_function,
                ranges=[(0.005, 0.5)],
            )

            optimal_threshold = float(optimization_result[0])

            # Extract final parameters using the optimal threshold
            total_occupation, effective_mass = get_momentum_threshold_effective_mass(
                system,
                config,
                threshold=optimal_threshold,
            )
            print(m_0 * mass_ratio, effective_mass, effective_mass / (m_0 * mass_ratio))
            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system.with_mass(effective_mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_BLUE.warm,
                linestyle=":",
                label="Effective Mass",
            )

            get_ordered_momentum.delete_cache(system, config)

    fig.savefig("scripts/perturbation/mass_trends.validity.pdf")


def _get_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    total_occupation, effective_mass = get_momentum_threshold_effective_mass(
        system,
        config,
    )
    return total_occupation, effective_mass / system.mass


def _get_occupation_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    isf = get_weak_boltzmann_isf(
        system,
        config,
        GaussianMethod().get_fit_times(system=system, config=config),
    )
    total_occupation, effective_mass = get_occupation_threshold_effective_mass(
        system,
        config,
        threshold=1 - np.min(np.abs(isf["data"])),
    )
    return total_occupation, effective_mass / system.mass


def _get_optimal_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_weak_boltzmann_isf(system, config, times)

    def loss_function(threshold_guess: float) -> float:
        if threshold_guess <= 0:
            return float("inf")
        tot_occ, eff_mass = get_momentum_threshold_effective_mass(
            system,
            config,
            threshold=threshold_guess,
        )
        fit_system = system.with_mass(eff_mass)

        free_time = get_free_particle_time(fit_system, config)
        t_cutoff = np.sqrt(2) * free_time
        time_mask = times.times <= t_cutoff

        predicted_isf = get_free_particle_isf(
            fit_system,
            config,
            times.times[time_mask],
            offset=1 - tot_occ,
        )
        return float(
            np.mean((np.abs(isf["data"][time_mask]) - predicted_isf) ** 2),
        )

    optimization_result = scipy.optimize.brute(
        loss_function,
        ranges=[(0.005, 0.5)],
    )

    optimal_threshold = float(optimization_result[0])

    total_occupation, effective_mass = get_momentum_threshold_effective_mass(
        system,
        config,
        threshold=optimal_threshold,
    )
    return total_occupation, effective_mass / system.mass


def _all_mass_ratio_path() -> Path:
    return Path("scripts/perturbation/mass_trends.all_mass_ratios.npz")


class _MassRatioData(TypedDict):
    xv: np.ndarray
    yv: np.ndarray
    mass_ratios: tuple[np.ndarray, np.ndarray]
    occupation_mass_ratios: tuple[np.ndarray, np.ndarray]
    optimal_mass_ratios: tuple[np.ndarray, np.ndarray]
    shape: tuple[int, int]


@cached(_all_mass_ratio_path)
def get_all_mass_ratios() -> _MassRatioData:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=155,
        offset=(0.01,),  # Breaks some of the symmetry
    )

    barrier_ratios = np.linspace(0, 4, 50)
    mass_ratios = np.linspace(1, 20, 50)
    m_0 = (2 * np.pi * hbar) ** 2 / (
        2 * Boltzmann * config.temperature * system.lattice_constant**2
    )
    v_0 = Boltzmann * config.temperature
    xv, yv = np.meshgrid(barrier_ratios, mass_ratios)
    xv = xv.ravel()
    yv = yv.ravel()

    out: _MassRatioData = {
        "xv": xv,
        "yv": yv,
        "mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "occupation_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optimal_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "shape": (50, 50),
    }
    for i, (barrier_ratio, mass_ratio) in enumerate(
        zip(xv, yv, strict=True),
    ):
        print(f"i: {i}")  # noqa: T201
        with disabled_timing():
            system = system.with_mass(m_0 * mass_ratio)
            system = system.with_barrier_energy(v_0 * barrier_ratio)

            get_ordered_momentum.load_or_call_cached(system, config)
            out["mass_ratios"][0][i], out["mass_ratios"][1][i] = (
                _get_threshold_mass_ratio(
                    system,
                    config,
                )
            )
            out["occupation_mass_ratios"][0][i], out["occupation_mass_ratios"][1][i] = (
                _get_occupation_mass_ratio(
                    system,
                    config,
                )
            )

            out["optimal_mass_ratios"][0][i], out["optimal_mass_ratios"][1][i] = (
                _get_optimal_mass_ratio(
                    system,
                    config,
                )
            )

            get_ordered_momentum.delete_cache(system, config)
    return out


def _plot_isf_mass_ratios() -> None:

    PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=155,
        offset=(0.01,),  # Breaks some of the symmetry
    )

    data = get_all_mass_ratios()
    xv, yv, _mass_ratios, _occupation_mass_ratios, optimal_mass_ratios = (
        data["xv"],
        data["yv"],
        data["mass_ratios"],
        data["occupation_mass_ratios"],
        data["optimal_mass_ratios"],
    )

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(50, 50),
        yv.reshape(50, 50),
        optimal_mass_ratios[1].reshape(50, 50),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")
    ax.set_title("Intrinsic Mass")
    mesh.set_clim(0, 1)

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/mass_trends.pdf")

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(50, 50),
        yv.reshape(50, 50),
        # If we fit to a Gaussian which ends at 0, what do we
        # think the mass will be?
        (optimal_mass_ratios[1] / optimal_mass_ratios[0]).reshape(50, 50),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")
    ax.set_title("Empirical Mass")

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/mass_trends.empirical.pdf")


if __name__ == "__main__":
    _assess_isf_validity()
    _plot_isf_mass_ratios()

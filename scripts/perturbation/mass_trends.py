import itertools
from pathlib import Path

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
from coherent_rates.solve import get_hamiltonian
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

            hamiltonian = get_hamiltonian(system, config)
            energies_per_band = hamiltonian["data"].reshape(config.n_bands, -1)
            occupations = np.exp(
                -energies_per_band / (Boltzmann * config.temperature),
            )
            occupations /= np.sum(occupations)
            missing_occupation = 1 - np.sum(occupations[:50])
            print("Missing occupation:", missing_occupation)  # noqa: T201

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

            print("Optimization result:", optimization_result)  # noqa: T201

            optimal_threshold = float(optimization_result[0])

            # Extract final parameters using the optimal threshold
            total_occupation, effective_mass = get_momentum_threshold_effective_mass(
                system,
                config,
                threshold=optimal_threshold,
            )
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
            print("Missing occupation:", 1 - total_occupation)

            get_ordered_momentum.delete_cache(system, config)

    fig.savefig("scripts/perturbation/mass_trends.validity.pdf")


def _get_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    _total_occupation, effective_mass = get_momentum_threshold_effective_mass(
        system,
        config,
    )
    return effective_mass / system.mass


def _get_occupation_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    isf = get_weak_boltzmann_isf(
        system,
        config,
        GaussianMethod().get_fit_times(system=system, config=config),
    )
    _total_occupation, effective_mass = get_occupation_threshold_effective_mass(
        system,
        config,
        threshold=1 - np.min(np.abs(isf["data"])),
    )
    return effective_mass / system.mass


def _get_optimal_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

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
    # Extract final parameters using the optimal threshold
    _total_occupation, effective_mass = get_momentum_threshold_effective_mass(
        system,
        config,
        threshold=optimal_threshold,
    )
    return effective_mass / system.mass


def _all_mass_ratio_path() -> Path:
    return Path("scripts/perturbation/mass_trends.all_mass_ratios.npz")


@cached(_all_mass_ratio_path)
def get_all_mass_ratios() -> dict[str, np.ndarray]:
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

    mass_ratios = np.zeros_like(xv)
    occupation_mass_ratios = np.zeros_like(xv)
    optimal_mass_ratios = np.zeros_like(xv)
    for i, (barrier_ratio, mass_ratio) in enumerate(
        zip(xv.flat, yv.flat, strict=True),
    ):
        print(f"i: {i}")
        with disabled_timing():
            system = system.with_mass(m_0 * mass_ratio)
            system = system.with_barrier_energy(v_0 * barrier_ratio)

            get_ordered_momentum.load_or_call_cached(system, config)
            mass_ratios.flat[i] = _get_threshold_mass_ratio(
                system,
                config,
            )
            occupation_mass_ratios.flat[i] = _get_occupation_mass_ratio(
                system,
                config,
            )

            optimal_mass_ratios.flat[i] = _get_optimal_mass_ratio(
                system,
                config,
            )

            get_ordered_momentum.delete_cache(system, config)
    return {
        "xv": xv,
        "yv": yv,
        "mass_ratios": mass_ratios,
        "occupation_mass_ratios": occupation_mass_ratios,
        "optimal_mass_ratios": optimal_mass_ratios,
    }


def _plot_isf_mass_ratios() -> None:

    PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=155,
        offset=(0.01,),  # Breaks some of the symmetry
    )

    fig, ax = get_thesis_figure()

    data = get_all_mass_ratios()
    xv, yv, _mass_ratios, _occupation_mass_ratios, optimal_mass_ratios = (
        data["xv"],
        data["yv"],
        data["mass_ratios"],
        data["occupation_mass_ratios"],
        data["optimal_mass_ratios"],
    )

    mesh = ax.pcolormesh(
        xv,
        yv,
        optimal_mass_ratios,
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy / $k_bT$")
    ax.set_ylabel(r"Kinetic Energy $\frac{m}{m_0}$")
    mesh.set_clim(0, 1)

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/mass_trends.pdf")


if __name__ == "__main__":
    # _assess_isf_validity()
    _plot_isf_mass_ratios()

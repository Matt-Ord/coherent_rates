import itertools
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, atomic_mass, hbar
from scipy.integrate import quad
from scipy.special import ellipk
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.util.decorators import cached, disabled_timing

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
    get_free_particle_isf,
    get_free_particle_time,
    get_scattered_momentum,
)
from coherent_rates.isf import (
    get_ordered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    System,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    format_axis_scientific,
    get_fancy_figure,
    get_thesis_figure,
)


def _get_fixed_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
    *,
    target_occupation: float,
) -> tuple[float, float]:
    momentum, energy_per_state = get_ordered_momentum(system, config)
    prefactor = 1 / (config.temperature * Boltzmann * system.mass)
    momentum *= prefactor

    sort_indices = np.argsort(energy_per_state)[::-1]
    momentum = momentum[sort_indices]
    energy_per_state = energy_per_state[sort_indices]

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    cumsum_thermal_factors = np.cumsum(thermal_factors)
    cumsum_inverse_mass = np.cumsum(thermal_factors * momentum / system.mass)

    # Find the state cutoff index closest to the target occupation
    idx = int(np.argmin(np.abs(cumsum_thermal_factors - target_occupation)))

    actual_occupation = cumsum_thermal_factors[idx]
    inverse_mass = cumsum_inverse_mass[idx] / actual_occupation
    effective_mass = 1 / inverse_mass

    return float(actual_occupation), float(effective_mass / system.mass)


def _get_classical_above_barrier_occupation(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    u0 = system.barrier_energy / (config.temperature * Boltzmann)

    def integrand_below(epsilon: float) -> float:
        # Trapped states (0 <= E < U0)
        return ellipk(epsilon) * np.exp(-u0 * epsilon)

    def integrand_above(epsilon: float) -> float:
        # Running states (E >= U0)
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    z_below, _ = quad(integrand_below, 0, 1)
    z_above, _ = quad(integrand_above, 1, np.inf)

    return z_above / (z_below + z_above)


def _get_classical_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:
    return _get_fixed_threshold_mass_ratio(
        system,
        config,
        target_occupation=_get_classical_above_barrier_occupation(system, config),
    )


def _get_above_barrier_occupation(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    hamiltonian = get_hamiltonian(system, config)
    energy_per_state = hamiltonian["data"]

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    return np.sum(thermal_factors * (energy_per_state >= system.barrier_energy))


def _get_above_barrier_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:
    return _get_fixed_threshold_mass_ratio(
        system,
        config,
        target_occupation=_get_above_barrier_occupation(system, config),
    )


def _get_long_time_occupation(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    hamiltonian = get_hamiltonian(system, config)
    energy_per_state = hamiltonian["data"]

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    return np.sum(thermal_factors * (energy_per_state >= system.barrier_energy))


def _get_long_time_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
    *,
    t_factor: float = 8,
) -> tuple[float, float]:
    times = GaussianMethod(measure="abs", t_factor=t_factor).get_fit_times(
        system=system,
        config=config,
    )
    target_occupation = (
        1
        - np.abs(
            get_weak_boltzmann_isf.call_uncached(system, config, times)["data"],
        )[-1]
    )
    return _get_fixed_threshold_mass_ratio(
        system,
        config,
        target_occupation=target_occupation,
    )


def _get_zero_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    return _get_fixed_threshold_mass_ratio(system, config, target_occupation=1)


def _get_optimal_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
    *,
    t_factor: float = 4,
) -> tuple[float, float]:

    times = GaussianMethod(measure="abs", t_factor=t_factor).get_fit_times(
        system=system,
        config=config,
    )
    isf = np.abs(get_weak_boltzmann_isf.call_uncached(system, config, times)["data"])

    momentum, energy_per_state = get_ordered_momentum(system, config)
    prefactor = 1 / (config.temperature * Boltzmann * system.mass)
    momentum *= prefactor

    sort_indices = np.argsort(momentum)[::-1]
    momentum = momentum[sort_indices]
    energy_per_state = energy_per_state[sort_indices]

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    # Determine valid cut indices
    cut_indices = np.arange(1, len(momentum) + 1)
    min_momentum = 1e-8
    condition_lower_bound = momentum[cut_indices - 1] > min_momentum
    condition_upper_bound = np.ones_like(cut_indices, dtype=bool)
    max_momentum = 2.0
    condition_upper_bound[:-1] = momentum[1:] < max_momentum
    cut_indices = cut_indices[condition_lower_bound & condition_upper_bound]

    assert len(cut_indices) > 0, (  # noqa: S101
        "No valid cut indices found within the threshold bounds"
    )

    # 3. Compute cumulative sums for fast prefix-slice lookups
    cumsum_thermal_factors = np.cumsum(thermal_factors)
    cumsum_inverse_mass = np.cumsum(thermal_factors * momentum / system.mass)

    # 4. Extract total occupations and effective masses for all cut indices
    total_occupations = cumsum_thermal_factors[cut_indices - 1]
    inverse_masses = cumsum_inverse_mass[cut_indices - 1] / total_occupations
    effective_masses = 1 / inverse_masses

    # 5. Evaluate the losses for each valid cut index
    losses = [
        np.mean(
            (
                isf
                - get_free_particle_isf(
                    system.with_mass(mass),
                    config,
                    times.times,
                    offset=1 - occupation,
                )
            )
            ** 2,
        )
        for occupation, mass in zip(total_occupations, effective_masses, strict=False)
    ]

    # 6. Find the optimal index
    optimal_array_index = np.argmin(losses)
    optimal_total_occupation = float(total_occupations[optimal_array_index])
    optimal_effective_mass = float(effective_masses[optimal_array_index])

    print(  # noqa: T201
        f"({optimal_array_index}/{len(cut_indices)})",
    )
    print(  # noqa: T201
        "Optimal p^2 / (m k_B T) threshold:",
        momentum[cut_indices[optimal_array_index] - 1],
    )

    return optimal_total_occupation, optimal_effective_mass / system.mass


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
            print(f"Barrier: {barrier_ratio:.2f}, Mass: {mass_ratio:.2f}")  # noqa: T201
            system = system.with_mass(m_0 * mass_ratio)
            system = system.with_barrier_energy(v_0 * barrier_ratio)

            get_ordered_momentum.load_or_call_cached(system, config)

            times = GaussianMethod(measure="abs").get_fit_times(
                system=system,
                config=config,
            )

            isf = get_weak_boltzmann_isf.call_uncached(system, config, times)
            _, _, _line = plot_value_list_against_time(isf, measure="abs", ax=ax)

            total_occupation, effective_mass = _get_classical_threshold_mass_ratio(
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
                    system.with_mass(effective_mass * system.mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_CHERRY.dark,
                linestyle="--",
                label="Effective Mass",
            )
            ax.set_title(
                f"Barrier: {barrier_ratio:.2f}, Mass: {mass_ratio:.2f}",
                fontsize=8,
            )

            total_occupation, effective_mass = _get_optimal_threshold_mass_ratio(
                system,
                config,
            )

            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system.with_mass(effective_mass * system.mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_BLUE.warm,
                linestyle=":",
                label="Effective Mass",
            )

            ax.set_ylim(((1 - 1.1 * total_occupation), 1))

            get_ordered_momentum.delete_cache(system, config)

    fig.savefig("scripts/perturbation/effective_mass.validity.pdf")


def _assess_isf_validity_2d() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(3 / 20, 0),
        truncation=625,
        temperature=155,
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
            print(f"Barrier: {barrier_ratio:.2f}, Mass: {mass_ratio:.2f}")  # noqa: T201
            system = system.with_mass(m_0 * mass_ratio)
            system = system.with_barrier_energy(v_0 * barrier_ratio)

            get_ordered_momentum.load_or_call_cached(system, config)

            times = GaussianMethod(measure="abs").get_fit_times(
                system=system,
                config=config,
            )

            isf = get_weak_boltzmann_isf.call_uncached(system, config, times)
            _, _, _line = plot_value_list_against_time(isf, measure="abs", ax=ax)

            total_occupation, effective_mass = _get_classical_threshold_mass_ratio(
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
                    system.with_mass(effective_mass * system.mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_CHERRY.dark,
                linestyle="--",
                label="Effective Mass",
            )
            ax.set_title(
                f"Barrier: {barrier_ratio:.2f}, Mass: {mass_ratio:.2f}",
                fontsize=8,
            )

            total_occupation, effective_mass = _get_optimal_threshold_mass_ratio(
                system,
                config,
                t_factor=6,
            )

            (_line,) = ax.plot(
                times.times,
                get_free_particle_isf(
                    system.with_mass(effective_mass * system.mass),
                    config,
                    times.times,
                    offset=1 - total_occupation,
                ),
                color=CAM_BLUE.dark,
                linestyle=":",
                label="Effective Mass",
            )
            ax.set_ylim(((1 - 1.1 * total_occupation), 1))
            get_ordered_momentum.delete_cache(system, config)

    fig.savefig("scripts/perturbation/effective_mass.validity.pdf")


def _all_mass_ratio_path() -> Path:
    return Path("scripts/perturbation/effective_mass.all_mass_ratios.npz")


class _MassRatioData(TypedDict):
    xv: np.ndarray
    yv: np.ndarray

    shape: tuple[int, int]
    classical_threshold_mass_ratios: tuple[np.ndarray, np.ndarray]
    optimal_threshold_mass_ratios: tuple[np.ndarray, np.ndarray]


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
        "classical_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optimal_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
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

            (
                out["classical_threshold_mass_ratios"][0][i],
                out["classical_threshold_mass_ratios"][1][i],
            ) = _get_classical_threshold_mass_ratio(system, config)

            (
                out["optimal_threshold_mass_ratios"][0][i],
                out["optimal_threshold_mass_ratios"][1][i],
            ) = _get_optimal_threshold_mass_ratio(system, config)

            get_ordered_momentum.delete_cache(system, config)
    return out


def _plot_isf_mass_ratios() -> None:

    data = get_all_mass_ratios()
    xv, yv, shape, zero_threshold_mass_ratios, optimal_threshold_mass_ratios = (
        data["xv"],
        data["yv"],
        data["shape"],
        data["classical_threshold_mass_ratios"],
        data["optimal_threshold_mass_ratios"],
    )

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(shape),
        yv.reshape(shape),
        optimal_threshold_mass_ratios[1].reshape(shape),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")
    mesh.set_clim(0, 1)

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/effective_mass.optimal.pdf")

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(shape),
        yv.reshape(shape),
        # If we fit to a Gaussian which ends at 0, what do we
        # think the mass will be?
        (zero_threshold_mass_ratios[1]).reshape(
            shape,
        ),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/effective_mass.classical.pdf")


def _all_mass_ratio_path_2d() -> Path:
    return Path("scripts/perturbation/effective_mass.all_mass_ratios.2d.npz")


@cached(_all_mass_ratio_path_2d)
def get_all_mass_ratios_2d() -> _MassRatioData:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(3 / 20, 0),
        truncation=625,
        temperature=155,
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
        "classical_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optimal_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
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

            (
                out["classical_threshold_mass_ratios"][0][i],
                out["classical_threshold_mass_ratios"][1][i],
            ) = _get_classical_threshold_mass_ratio(
                system,
                config,
            )

            (
                out["optimal_threshold_mass_ratios"][0][i],
                out["optimal_threshold_mass_ratios"][1][i],
            ) = _get_optimal_threshold_mass_ratio(
                system,
                config,
            )

            get_ordered_momentum.delete_cache(system, config)
    return out


def _plot_isf_mass_ratios_2d() -> None:

    data = get_all_mass_ratios_2d()
    xv, yv, shape, classical_threshold_mass_ratios, optimal_threshold_mass_ratios = (
        data["xv"],
        data["yv"],
        data["shape"],
        data["classical_threshold_mass_ratios"],
        data["optimal_threshold_mass_ratios"],
    )

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(shape),
        yv.reshape(shape),
        optimal_threshold_mass_ratios[1].reshape(shape),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")
    mesh.set_clim(0, 1)

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/effective_mass.optimal.2d.pdf")

    fig, ax = get_thesis_figure()
    mesh = ax.pcolormesh(
        xv.reshape(shape),
        yv.reshape(shape),
        # If we fit to a Gaussian which ends at 0, what do we
        # think the mass will be?
        (classical_threshold_mass_ratios[1]).reshape(
            shape,
        ),
        shading="nearest",
    )
    ax.set_xlabel(r"Barrier Energy $\frac{E_b}{k_bT}$")
    ax.set_ylabel(r"Mass $\frac{m}{m_0}$")

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/effective_mass.classical.2d.pdf")


def _plot_isf_mass_fit_1d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["classical", "optimal"] = "optimal",
) -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(energy_factor * system.barrier_energy)
    system = system.with_mass(mass_factor * system.mass)

    config = PeriodicSystemConfig(
        (50,),
        (250,),
        direction=(1,),
        truncation=200,
        temperature=temperature,
        offset=(0.01,),
    )
    config_1 = PeriodicSystemConfig(
        (50,),
        (150,),
        direction=(1,),
        truncation=100,
        temperature=temperature,
    )

    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201

    times = GaussianMethod(measure="abs", t_factor=16).get_fit_times(
        system=system,
        config=config,
    )

    get_ordered_momentum.load_or_call_cached(system, config)
    isf = get_weak_boltzmann_isf.call_uncached(system, config, times)

    if ty == "classical":
        total_occupation, effective_mass = _get_classical_threshold_mass_ratio(
            system,
            config,
        )
        total_occupation, effective_mass = _get_classical_threshold_mass_ratio(
            system,
            config,
        )
    else:
        total_occupation, effective_mass = _get_optimal_threshold_mass_ratio(
            system,
            config,
        )
        total_occupation, effective_mass = _get_zero_threshold_mass_ratio(
            system,
            config,
        )
    get_ordered_momentum.delete_cache(system, config)

    fig, ax = get_thesis_figure()

    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_color(CAM_CHERRY.base)

    isf = get_weak_boltzmann_isf.call_uncached(system, config_1, times)
    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_linestyle("--")
    line_first_order.set_color(CAM_CHERRY.dark)

    print(effective_mass)  # noqa: T201
    (line_effective_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system.with_mass(effective_mass * system.mass),
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_CHERRY.dark,
        linestyle="--",
        label="Effective Mass",
    )

    (line_real_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system,
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_BLUE.warm,
        linestyle="--",
        label="Actual Mass",
    )

    ax.axhline(
        1 - _get_above_barrier_occupation(system, config),
        color=CAM_BLUE.dark,
        linestyle=":",
        label="Above Barrier Occupation",
    )

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)
    ax.legend(
        loc="upper right",
        handles=[line_real_mass, line_effective_mass, line_first_order],
        fontsize=9,
        frameon=False,
    )

    if ty in {"optimal", "alt"}:
        ax.set_ylim(((1 - 1.1 * total_occupation), 1))
    else:
        ax.set_ylim((0.8 * np.min(np.abs(isf["data"]))), 1)

    fig.savefig(f"scripts/perturbation/effective_mass.fit.{ty}.1d.pdf")


def _plot_isf_mass_fit_2d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["classical", "optimal"] = "optimal",
) -> None:

    system = SODIUM_COPPER_SYSTEM_2D
    system = system.with_barrier_energy(energy_factor * system.barrier_energy)
    system = system.with_mass(mass_factor * system.mass)

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(3 / 20, 0),
        truncation=625,
        temperature=155,
    )
    config = config.with_temperature(temperature)

    times = GaussianMethod(measure="abs", t_factor=8).get_fit_times(
        system=system,
        config=config,
    )

    get_ordered_momentum.load_or_call_cached(system, config)
    isf = get_weak_boltzmann_isf.call_uncached(system, config, times)
    if ty == "classical":
        total_occupation, effective_mass = _get_classical_threshold_mass_ratio(
            system,
            config,
        )
    elif ty == "optimal":
        total_occupation, effective_mass = _get_optimal_threshold_mass_ratio(
            system,
            config,
        )

    get_ordered_momentum.delete_cache(system, config)

    fig, ax = get_thesis_figure()

    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_color(CAM_CHERRY.base)

    (line_effective_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system.with_mass(effective_mass * system.mass),
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_CHERRY.dark,
        linestyle="--",
        label="Effective Mass",
    )

    (line_real_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system,
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_BLUE.warm,
        linestyle="--",
        label="Actual Mass",
    )

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    ax.legend(
        loc="upper right",
        handles=[line_real_mass, line_effective_mass, line_first_order],
        fontsize=9,
        frameon=False,
    )

    if ty in {"optimal", "alt"}:
        ax.set_ylim(((1 - 1.1 * total_occupation), 1))
    else:
        ax.set_ylim((0.8 * np.min(np.abs(isf["data"]))), 1)

    fig.savefig(f"scripts/perturbation/effective_mass.fit.{ty}.2d.pdf")


def _print_free_times() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=155,
    )
    free_time = get_free_particle_time(system, config)
    print(f"Free time: {free_time:.2e} s")  # noqa: T201

    system = SODIUM_COPPER_SYSTEM_2D
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(3 / 20, 0),
        truncation=625,
        temperature=155,
    )
    free_time = get_free_particle_time(system, config)
    print(f"Free time: {free_time:.2e} s")  # noqa: T201


def _charlie_mass_ratios() -> Path:
    return Path("scripts/perturbation/effective_mass.charlie_mass_ratios.npz")


@cached(_charlie_mass_ratios)
def get_charlie_mass_ratios() -> dict[str, Any]:
    base_system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    config = PeriodicSystemConfig(
        (50,),
        (150,),
        direction=(1,),
        truncation=100,
        temperature=155,
        offset=(0.01,),
    )

    element_masses = {
        "H": 1.00784 * atomic_mass,
        "He": 4.00260 * atomic_mass,
        "Li": 6.94100 * atomic_mass,
        "Na": base_system.mass,
    }

    mass_min = element_masses["H"]
    mass_max = 10 * base_system.mass
    grid_masses = np.logspace(np.log10(mass_min), np.log10(mass_max), 100)

    def compute_ratio(mass: float) -> float:
        sys = base_system.with_mass(mass)
        with disabled_timing():
            get_ordered_momentum.load_or_call_cached(sys, config)
            _, eff_mass_ratio = _get_zero_threshold_mass_ratio(sys, config)
            get_ordered_momentum.delete_cache(sys, config)
        return eff_mass_ratio

    grid_ratios = np.array([compute_ratio(m) for m in grid_masses])
    element_ratios = {elem: compute_ratio(m) for elem, m in element_masses.items()}

    return {
        "grid_masses": grid_masses,
        "grid_ratios": grid_ratios,
        "element_masses": element_masses,
        "element_ratios": element_ratios,
    }


def plot_charlie_mass_ratios() -> None:
    data = get_charlie_mass_ratios()
    grid_masses = data["grid_masses"]
    grid_ratios = data["grid_ratios"]
    element_masses = data["element_masses"]
    element_ratios = data["element_ratios"]

    fig, ax = get_fancy_figure()

    ax.plot(grid_masses / atomic_mass, grid_ratios)

    for elem, m in element_masses.items():
        ratio = element_ratios[elem]
        mass_in_u = m / atomic_mass
        ax.scatter(mass_in_u, ratio, color=CAM_BLUE.dark, marker="x", zorder=5)
        ax.annotate(
            elem,
            (mass_in_u, ratio),
            xytext=(7, -5) if elem == "H" else (7, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Mass / Atomic Mass Units")
    ax.set_ylabel(r"Effective Mass Ratio $m_{\mathrm{eff}} / m$")
    ax.set_xlim(0.8, np.max(grid_masses) / atomic_mass)

    ax.axhline(
        2.3938907263995297,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.set_ylim(2.2, None)
    ax.set_xscale("log")

    fig.savefig("scripts/perturbation/effective_mass.charlie.pdf")


if __name__ == "__main__":
    _assess_isf_validity()
    _assess_isf_validity_2d()
    _plot_isf_mass_ratios()
    _plot_isf_mass_ratios_2d()
    _print_free_times()
    _plot_isf_mass_fit_1d(ty="optimal", mass_factor=50)
    _plot_isf_mass_fit_1d()
    plot_charlie_mass_ratios()

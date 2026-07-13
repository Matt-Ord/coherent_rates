import itertools
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
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
    get_ordered_momentum,
    get_scaled_momentum_threshold_effective_mass,
    get_weak_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    System,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    format_axis_scientific,
    get_thesis_figure,
)


def _get_zero_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    total_occupation, effective_mass = get_scaled_momentum_threshold_effective_mass(
        system,
        config,
    )
    return total_occupation, effective_mass / system.mass


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
    isf = get_weak_boltzmann_isf.call_uncached(system, config, times)

    def loss_function(threshold_guess: float) -> float:
        if threshold_guess <= 0:
            return float("inf")
        tot_occ, eff_mass = get_scaled_momentum_threshold_effective_mass(
            system,
            config,
            threshold=threshold_guess,
        )
        fit_system = system.with_mass(eff_mass)

        free_time = get_free_particle_time(fit_system, config)
        t_cutoff = np.sqrt(2) * free_time
        time_mask = times.times <= 2 * t_cutoff

        predicted_isf = get_free_particle_isf(
            fit_system,
            config,
            times.times[time_mask],
            offset=1 - tot_occ,
        )
        return float(
            np.mean((np.abs(isf["data"][time_mask]) - predicted_isf) ** 2),
        )

    momentum, energy_per_state = get_ordered_momentum(system, config)
    scaled_momentum = momentum / (2 * system.mass * energy_per_state)
    possible_thresholds = 0.5 * (scaled_momentum[:-1] + scaled_momentum[1:])

    min_threshold = 1e-8
    possible_thresholds = possible_thresholds[possible_thresholds > min_threshold]
    max_threshold = 0.5
    possible_thresholds = possible_thresholds[possible_thresholds < max_threshold]

    optimal_idx = 0
    optimal_loss = loss_function(possible_thresholds[optimal_idx])
    for i in range(1, len(possible_thresholds)):
        if loss_function(possible_thresholds[i]) < optimal_loss:
            optimal_idx = i
            optimal_loss = loss_function(possible_thresholds[optimal_idx])

    optimal_threshold = float(possible_thresholds[optimal_idx])
    print(  # noqa: T201
        f"Optimal threshold: {optimal_threshold:.2e} "
        f"({optimal_idx}/{len(possible_thresholds)})",
    )

    total_occupation, effective_mass = get_scaled_momentum_threshold_effective_mass(
        system,
        config,
        threshold=optimal_threshold,
    )
    return total_occupation, effective_mass / system.mass


def _get_optimal_threshold_mass_ratio_alt(
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

            total_occupation, effective_mass = _get_zero_threshold_mass_ratio(
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

            total_occupation, effective_mass = _get_optimal_threshold_mass_ratio_alt(
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

            total_occupation, effective_mass = _get_zero_threshold_mass_ratio(
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

            total_occupation, effective_mass = _get_optimal_threshold_mass_ratio_alt(
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
    zero_threshold_mass_ratios: tuple[np.ndarray, np.ndarray]
    optimal_threshold_mass_ratios: tuple[np.ndarray, np.ndarray]
    optional_threshold_mass_ratios_alt: tuple[np.ndarray, np.ndarray]


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
        "zero_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optimal_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optional_threshold_mass_ratios_alt": (
            np.zeros_like(xv),
            np.zeros_like(xv),
        ),
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
                out["zero_threshold_mass_ratios"][0][i],
                out["zero_threshold_mass_ratios"][1][i],
            ) = _get_zero_threshold_mass_ratio(
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

            (
                out["optional_threshold_mass_ratios_alt"][0][i],
                out["optional_threshold_mass_ratios_alt"][1][i],
            ) = _get_optimal_threshold_mass_ratio_alt(
                system,
                config,
                t_factor=6,
            )

            get_ordered_momentum.delete_cache(system, config)
    return out


def _plot_isf_mass_ratios() -> None:

    data = get_all_mass_ratios()
    xv, yv, shape, zero_threshold_mass_ratios, optimal_threshold_mass_ratios = (
        data["xv"],
        data["yv"],
        data["shape"],
        data["zero_threshold_mass_ratios"],
        data["optional_threshold_mass_ratios_alt"],
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
    fig.savefig("scripts/perturbation/effective_mass.zero.pdf")


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
        "zero_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optimal_threshold_mass_ratios": (np.zeros_like(xv), np.zeros_like(xv)),
        "optional_threshold_mass_ratios_alt": (
            np.zeros_like(xv),
            np.zeros_like(xv),
        ),
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
                out["zero_threshold_mass_ratios"][0][i],
                out["zero_threshold_mass_ratios"][1][i],
            ) = _get_zero_threshold_mass_ratio(
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

            (
                out["optional_threshold_mass_ratios_alt"][0][i],
                out["optional_threshold_mass_ratios_alt"][1][i],
            ) = _get_optimal_threshold_mass_ratio_alt(
                system,
                config,
                t_factor=6,
            )

            get_ordered_momentum.delete_cache(system, config)
    return out


def _plot_isf_mass_ratios_2d() -> None:

    data = get_all_mass_ratios_2d()
    xv, yv, shape, zero_threshold_mass_ratios, optimal_threshold_mass_ratios = (
        data["xv"],
        data["yv"],
        data["shape"],
        data["zero_threshold_mass_ratios"],
        data["optional_threshold_mass_ratios_alt"],
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
    fig.savefig("scripts/perturbation/effective_mass.zero.2d.pdf")


def _plot_isf_mass_fit_1d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["zero", "optimal", "alt"] = "optimal",
) -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(energy_factor * system.barrier_energy)
    system = system.with_mass(mass_factor * system.mass)

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=temperature,
    )

    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )

    get_ordered_momentum.load_or_call_cached(system, config)
    isf = get_weak_boltzmann_isf.call_uncached(system, config, times)
    if ty == "alt":
        total_occupation, effective_mass = _get_optimal_threshold_mass_ratio_alt(
            system,
            config,
            t_factor=6,
        )
    elif ty == "zero":
        total_occupation, effective_mass = _get_zero_threshold_mass_ratio(
            system,
            config,
        )
    else:
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

    fig.savefig(f"scripts/perturbation/effective_mass.fit.{ty}.1d.pdf")


def _plot_isf_mass_fit_2d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["zero", "optimal", "alt"] = "optimal",
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
    if ty == "alt":
        total_occupation, effective_mass = _get_optimal_threshold_mass_ratio_alt(
            system,
            config,
            t_factor=6,
        )
    elif ty == "zero":
        total_occupation, effective_mass = _get_zero_threshold_mass_ratio(
            system,
            config,
        )
    else:
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


if __name__ == "__main__":
    _assess_isf_validity()
    _assess_isf_validity_2d()
    _plot_isf_mass_ratios()
    _plot_isf_mass_ratios_2d()
    _print_free_times()
    _plot_isf_mass_fit_1d(ty="zero")
    _plot_isf_mass_fit_1d(ty="alt")
    _plot_isf_mass_fit_2d(ty="zero")
    _plot_isf_mass_fit_2d(ty="alt")

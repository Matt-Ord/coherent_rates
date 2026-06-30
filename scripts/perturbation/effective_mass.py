import itertools
from pathlib import Path
from typing import Literal, TypedDict

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
    get_thesis_figure,
)


def _get_zero_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:

    total_occupation, effective_mass = get_momentum_threshold_effective_mass(
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
    xv, yv, shape, zero_threshold_mass_ratios, optimal_threshold_mass_ratios = (
        data["xv"],
        data["yv"],
        data["shape"],
        data["zero_threshold_mass_ratios"],
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
    ax.set_title("Intrinsic Mass")
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
    ax.set_title("Empirical Mass")

    ax.set_xlim(np.min(xv), np.max(xv))
    ax.set_ylim(np.min(yv), np.max(yv))

    fig.colorbar(mesh, ax=ax)
    fig.savefig("scripts/perturbation/effective_mass.zero.pdf")


def _plot_isf_mass_fit_1d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["zero", "optimal"] = "optimal",
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

    isf = get_weak_boltzmann_isf(system, config, times)
    fig, ax = get_thesis_figure()

    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_color(CAM_CHERRY.base)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    get_ordered_momentum.load_or_call_cached(system, config)
    total_occupation, effective_mass = (
        _get_zero_threshold_mass_ratio(system, config)
        if ty == "zero"
        else _get_optimal_threshold_mass_ratio(system, config)
    )
    get_ordered_momentum.delete_cache(system, config)

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

    ax.legend(
        loc="upper right",
        handles=[line_real_mass, line_effective_mass, line_first_order],
        fontsize=9,
    )

    if ty == "optimal":
        ax.set_ylim(((1 - 1.1 * total_occupation), 1))
    else:
        ax.set_ylim((0.8 * np.min(np.abs(isf["data"]))), 1)

    fig.savefig(f"scripts/perturbation/effective_mass.fit.{ty}.pdf")


def _plot_isf_mass_fit_2d(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
    ty: Literal["zero", "optimal"] = "optimal",
) -> None:

    system = SODIUM_COPPER_SYSTEM_2D
    system = system.with_barrier_energy(energy_factor * system.barrier_energy)
    system = system.with_mass(mass_factor * system.mass)

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    config = config.with_temperature(temperature)
    get_hamiltonian.load_or_call_cached(system, config)

    times = GaussianMethod(measure="abs", t_factor=32).get_fit_times(
        system=system,
        config=config,
    )

    isf = get_weak_boltzmann_isf(system, config, times)
    fig, ax = get_thesis_figure()

    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_color(CAM_CHERRY.base)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    get_ordered_momentum.load_or_call_cached(system, config)
    total_occupation, effective_mass = (
        _get_zero_threshold_mass_ratio(system, config)
        if ty == "zero"
        else _get_optimal_threshold_mass_ratio(system, config, t_factor=32)
    )
    get_ordered_momentum.delete_cache(system, config)

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

    ax.legend(
        loc="upper right",
        handles=[line_real_mass, line_effective_mass, line_first_order],
        fontsize=9,
    )

    if ty == "optimal":
        ax.set_ylim(((1 - 1.1 * total_occupation), 1))
    else:
        ax.set_ylim((0.8 * np.min(np.abs(isf["data"]))), 1)

    fig.savefig(f"scripts/perturbation/effective_mass.fit.{ty}.2d.pdf")


if __name__ == "__main__":
    _assess_isf_validity()
    _plot_isf_mass_ratios()
    _plot_isf_mass_fit_1d(ty="zero")
    _plot_isf_mass_fit_1d(ty="optimal")
    _plot_isf_mass_fit_2d(ty="zero")
    _plot_isf_mass_fit_2d(ty="optimal")

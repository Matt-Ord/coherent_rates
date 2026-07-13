from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import scipy
import scipy.optimize
from surface_potential_analysis.basis.momentum_basis_like import MomentumBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_momentum,
)
from surface_potential_analysis.util.decorators import cached

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    DoubleGaussianMethod,
    GaussianMethod,
    get_free_particle_rate,
)
from coherent_rates.isf import (
    SimulationCondition,
    get_boltzmann_rate_against_momentum_data,
    get_conditions_at_barrier_energy,
    get_conditions_at_mass,
    get_conditions_at_temperatures,
    get_weak_boltzmann_rate,
    get_weak_boltzmann_rate_against_momentum_data,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    FreeSystem,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_PURPLE,
    format_axis_scientific,
    get_paper_figure,
    get_thesis_figure,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList


def select_idx(
    rates: ValueList[MomentumBasis],
    idx: list[int],
) -> ValueList[MomentumBasis]:
    selected_momenta = rates["data"][idx]
    return {
        "basis": MomentumBasis(rates["basis"].k_points[idx]),
        "data": selected_momenta,
    }


def _2d_boltzmann_rate_thesis() -> None:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_thesis_figure()

    directions = [(i, i) for i in range(1, 7)]
    data_111_double = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=DoubleGaussianMethod(measure="abs", ty="Fast"),
        directions=directions,
        n_repeats=20,
    )
    data_111_double = select_idx(data_111_double, list(range(6)))
    fig, ax, line = plot_value_list_against_momentum(data_111_double, ax=ax)
    line.set_label("111")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_PURPLE.base)

    directions = [(i, -i) for i in range(1, 7)]
    data_112_double = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=DoubleGaussianMethod(measure="abs", ty="Fast"),
        directions=directions,
        n_repeats=20,
    )
    data_112_double = select_idx(data_112_double, list(range(3)))
    fig, ax, line = plot_value_list_against_momentum(data_112_double, ax=ax)
    line.set_label("$11\\bar{2}$")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_CHERRY.base)
    line.set_alpha(1.0)

    directions = [(i, i) for i in range(1, 16)]
    data_111_single = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    data_111_single = select_idx(data_111_single, list(range(6, 15)))
    fig, ax, line = plot_value_list_against_momentum(data_111_single, ax=ax)
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_PURPLE.base)

    directions = [(i, -i) for i in range(1, 16)]
    data_112_single = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    data_112_single = select_idx(data_112_single, list(range(3, 15)))
    fig, ax, line = plot_value_list_against_momentum(data_112_single, ax=ax)
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_CHERRY.base)

    free = FreeSystem(system)
    directions = [(i, i) for i in range(1, 16)]
    data_112_free = get_boltzmann_rate_against_momentum_data(
        free,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    fig, ax, line = plot_value_list_against_momentum(data_112_free, ax=ax)
    line.set_label("Free system")
    line.set_color(CAM_BLUE.warm)
    line.set_linestyle("")
    line.set_marker("x")

    ax.set_xlim(0, 1.5e10)
    ax.set_ylim(0, 1e13)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.thesis.pdf")


def _2d_boltzmann_rate_paper() -> None:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_paper_figure()

    directions = [(i, i) for i in range(1, 7)]
    data_111_double = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=DoubleGaussianMethod(measure="abs", ty="Fast"),
        directions=directions,
        n_repeats=20,
    )
    data_111_double = select_idx(data_111_double, list(range(6)))
    fig, ax, line = plot_value_list_against_momentum(data_111_double, ax=ax)
    line.set_label("111")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_PURPLE.base)

    directions = [(i, -i) for i in range(1, 7)]
    data_112_double = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=DoubleGaussianMethod(measure="abs", ty="Fast"),
        directions=directions,
        n_repeats=20,
    )
    data_112_double = select_idx(data_112_double, list(range(3)))
    fig, ax, line = plot_value_list_against_momentum(data_112_double, ax=ax)
    line.set_label("$11\\bar{2}$")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_CHERRY.base)
    line.set_alpha(1.0)

    directions = [(i, i) for i in range(1, 16)]
    data_111_single = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    data_111_single = select_idx(data_111_single, list(range(6, 15)))
    fig, ax, line = plot_value_list_against_momentum(data_111_single, ax=ax)
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_PURPLE.base)

    directions = [(i, -i) for i in range(1, 16)]
    data_112_single = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    data_112_single = select_idx(data_112_single, list(range(3, 15)))
    fig, ax, line = plot_value_list_against_momentum(data_112_single, ax=ax)
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_CHERRY.base)

    free = FreeSystem(system)
    directions = [(i, i) for i in range(1, 16)]
    data_112_free = get_boltzmann_rate_against_momentum_data(
        free,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    fig, ax, line = plot_value_list_against_momentum(data_112_free, ax=ax)
    line.set_label("Free system")
    line.set_color(CAM_BLUE.warm)
    line.set_linestyle("")
    line.set_marker("x")

    ax.set_xlim(0, 1.5e10)
    ax.set_ylim(0, 1e13)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.paper.pdf")


def _2d_boltzmann_rate_weak() -> None:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_thesis_figure()

    directions = [(i, i) for i in range(1, 16)]
    data_111_double = get_weak_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        directions=directions,
    )
    fig, ax, line = plot_value_list_against_momentum(data_111_double, ax=ax)
    line.set_label("111")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_PURPLE.base)

    directions = [(i, -i) for i in range(1, 16)]
    data_112_double = get_weak_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        directions=directions,
    )
    fig, ax, line = plot_value_list_against_momentum(data_112_double, ax=ax)
    line.set_label("$11\\bar{2}$")
    line.set_linestyle("")
    line.set_marker("x")
    line.set_color(CAM_CHERRY.base)
    line.set_alpha(1.0)

    free = FreeSystem(system)
    directions = [(i, i) for i in range(1, 16)]
    data_112_free = get_weak_boltzmann_rate_against_momentum_data(
        free,
        config,
        fit_method=GaussianMethod(measure="abs"),
        directions=directions,
    )
    fig, ax, line = plot_value_list_against_momentum(data_112_free, ax=ax)
    line.set_label("Free system")
    line.set_color(CAM_BLUE.warm)
    line.set_linestyle("")
    line.set_marker("x")

    ax.set_xlim(0, 2e10)
    ax.set_ylim(0, 0.6e13)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.weak.pdf")


class _RatesData(TypedDict):
    data: list[float]
    conditions: list[SimulationCondition]


@cached(Path("data/boltzmann_mass.2d.rate_vs_mass"))
def _get_cached_rates() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    masses = [system.mass * t for t in [0.25, 0.5, *range(1, 6)]]
    conditions = get_conditions_at_mass(
        system,
        config,
        masses=masses,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


@cached(Path("data/boltzmann_mass.2d.rate_vs_mass.hi_res"))
def _get_cached_rates_hi_res() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (45, 45),
        direction=(1, 0),
        truncation=800,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    masses = [system.mass * t for t in [0.25, 0.5, *range(1, 8)]]
    conditions = get_conditions_at_mass(
        system,
        config,
        masses=masses,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


def _2d_effective_mass_vs_mass() -> None:
    data = _get_cached_rates()
    data_hi_res = _get_cached_rates_hi_res()
    conditions = data["conditions"]
    masses = [condition[0].mass for condition in conditions]
    rates = data["data"]

    free_rates = [
        get_free_particle_rate(system, config) for (system, config, _) in conditions
    ]

    fig, ax = get_thesis_figure()
    (line,) = ax.plot(masses, rates, marker="x", linestyle="")
    line.set_label("Actual rate")
    (line,) = ax.plot(masses, free_rates, linestyle="--")
    line.set_label("Free particle rate")
    (line,) = ax.plot(
        [condition[0].mass for condition in data_hi_res["conditions"]],
        data_hi_res["data"],
        marker="x",
        linestyle="",
    )
    line.set_label("Actual rate (high res)")
    ax.set_xlabel("Mass / kg")
    ax.set_ylabel("Rate / $s^{-1}$")
    ax.set_title("Rate against mass for 2D system")
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/boltzmann_mass.2d.rate_vs_mass.pdf")

    fig, ax = get_thesis_figure()
    inverse_rates = [c**-2 for c in rates]
    (line,) = ax.plot(masses, inverse_rates, marker="x", linestyle="")
    line.set_label("Actual")

    (line,) = ax.plot(
        [condition[0].mass for condition in data_hi_res["conditions"]],
        [c**-2 for c in data_hi_res["data"]],
        marker="x",
        linestyle="",
    )
    line.set_label("Actual (high res)")

    # Fit to a straight line
    def _model(x: float, a: float) -> float:
        return a * x

    p_opt, _ = scipy.optimize.curve_fit(_model, masses, inverse_rates)
    (line,) = ax.plot(
        masses,
        [_model(x, p_opt[0]) for x in masses],
        marker="",
        linestyle="--",
    )
    line.set_label(f"Fit to $Y = {p_opt[0]:.2e} x$")

    (line,) = ax.plot(masses, [c**-2 for c in free_rates], linestyle="--")
    line.set_label("Free particle rate")
    ax.set_xlabel("Mass / kg")
    ax.set_ylabel("Rate$^{-2}$ / $s^2$")
    ax.set_title("Rate against mass for 2D system")
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/boltzmann_mass.2d.rate_vs_mass.inverse.pdf")

    effective_mass = [
        (free / actual) ** 2 for free, actual in zip(free_rates, rates, strict=False)
    ]
    fig, ax = get_thesis_figure()
    ax.plot(masses, effective_mass, marker="x", linestyle="")
    hi_res_effective_mass = [
        (get_free_particle_rate(c[0], c[1]) / actual) ** 2
        for c, actual in zip(
            data_hi_res["conditions"],
            data_hi_res["data"],
            strict=False,
        )
    ]
    ax.plot(
        [condition[0].mass for condition in data_hi_res["conditions"]],
        hi_res_effective_mass,
        marker="x",
        linestyle="",
    )
    ax.set_xlabel("Mass / kg")
    ax.set_ylabel("Effective mass Factor")
    ax.set_title("Effective mass against mass for 2D system")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.effective_mass_vs_mass.pdf")


@cached(Path("data/boltzmann_mass.2d.rate_vs_temperature"))
def _get_cached_rates_vs_temperature() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    temperatures = [
        config.temperature * t
        for t in [
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]
    ]
    conditions = get_conditions_at_temperatures(
        system,
        config,
        temperatures=temperatures,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


@cached(Path("data/boltzmann_mass.2d.rate_vs_temperature.hi_res"))
def _get_cached_rates_vs_temperature_hi_res() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (45, 45),
        direction=(1, 0),
        truncation=800,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    temperatures = [
        config.temperature * t
        for t in [
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]
    ]
    conditions = get_conditions_at_temperatures(
        system,
        config,
        temperatures=temperatures,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


def _2d_effective_mass_vs_temperature() -> None:
    data = _get_cached_rates_vs_temperature()
    high_res_data = _get_cached_rates_vs_temperature_hi_res()
    conditions = data["conditions"]
    temperatures = [condition[1].temperature for condition in conditions]
    rates = data["data"]

    free_rates = [
        get_free_particle_rate(system, config) for (system, config, _) in conditions
    ]

    fig, ax = get_thesis_figure()
    (line,) = ax.plot(temperatures, rates, marker="x", linestyle="")
    line.set_label("Actual rate")
    (line,) = ax.plot(temperatures, high_res_data["data"], marker="x", linestyle="")
    line.set_label("Actual rate (high res)")
    (line,) = ax.plot(temperatures, free_rates, linestyle="--")
    line.set_label("Free particle rate")
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Rate / $s^{-1}$")
    ax.set_title("Rate against temperature for 2D system")
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/boltzmann_mass.2d.rate_vs_temperature.pdf")

    effective_mass = [
        (free / actual) ** 2 for free, actual in zip(free_rates, rates, strict=False)
    ]
    fig, ax = get_thesis_figure()
    ax.plot(temperatures, effective_mass, marker="x", linestyle="")
    hi_res_effective_mass = [
        (free / actual) ** 2
        for free, actual in zip(
            free_rates,
            high_res_data["data"],
            strict=False,
        )
    ]
    ax.plot(temperatures, hi_res_effective_mass, marker="x", linestyle="")
    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Effective mass Factor")
    ax.set_title("Effective mass against temperature for 2D system")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.effective_mass_vs_temperature.pdf")


@cached(Path("data/boltzmann_mass.2d.rate_vs_barrier"))
def _get_cached_rates_vs_barrier() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    barrier_energies = [
        system.barrier_energy * f
        for f in [
            0,
            0.1,
            0.25,
            0.5,
            1,
            2,
            4,
        ]
    ]
    conditions = get_conditions_at_barrier_energy(
        system,
        config,
        barrier_energies=barrier_energies,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=8.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


@cached(Path("data/boltzmann_mass.2d.rate_vs_barrier.slow"))
def _get_cached_rates_vs_barrier_slow() -> _RatesData:
    config = PeriodicSystemConfig(
        (20, 20),
        (45, 45),
        direction=(1, 0),
        truncation=800,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    barrier_energies = [
        system.barrier_energy * f
        for f in [
            # 0,
            # 0.1,
            # 0.25,
            # 0.5,
            1,
            2,
            4,
        ]
    ]
    conditions = get_conditions_at_barrier_energy(
        system,
        config,
        barrier_energies=barrier_energies,
    )

    rates = []
    for condition in conditions:
        rate = get_weak_boltzmann_rate(
            condition[0],
            condition[1],
            fit_method=GaussianMethod(measure="abs", t_factor=32.0),
        )
        rates.append(rate)

    return {"data": rates, "conditions": conditions}


def _2d_effective_mass_vs_barrier() -> None:
    data = _get_cached_rates_vs_barrier()
    data_slow = _get_cached_rates_vs_barrier_slow()
    conditions = data["conditions"]
    barrier_energies = [condition[0].barrier_energy for condition in conditions]
    rates = data["data"]

    free_rates = [
        get_free_particle_rate(system, config) for (system, config, _) in conditions
    ]

    fig, ax = get_thesis_figure()
    (line,) = ax.plot(barrier_energies, rates, marker="x", linestyle="")
    line.set_label("Actual rate")
    (line,) = ax.plot(barrier_energies, free_rates, linestyle="--")
    line.set_label("Free particle rate")
    ax.set_xlabel("Barrier energy / J")
    ax.set_ylabel("Rate / $s^{-1}$")
    ax.set_title("Rate against barrier energy for 2D system")
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/boltzmann_mass.2d.rate_vs_barrier.pdf")

    effective_mass = [
        (free / actual) ** 2 for free, actual in zip(free_rates, rates, strict=False)
    ]
    fig, ax = get_thesis_figure()
    ax.plot(barrier_energies, effective_mass, marker="x", linestyle="")
    ax.plot(
        [condition[0].barrier_energy for condition in data_slow["conditions"]],
        [
            (get_free_particle_rate(c[0], c[1]) / actual) ** 2
            for c, actual in zip(
                data_slow["conditions"],
                data_slow["data"],
                strict=False,
            )
        ],
        marker="x",
        linestyle="",
    )
    ax.set_xlabel("Barrier energy / J")
    ax.set_ylabel("Effective mass Factor")
    ax.set_title("Effective mass against barrier energy for 2D system")
    fig.savefig("scripts/thesis/boltzmann_mass.2d.effective_mass_vs_barrier.pdf")


if __name__ == "__main__":
    # _2d_boltzmann_rate()
    _2d_boltzmann_rate_paper()
    # _2d_boltzmann_rate_weak()
    # _2d_effective_mass_vs_mass()
    # _2d_effective_mass_vs_temperature()
    # _2d_effective_mass_vs_barrier()

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import electron_volt  # type:ignore lib

from coherent_rates.config import ExponentialInstrumentFunction, PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
    GaussianMethodWithOffset,
)
from coherent_rates.plot import (
    plot_boltzmann_isf_fit_for_directions,
    plot_boltzmann_rate_against_momentum,
    plot_local_boltzmann_rate_against_momentum,
)
from coherent_rates.state import UniformXLocalizationStrategy
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    FreeSystem,
    System,
)
from scripts.thesis.util import (
    CAM_BLUE,
    format_axis_scientific,
    get_fancy_figure,
    setup_rc_params,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from coherent_rates.fit import FitMethod


setup_rc_params()


def _test_convergence_with_shape(
    system: System,
    config: PeriodicSystemConfig,
    directions: list[tuple[int, ...]],
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fig, ax, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
    )
    line.set_label("Standard")
    directions = [tuple(2 * j for j in i) for i in directions]
    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config.with_shape(tuple(2 * j for j in config.shape)),
        directions=directions,
        fit_method=fit_method,
        ax=ax,
    )
    line.set_label("2x shape")
    ax.legend()  # type: ignore unknown
    fig.show()


def _test_convergence_with_resolution(
    system: System,
    config: PeriodicSystemConfig,
    directions: list[tuple[int, ...]],
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fig, ax, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
    )
    line.set_label("Standard")

    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config.with_resolution(tuple(2 * j for j in config.resolution)),
        directions=directions,
        fit_method=fit_method,
        ax=ax,
    )
    line.set_label("2x resolution")
    ax.legend()  # type: ignore unknown
    fig.show()


def _test_convergence_with_truncation(
    system: System,
    config: PeriodicSystemConfig,
    directions: list[tuple[int, ...]],
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fig, ax, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
    )
    line.set_label("Standard")

    config = config.with_resolution(
        tuple(2 * j for j in config.resolution),
    ).with_truncation(2 * config.n_bands)
    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
        ax=ax,
    )
    line.set_label("2x truncation")
    ax.legend()  # type: ignore unknown
    fig.show()


def _compare_rate_against_free_surface(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    free_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Figure:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    free_fit_method = GaussianMethod() if free_fit_method is None else free_fit_method

    fig, ax = get_fancy_figure()

    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Bound system")
    line.set_color(CAM_BLUE.dark)

    _, _, line = plot_boltzmann_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=free_fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Free system")
    line.set_color(CAM_BLUE.warm)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_title("")
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")

    return fig


def _compare_rate_against_free_surface_localized(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    free_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Figure:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    free_fit_method = GaussianMethod() if free_fit_method is None else free_fit_method

    fig, ax = get_fancy_figure()

    sigma_0 = (system.lattice_constant / 20,)
    _, _, line = plot_local_boltzmann_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
        strategy=UniformXLocalizationStrategy(
            delta_x=np.array((system.lattice_constant,)),
            sigma_0=sigma_0,
        ),
    )
    line.set_label("Bound system")
    line.set_color(CAM_BLUE.dark)

    _, _, line = plot_boltzmann_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=free_fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Free system")
    line.set_color(CAM_BLUE.warm)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_title("")
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")

    return fig


def _compare_rate_against_corrected_surface(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    corrected_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Figure:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    corrected_fit_method = (
        GaussianMethod() if corrected_fit_method is None else corrected_fit_method
    )
    fig, ax = get_fancy_figure()

    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Ideal System")
    line.set_color(CAM_BLUE.warm)

    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        dataclasses.replace(
            config,
            instrument_function=ExponentialInstrumentFunction(
                width=8.03 * 10**-3 * electron_volt,
                optimal_energy_out=7.7 * 10**-3 * electron_volt,
                incoming_energy=8 * 10**-3 * electron_volt,
            ),
        ),
        fit_method=corrected_fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Corrected System")
    line.set_color(CAM_BLUE.dark)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_title("")
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")

    return fig


def _compare_rate_against_corrected_surface_localized(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    corrected_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Figure:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    corrected_fit_method = (
        GaussianMethod() if corrected_fit_method is None else corrected_fit_method
    )
    fig, ax = get_fancy_figure()

    _, _, line = plot_boltzmann_rate_against_momentum(
        dataclasses.replace(
            system,
            barrier_energy=0.0,
        ),
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Ideal System")
    line.set_color(CAM_BLUE.warm)

    _, _, line = plot_local_boltzmann_rate_against_momentum(
        system,
        dataclasses.replace(
            config,
            instrument_function=ExponentialInstrumentFunction(
                width=8.03 * 10**-3 * electron_volt,
                optimal_energy_out=7.7 * 10**-3 * electron_volt,
                incoming_energy=8 * 10**-3 * electron_volt,
            ),
        ),
        fit_method=corrected_fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Corrected System")
    line.set_color(CAM_BLUE.dark)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_title("")
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")

    return fig


def _compare_rate_against_localized_surface(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    localized_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Figure:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    localized_fit_method = (
        GaussianMethod() if localized_fit_method is None else localized_fit_method
    )
    fig, ax = get_fancy_figure()

    _, _, line = plot_boltzmann_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Standard System")
    line.set_color(CAM_BLUE.warm)
    sigma_0 = (system.lattice_constant / 20,)
    _, _, line = plot_local_boltzmann_rate_against_momentum(
        system,
        config,
        fit_method=localized_fit_method,
        directions=directions,
        ax=ax,
        strategy=UniformXLocalizationStrategy(
            delta_x=np.array((system.lattice_constant,)),
            sigma_0=sigma_0,
        ),
    )
    line.set_label("Localized System")
    line.set_color(CAM_BLUE.dark)

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    format_axis_scientific(ax.xaxis)
    ax.set_title("")
    ax.set_ylabel(r"Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel(r"$\Delta k$ / $\mathrm{m}^{-1}$")

    return fig


if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (400,),
        (100,),
        truncation=25,
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    print(system.barrier_energy)  # noqa: T201
    directions = [(i,) for i in [1, 2, *list(range(5, 155, 5))]]

    if False:
        _test_convergence_with_shape(
            system,
            config,
            directions=directions,
            fit_method=GaussianMethod(measure="abs"),
        )
        _test_convergence_with_resolution(
            system,
            config,
            directions=directions,
            fit_method=GaussianMethod(measure="abs"),
        )
        _test_convergence_with_truncation(
            system,
            config,
            directions=directions,
            fit_method=GaussianMethod(measure="abs"),
        )
        plot_boltzmann_isf_fit_for_directions(
            system,
            config,
            directions=directions,
            fit_method=GaussianMethod(measure="abs"),
        )
    fig = _compare_rate_against_free_surface(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        free_fit_method=GaussianMethod(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass.1d.pdf")

    fig = _compare_rate_against_free_surface_localized(
        FreeSystem(system),
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        free_fit_method=GaussianMethod(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass.local.1d.pdf")
    fig = _compare_rate_against_corrected_surface(
        dataclasses.replace(
            system,
            barrier_energy=0.0,
        ),
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        corrected_fit_method=GaussianMethodWithOffset(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass.1d.corrected.free.pdf")
    fig = _compare_rate_against_corrected_surface_localized(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        corrected_fit_method=GaussianMethodWithOffset(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass.local.1d.corrected.free.pdf")
    fig = _compare_rate_against_localized_surface(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        localized_fit_method=GaussianMethod(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass_vs_localized.1d.pdf")

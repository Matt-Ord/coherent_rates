import dataclasses
from typing import Any

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.constants import (
    Boltzmann,
    hbar,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_analytical_isf,
    get_boltzmann_isf,
    get_local_boltzmann_isf,
    get_scattered_momentum,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.state import LocalizationParams, ThermalLocalizationStrategy
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    PeriodicSystem1d,
)
from scripts.thesis.bandstructure_plot import CAM_DARK_BLUE, CAM_SLATE_1, CAM_WARM_BLUE
from scripts.thesis.util import (
    format_axis_scientific,
    get_fancy_figure,
    setup_rc_params,
)

setup_rc_params()


def plot_periodic_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(100,),
        truncation=25,
        temperature=155,
    )
    times = EvenlySpacedTimeBasis(100, 1, 0, 1.5e-10)
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=100)

    fig, ax = get_fancy_figure()

    fit = GaussianMethod(measure="abs").get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = GaussianMethod.get_fitted_data(fit, isf["basis"])

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="center right",
        fontsize=9,
        bbox_to_anchor=(1.0, 0.6),
    )
    legend.get_frame().set_alpha(0)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower left",
        borderpad=1.0,
        bbox_to_anchor=(0.1, 0, 1, 1),
        bbox_transform=ax.transAxes,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)

    inset_ax.set_facecolor((0, 0, 0, 0))
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9, labelpad=-1)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.periodic.pdf")


class PreferentailLocalizationStrategy(ThermalLocalizationStrategy):
    """Localization strategy that preferentially localizes in potential wells."""

    def generate_params(self) -> LocalizationParams:
        rng = np.random.default_rng()
        parent_params = super().generate_params()

        barrier_energy = self.system.barrier_energy
        # Ratio between inside and outside
        ratio = np.exp(-2 * barrier_energy / (Boltzmann * self.config.temperature))
        x_0_bridge = (0.0,)
        x_0_hollow = (0.0,)

        x_0 = rng.choice(
            [x_0_bridge, x_0_hollow],
            p=[ratio / (1 + ratio), 1 / (1 + ratio)],
        )

        jitter = rng.random(len(x_0)) * (0.05 * self.system.lattice_constant)
        return dataclasses.replace(
            parent_params,
            x_0=tuple(x + j for x, j in zip(x_0, jitter, strict=True)),
        )

    def __hash__(self) -> int:
        return hash((-11, self.system, self.config, self.sigma_0))


def plot_periodic_local_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(400,),
        truncation=25,
        temperature=155,
    )

    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )

    strategy = ThermalLocalizationStrategy(
        system=system,
        config=config,
        sigma_0=(system.lattice_constant / 30,),
    )
    isf = get_local_boltzmann_isf.call_cached(
        system,
        config,
        times,
        n_repeats=1000,
        strategy=strategy,
    )

    fig, ax = get_fancy_figure()
    measure = "abs"
    fit = GaussianMethod(measure=measure, truncate=True).get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )

    fitted_data = GaussianMethod().get_fitted_data(fit, isf["basis"])
    fig, ax, line = plot_value_list_against_time(isf, measure=measure, ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color("pink")
    line.set_linestyle("--")

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    isf_nonlocal = get_boltzmann_isf(system, config, times, n_repeats=100)
    fig, ax, line = plot_value_list_against_time(isf_nonlocal, measure=measure, ax=ax)
    line.set_label("Simulated (NL)")
    line.set_color(CAM_DARK_BLUE)

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="center right",
        fontsize=9,
        bbox_to_anchor=(1.0, 0.6),
    )
    legend.get_frame().set_alpha(0)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower left",
        borderpad=1.0,
        bbox_to_anchor=(0.1, 0, 1, 1),
        bbox_transform=ax.transAxes,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)
    _, _, inset_line = plot_value_list_against_time(
        isf_nonlocal,
        measure="angle",
        ax=inset_ax,
    )
    inset_line.set_color(CAM_DARK_BLUE)

    inset_ax.set_facecolor((0, 0, 0, 0))
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9, labelpad=-1)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.periodic.local.pdf")


def _get_occupation_probabilities(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the occupation probability for a given system, configuration, and time."""
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * config.temperature),
    )
    return boltzmann_distribution / np.sum(boltzmann_distribution)


def get_occupation_loss(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
) -> float:
    """Get the occupation loss for a given system and configuration."""
    config_full = config.with_truncation(None)
    probabilities = _get_occupation_probabilities(system, config_full)
    total_p = np.sum(probabilities.reshape(config_full.n_bands, -1)[: config.n_bands])
    return np.real(1 - total_p.item())


def get_max_simulation_time(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
) -> float:
    """Get the occupation loss for a given system and configuration."""
    n_b = config.n_bands
    velocity = (2 * hbar * n_b * np.pi) / (system.mass * system.lattice_constant)
    return (config.shape[0] * system.lattice_constant) / velocity


def get_max_simulation_time_thermal(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
) -> float:
    """Get the occupation loss for a given system and configuration."""
    velocity = (system.mass / (2 * Boltzmann * config.temperature)) ** 0.5
    return (config.shape[0] * system.lattice_constant) * velocity


def get_band_energy(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
    band_index: int,
) -> float:
    """Get the energy of a given band index."""
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    energies = hamiltonian["data"].reshape(config.n_bands, -1)
    return np.real(energies[band_index, 0].item())


def plot_free_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(2,),
        truncation=25,
        temperature=155,
    )
    print(f"Missing Occupation {get_occupation_loss(system, config):0.3e}")  # noqa: T201
    print("Max Band Energy Level:")  # noqa: T201
    print(f"{get_band_energy(system, config, config.n_bands - 1):0.3e}")  # noqa: T201
    print(f"Max Simulation Time {get_max_simulation_time(system, config):0.3e}")  # noqa: T201
    print(f"Max Simulation Time {get_max_simulation_time_thermal(system, config):0.3e}")  # noqa: T201

    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201
    print("Decay after 1e-10s")  # noqa: T201
    decayed_isf = np.exp(
        -(Boltzmann * config.temperature * (1e-10 * delta_k) ** 2) / (2 * system.mass),
    )
    print(f"I: {decayed_isf:0.3e}")  # noqa: T201
    times = EvenlySpacedTimeBasis(100, 1, 0, 1.5e-10)
    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    fig, ax = get_fancy_figure()
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)
    fig, ax, line = plot_value_list_against_time(
        analytical_isf,
        ax=ax,
        measure="abs",
    )
    line.set_label("Analytical")
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")
    ax.set_xlim(0, 1.5e-10)

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower right",
        borderpad=1.0,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)
    _, _, inset_line = plot_value_list_against_time(
        analytical_isf,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_DARK_BLUE)
    inset_line.set_linestyle("--")
    inset_ax.set_ylim(0, 1.1 * np.max(np.angle(analytical_isf["data"])))
    inset_ax.set_xlim(ax.get_xlim())
    inset_ax.set_facecolor(CAM_SLATE_1)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9)

    format_axis_scientific(inset_ax.yaxis)

    fig.savefig("scripts/thesis/boltzmann_isf.free.pdf")


def plot_free_local_isf() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(100,),
        truncation=25,
        temperature=155,
    )
    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    strategy = ThermalLocalizationStrategy(
        system=system,
        config=config,
        sigma_0=(system.lattice_constant / 14,),
    )
    isf = get_local_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
        strategy=strategy,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    fig, ax = get_fancy_figure()
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_WARM_BLUE)
    fig, ax, line = plot_value_list_against_time(
        analytical_isf,
        ax=ax,
        measure="abs",
    )
    line.set_label("Analytical")
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="lower right",
        borderpad=1.0,
    )
    _, _, inset_line = plot_value_list_against_time(isf, measure="angle", ax=inset_ax)
    inset_line.set_color(CAM_WARM_BLUE)
    _, _, inset_line = plot_value_list_against_time(
        analytical_isf,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_DARK_BLUE)
    inset_line.set_linestyle("--")
    inset_ax.set_ylim(0, 1.1 * np.max(np.angle(analytical_isf["data"])))
    inset_ax.set_xlim(ax.get_xlim())
    inset_ax.set_facecolor(CAM_SLATE_1)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel(r"$\arg{(I(\Delta k, t))}$", fontsize=9)

    format_axis_scientific(inset_ax.yaxis)

    fig.savefig("scripts/thesis/boltzmann_isf.local.free.pdf")


if __name__ == "__main__":
    plot_free_isf()
    plot_free_local_isf()
    plot_periodic_isf()
    plot_periodic_local_isf()

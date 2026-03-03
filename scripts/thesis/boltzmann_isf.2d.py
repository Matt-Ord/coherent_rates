import dataclasses
from typing import Any

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.constants import Boltzmann
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_split_value_list_against_time,
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    DoubleGaussianMethod,
    GaussianMethod,
    GaussianParameters,
    LinearRecoilMethod,
    get_free_particle_time,
    get_free_recoil,
)
from coherent_rates.isf import (
    get_analytical_isf,
    get_band_resolved_boltzmann_isf,
    get_boltzmann_isf,
    get_local_boltzmann_isf,
    get_scattered_momentum,
    get_weak_boltzmann_isf,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.state import LocalizationParams, ThermalLocalizationStrategy
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystem,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_SLATE_1,
    format_axis_scientific,
    get_thesis_figure,
)


def plot_periodic_isf_dg_split() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = DoubleGaussianMethod(measure="abs", ty="Fast").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_band_resolved_boltzmann_isf(system, config, times, n_repeats=20)

    fig, ax = get_thesis_figure()

    fig, ax = plot_split_value_list_against_time(isf, measure="abs", ax=ax)

    fig.savefig("scripts/thesis/boltzmann_isf.2d.periodic.dg.split.pdf")


class PreferentialLocalizationStrategy(ThermalLocalizationStrategy):
    """Localization strategy that preferentially localizes in potential wells."""

    def generate_params(self) -> LocalizationParams:
        rng = np.random.default_rng()
        parent_params = super().generate_params()

        x_0_hollow = (
            self.system.lattice_constant / 3,
            self.system.lattice_constant / 3,
        )

        jitter = rng.random(len(x_0_hollow)) * (0.05 * self.system.lattice_constant)
        return dataclasses.replace(
            parent_params,
            x_0=tuple(x + j for x, j in zip(x_0_hollow, jitter, strict=True)),
        )

    def __hash__(self) -> int:
        return hash((-11, self.system, self.config, self.sigma_0))


def plot_periodic_isf_dg() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(10, 10),
        truncation=625,
        temperature=155,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = DoubleGaussianMethod(measure="abs", ty="Fast").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    # Maybe there are more optimal parameters here
    strategy = PreferentialLocalizationStrategy(
        system,
        config,
        sigma_0=(system.lattice_constant / 52, system.lattice_constant / 52),
    )
    local_isf = get_local_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
        strategy=strategy,
    )

    fig, ax = get_thesis_figure()

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)
    fig, ax, line = plot_value_list_against_time(local_isf, measure="abs", ax=ax)
    line.set_label("Simulated Local")
    line.set_color(CAM_BLUE.dark)

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
    inset_line.set_color(CAM_BLUE.warm)

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
    fig.savefig("scripts/thesis/boltzmann_isf.2d.periodic.dg.pdf")


def plot_periodic_isf() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(10, 10),
        truncation=625,
        temperature=155,
    )
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 5),
        truncation=625,
        temperature=155,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = GaussianMethod(measure="abs", t_factor=8.0).get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)

    fig, ax = get_thesis_figure()

    method = GaussianMethod(measure="abs")
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color(CAM_BLUE.dark)
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
    inset_line.set_color(CAM_BLUE.warm)

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
    fig.savefig("scripts/thesis/boltzmann_isf.2d.periodic.pdf")


def plot_periodic_weak_isf() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(10, 10),
        truncation=625,
        temperature=155,
    )
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )
    config = PeriodicSystemConfig(
        (20, 20),
        (45, 45),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    system = dataclasses.replace(system, barrier_energy=system.barrier_energy * 1.5)
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = GaussianMethod(measure="abs", t_factor=32.0).get_fit_times(
        system=system,
        config=config,
    )
    isf = get_weak_boltzmann_isf(system, config, times)

    fig, ax = get_thesis_figure()

    method = GaussianMethod(measure="abs", t_factor=8.0)
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)
    fitted_data = method.get_fitted_data(
        fit,
        isf["basis"],
    )
    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")

    free_fit = GaussianParameters(
        amplitude=fit.amplitude,
        width=get_free_particle_time(system=system, config=config),
    )
    fitted_data = method.get_fitted_data(free_fit, isf["basis"])
    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Free Gaussian Fit")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("-.")

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
    inset_line.set_color(CAM_BLUE.warm)

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

    inset_ax.plot(
        times.fundamental_times,
        get_free_recoil(system, config) * times.fundamental_times,
        color=CAM_BLUE.dark,
        linestyle="--",
        linewidth=2,
    )

    method = LinearRecoilMethod()
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])
    _, _, inset_line = plot_value_list_against_time(
        fitted_data,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_CHERRY.dark)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.2d.periodic_weak.pdf")


def plot_periodic_weak_isf_high_mass() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (45, 45),
        direction=(1, 0),
        truncation=800,
        temperature=155,
    )
    system = dataclasses.replace(system, mass=system.mass * 8)
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201

    times = GaussianMethod(measure="abs", t_factor=8.0).get_fit_times(
        system=system,
        config=config,
    )
    isf = get_weak_boltzmann_isf(system, config, times)

    fig, ax = get_thesis_figure()

    method = GaussianMethod(measure="abs", t_factor=8.0)
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )

    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)
    fitted_data = method.get_fitted_data(
        fit,
        isf["basis"],
    )
    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Gaussian Fit")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")

    free_fit = GaussianParameters(
        amplitude=fit.amplitude,
        width=get_free_particle_time(system=system, config=config),
    )
    fitted_data = method.get_fitted_data(free_fit, isf["basis"])
    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax, measure="abs")
    line.set_label("Free Gaussian Fit")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("-.")

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
    inset_line.set_color(CAM_BLUE.warm)

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
    inset_ax.plot(
        times.fundamental_times,
        get_free_recoil(system, config) * times.fundamental_times,
        color=CAM_BLUE.dark,
        linestyle="--",
        linewidth=2,
    )

    method = LinearRecoilMethod()
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])
    _, _, inset_line = plot_value_list_against_time(
        fitted_data,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_CHERRY.dark)

    format_axis_scientific(inset_ax.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.2d.periodic_weak_hm.pdf")


def _get_occupation_probabilities(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the occupation probability for a given system, configuration, and time."""
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * config.temperature),
    )
    return boltzmann_distribution / np.sum(boltzmann_distribution)


def get_occupation_loss(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> float:
    """Get the occupation loss for a given system and configuration."""
    config_full = config.with_truncation(None)
    probabilities = _get_occupation_probabilities(system, config_full)
    total_p = np.sum(probabilities.reshape(config_full.n_bands, -1)[: config.n_bands])
    return np.real(1 - total_p.item())


def get_max_simulation_time_thermal(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> float:
    """Get the occupation loss for a given system and configuration."""
    velocity = (system.mass / (2 * Boltzmann * config.temperature)) ** 0.5
    return (config.shape[0] * system.lattice_constant) * velocity


def get_band_energy(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    band_index: int,
) -> float:
    """Get the energy of a given band index."""
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    energies = hamiltonian["data"].reshape(config.n_bands, -1)
    return np.real(energies[band_index, 0].item())


def plot_free_isf() -> None:
    system = SODIUM_COPPER_SYSTEM_2D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )

    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=20,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    fig, ax = get_thesis_figure()
    fig, ax, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)
    fig, ax, line = plot_value_list_against_time(
        analytical_isf,
        ax=ax,
        measure="abs",
    )
    line.set_label("Analytical")
    line.set_color(CAM_BLUE.dark)
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
    inset_line.set_color(CAM_BLUE.warm)
    _, _, inset_line = plot_value_list_against_time(
        analytical_isf,
        ax=inset_ax,
        measure="angle",
    )
    inset_line.set_color(CAM_BLUE.dark)
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

    fig.savefig("scripts/thesis/boltzmann_isf.2d.free.pdf")


if __name__ == "__main__":
    plot_free_isf()
    plot_periodic_isf()
    plot_periodic_weak_isf()
    plot_periodic_weak_isf_high_mass()
    plot_periodic_isf_dg()
    plot_periodic_isf_dg_split()

import dataclasses

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import hbar
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_transformed_energy_1d_against_self_energy,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import get_scattered_momentum
from coherent_rates.isf import get_boltzmann_isf, get_local_boltzmann_isf
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.state import LocalizationParams, ThermalLocalizationStrategy
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_SLATE_1,
    format_axis_scientific,
    setup_rc_params_paper,
)


def get_toc_figure() -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params_paper()
    plt.rcParams.update(
        {"font.size": 7},
    )

    fig, (_ax0, _ax1) = plt.subplots(
        ncols=2,
        figsize=(3.14, 1.57),
        layout="constrained",
    )
    _ax0.set_facecolor(CAM_SLATE_1)
    _ax1.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    _ax0.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=7,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )
    _ax1.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=7,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    _ax0.xaxis.label.set_fontsize(7)
    _ax0.yaxis.label.set_fontsize(7)
    _ax1.xaxis.label.set_fontsize(7)
    _ax1.yaxis.label.set_fontsize(7)
    return fig, (_ax0, _ax1)


class PreferentailLocalizationStrategy(ThermalLocalizationStrategy):
    """Localization strategy that preferentially localizes in potential wells."""

    def generate_params(self) -> LocalizationParams:
        rng = np.random.default_rng()
        parent_params = super().generate_params()

        x_0_hollow = (0.0,) * len(parent_params.x_0)
        jitter = rng.random(len(x_0_hollow)) * (0.05 * self.system.lattice_constant)
        return dataclasses.replace(
            parent_params,
            x_0=tuple(x + j for x, j in zip(x_0_hollow, jitter, strict=True)),
        )

    def __hash__(self) -> int:
        return hash((-11, self.system, self.config, self.sigma_0))


def plot_periodic_local_isf_paper(ax: Axes) -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(300,),
        truncation=25,
        temperature=155,
    )

    times = EvenlySpacedTimeBasis(100, 1, 0, 2e-13)
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201

    strategy = PreferentailLocalizationStrategy(
        system=system,
        config=config,
        sigma_0=(system.lattice_constant / 30,),
    )
    isf = get_local_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=1000,
        strategy=strategy,
    )

    measure = "abs"

    fig, ax, line = plot_value_list_against_time(isf, measure=measure, ax=ax)
    line.set_label("Localized")
    line.set_color(CAM_BLUE.warm)
    line.set_linewidth(1)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"Scattering Function")

    isf_nonlocal = get_boltzmann_isf(system, config, times, n_repeats=100)
    fig, ax, line = plot_value_list_against_time(isf_nonlocal, measure=measure, ax=ax)
    line.set_label("Not Localized")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")
    line.set_linewidth(1)

    ax.set_xlim(0, times.times[-1])

    format_axis_scientific(ax.yaxis)

    legend = ax.legend(frameon=False, loc="lower left", fontsize=7)
    legend.get_frame().set_alpha(0)

    fig.set_facecolor((0, 0, 0, 0))


def plot_tunneling_rate_against_self_energy(ax: Axes) -> None:
    config = PeriodicSystemConfig((60,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)

    _fig, ax, (line, _) = plot_wavepacket_transformed_energy_1d_against_self_energy(
        wavefunctions,
        free_mass=None,
        measure="abs",
        ax=ax,
        scale_factor=system.lattice_constant / hbar,
        energy_scale_factor=10**20,
    )

    line.set_color(CAM_BLUE.warm)
    line.set_linestyle("-")
    line.set_linewidth(1)
    line.set_markersize(4)

    wavefunctions = get_bloch_wavefunctions(system.with_barrier_energy(0), config)
    _fig, ax, (free_line, _) = (
        plot_wavepacket_transformed_energy_1d_against_self_energy(
            wavefunctions,
            free_mass=None,
            measure="abs",
            ax=ax,
            scale_factor=system.lattice_constant / hbar,
            energy_scale_factor=10**20,
        )
    )
    free_line.set_color(CAM_BLUE.dark)
    free_line.set_linestyle("-")
    free_line.set_marker("")
    free_line.set_linewidth(1)

    barrier_energy = system.barrier_energy

    print(f"Barrier energy: {barrier_energy:0.2e} J")  # noqa: T201
    ax.set_ylim(None, 1e3)
    ax.set_xlim(0, 5 * barrier_energy * 10**20)
    ax.set_ylabel(r"Ballistic Rate / $\mathrm{s}^{-1}$")
    ax.set_xlabel("Average Energy / $\\times 10^{-20} \\mathrm{J}$")
    barrier_line = ax.axvline(barrier_energy * 10**20)
    barrier_line.set_linestyle("--")
    barrier_line.set_color(CAM_CHERRY.dark)
    barrier_line.set_linewidth(1)

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=7,
        handles=[free_line, barrier_line],
        labels=["Free Particle", "Barrier Energy"],
    )
    legend.get_frame().set_alpha(0)


def plot_table_of_contents() -> None:

    fig, (ax0, _ax1) = get_toc_figure()
    plot_tunneling_rate_against_self_energy(ax0)
    plot_periodic_local_isf_paper(_ax1)

    fig.savefig("scripts/thesis/table_of_contents.pdf")
    fig.savefig("scripts/thesis/table_of_contents.png", dpi=600)
    fig.savefig("scripts/thesis/table_of_contents.tif", dpi=600)


if __name__ == "__main__":
    plot_table_of_contents()

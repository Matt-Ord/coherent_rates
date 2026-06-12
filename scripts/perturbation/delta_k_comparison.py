import dataclasses

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import (
    PeriodicSystemConfig,
)
from coherent_rates.fit import get_scattered_direction
from coherent_rates.isf import (
    get_weak_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
)
from coherent_rates.util import (
    format_axis_scientific,
    get_thesis_figure,
)


def plot_delta_k_comparison_1st() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(10,),
        truncation=50,
        temperature=155,
    )

    times = EvenlySpacedTimeBasis(1000, 1, 0, delta_t=2e-12)

    fig, ax = get_thesis_figure()

    norm = Normalize(vmin=0, vmax=2)
    c_map = mpl.colormaps["viridis"]

    for delta_k in np.linspace(0, 2, 10, endpoint=True)[1::]:
        direction = get_scattered_direction(system, config, [10**10 * delta_k])[0]

        config = dataclasses.replace(config, direction=direction)
        isf = get_weak_boltzmann_isf(system, config, times)

        _, _, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
        line.set_color(c_map(norm(delta_k)))

    ax.set_ylim(0.6, 1)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    sm = cm.ScalarMappable(cmap=c_map, norm=norm)
    sm.set_array([])
    c_bar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    c_bar.set_label(r"$\Delta k$ / $\AA^{-1}$", fontsize=9)
    c_bar.ax.tick_params(labelsize=8)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/delta_k_comparison.1d.1st.pdf")


def plot_delta_k_comparison_2nd() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (20,),
        (100,),
        direction=(10,),
        truncation=50,
        temperature=155,
    )

    times = EvenlySpacedTimeBasis(1000, 1, 0, delta_t=2e-12)

    fig, ax = get_thesis_figure()

    norm = Normalize(vmin=0, vmax=2)
    c_map = mpl.colormaps["viridis"]

    for delta_k in np.linspace(0, 2, 10, endpoint=True)[1::]:
        direction = get_scattered_direction(system, config, [10**10 * delta_k])[0]

        config = dataclasses.replace(config, direction=direction)
        isf = get_weak_boltzmann_isf(system, config, times)
        isf_2o = get_weak_boltzmann_isf(system, config, times, second_order=True)

        _, _, line = plot_value_list_against_time(
            {
                "data": (isf_2o["data"] + (1 - isf["data"])),
                "basis": isf["basis"],
            },
            measure="abs",
            ax=ax,
        )
        line.set_color(c_map(norm(delta_k)))

    ax.set_ylim(0.6, 1)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    sm = cm.ScalarMappable(cmap=c_map, norm=norm)
    sm.set_array([])
    c_bar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    c_bar.set_label(r"$\Delta k$ / $\AA^{-1}$", fontsize=9)
    c_bar.ax.tick_params(labelsize=8)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/delta_k_comparison.1d.2nd.pdf")


def plot_delta_k_comparison_1st_2d() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )

    times = EvenlySpacedTimeBasis(1000, 1, 0, delta_t=2e-12)

    fig, ax = get_thesis_figure()

    norm = Normalize(vmin=0, vmax=2)
    c_map = mpl.colormaps["viridis"]

    for delta_k in np.linspace(0, 2, 10, endpoint=True)[1::]:
        direction = get_scattered_direction(system, config, [10**10 * delta_k])[0]

        config = dataclasses.replace(config, direction=direction)
        isf = get_weak_boltzmann_isf(system, config, times)

        _, _, line = plot_value_list_against_time(isf, measure="abs", ax=ax)
        line.set_color(c_map(norm(delta_k)))

    ax.set_ylim(0.6, 1)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    sm = cm.ScalarMappable(cmap=c_map, norm=norm)
    sm.set_array([])
    c_bar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    c_bar.set_label(r"$\Delta k$ / $\AA^{-1}$", fontsize=9)
    c_bar.ax.tick_params(labelsize=8)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/delta_k_comparison.2d.1st.pdf")


def plot_delta_k_comparison_2nd_2d() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )

    times = EvenlySpacedTimeBasis(1000, 1, 0, delta_t=2e-12)

    fig, ax = get_thesis_figure()

    norm = Normalize(vmin=0, vmax=2)
    c_map = mpl.colormaps["viridis"]

    for delta_k in np.linspace(0, 2, 10, endpoint=True)[1::]:
        direction = get_scattered_direction(system, config, [10**10 * delta_k])[0]

        config = dataclasses.replace(config, direction=direction)
        isf = get_weak_boltzmann_isf(system, config, times)
        isf_2o = get_weak_boltzmann_isf(system, config, times, second_order=True)

        _, _, line = plot_value_list_against_time(
            {
                "data": (isf_2o["data"] + (1 - isf["data"])),
                "basis": isf["basis"],
            },
            measure="abs",
            ax=ax,
        )
        line.set_color(c_map(norm(delta_k)))

    ax.set_ylim(0.6, 1)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    sm = cm.ScalarMappable(cmap=c_map, norm=norm)
    sm.set_array([])
    c_bar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    c_bar.set_label(r"$\Delta k$ / $\AA^{-1}$", fontsize=9)
    c_bar.ax.tick_params(labelsize=8)

    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/perturbation/delta_k_comparison.2d.2nd.pdf")


if __name__ == "__main__":
    plot_delta_k_comparison_1st()
    plot_delta_k_comparison_2nd()
    # plot_delta_k_comparison_1st_2d()
    # plot_delta_k_comparison_2nd_2d()

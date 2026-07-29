import dataclasses
from pathlib import Path
from typing import Any

import matplotlib.cm
import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import (
    Boltzmann,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
)
from surface_potential_analysis.state_vector.eigenstate_list import StatisticalValueList
from surface_potential_analysis.util.decorators import cached

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_band_resolved_boltzmann_isf,
    get_scattered_momentum,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    System,
)
from coherent_rates.util import (
    CAM_SLATE_1,
    format_axis_scientific,
    get_thesis_fig_size,
    setup_rc_params_thesis,
)


def _get_split_band_isf_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    *,
    n_repeats: int = 10,
) -> Path:
    return Path(
        f"data/{hash((system, config))}.{hash(times)}.{n_repeats}.boltzmann.isf.split",
    )


@cached(_get_split_band_isf_path)
def get_split_band_isf[BT0: BasisWithTimeLike[Any, Any]](
    system: System,
    config: PeriodicSystemConfig,
    times: BT0,
    *,
    n_repeats: int = 10,
) -> StatisticalValueList[TupleBasisLike[BasisLike[Any, Any], BT0]]:
    return get_band_resolved_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )


def get_average_band_energy(
    system: System,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    basis = hamiltonian["basis"][0].wavefunctions["basis"]
    n_bands = basis[0][0].n
    return np.real(np.average(hamiltonian["data"].reshape(n_bands, -1), axis=-1))


def get_double_thesis_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params_thesis()
    w, h = get_thesis_fig_size()
    fig, (ax1, ax2) = plt.subplots(
        figsize=fig_size or (2 * w, h),
        ncols=2,
        layout="constrained",
    )
    ax1.set_facecolor(CAM_SLATE_1)
    ax2.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax1.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=9,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )
    ax2.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=9,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax1.xaxis.label.set_fontsize(11)
    ax1.yaxis.label.set_fontsize(11)
    ax2.xaxis.label.set_fontsize(11)
    ax2.yaxis.label.set_fontsize(11)
    return fig, (ax1, ax2)


def _prepare_monotonic_stacked_data(
    isf_data: np.ndarray,
) -> np.ndarray:
    """Sorts complex ISF components by energy hierarchy, takes the absolute.

    value of the cumulative sum, enforces monotonicity, and returns individual
    magnitudes for stackplot.
    """
    # 2. Add complex components first, then take the absolute value
    abs_cumsum = np.abs(np.cumsum(isf_data, axis=0))

    # 3. Enforce monotonicity backwards along the band axis (axis 0)
    # If abs_cumsum[i, t] > abs_cumsum[i+1, t], truncate it to abs_cumsum[i+1, t]
    for i in range(abs_cumsum.shape[0] - 2, -1, -1):
        abs_cumsum[i, :] = np.minimum(abs_cumsum[i, :], abs_cumsum[i + 1, :])

    # 4. Convert the cumulative monotonic stack back to individual
    # block thicknesses for stackplot
    stacked = np.zeros_like(abs_cumsum)
    stacked[0, :] = abs_cumsum[0, :]
    stacked[1:, :] = np.diff(abs_cumsum, axis=0)
    return stacked


def plot_split_band_isf(*, barrier_energy: float = 1) -> None:  # noqa: PLR0915
    system_2d = SODIUM_COPPER_SYSTEM_2D
    system_2d = dataclasses.replace(
        system_2d,
        barrier_energy=system_2d.barrier_energy * barrier_energy,
    )
    config_2d = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system_2d,
        config=config_2d,
    )
    delta_k = get_scattered_momentum(system_2d, config_2d, [config_2d.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201

    isf_2d = get_split_band_isf(system_2d, config_2d, times, n_repeats=20)
    band_energies_2d = np.real(get_average_band_energy(system_2d, config_2d))
    band_energies_2d -= SODIUM_COPPER_SYSTEM_2D.barrier_energy / 9
    band_energies_2d /= Boltzmann * config_2d.temperature

    sort_indices = np.argsort(band_energies_2d)
    band_energies_2d = band_energies_2d[sort_indices]
    stacked_2d = _prepare_monotonic_stacked_data(
        isf_2d["data"].reshape(isf_2d["basis"].shape)[sort_indices, :],
    )
    times_axis = isf_2d["basis"][1].times

    # 2. Map the sorted energy values to a smooth colormap gradient
    cmap = matplotlib.colormaps["RdBu"]
    norm = matplotlib.colors.TwoSlopeNorm(
        vcenter=0.0,
        vmin=band_energies_2d.min() - 0.1,
        vmax=band_energies_2d.min() + 4.0,
    )
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array needed for ScalarMappable

    fig, (ax1, ax2) = get_double_thesis_figure()

    colors_2d = [cmap(norm(energy)) for energy in band_energies_2d]
    polygons = ax2.stackplot(
        times_axis,
        stacked_2d,
        colors=colors_2d,
        rasterized=True,
        edgecolors="none",
    )
    for poly, color in zip(polygons, colors_2d, strict=False):
        poly.set_edgecolor(color)
        poly.set_linewidth(0.5)
    # 4. Add a colorbar to map the colors back to physical energy values
    system_1d = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system_1d = dataclasses.replace(
        system_1d,
        barrier_energy=system_1d.barrier_energy * barrier_energy,
    )

    config_1d = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(66,),
        truncation=50,
        temperature=155,
    )
    isf_1d = get_split_band_isf(system_1d, config_1d, times, n_repeats=20)
    band_energies_1d = np.real(get_average_band_energy(system_1d, config_1d))
    band_energies_1d -= SODIUM_COPPER_BRIDGE_SYSTEM_1D.barrier_energy
    band_energies_1d /= Boltzmann * config_1d.temperature

    sort_indices = np.argsort(band_energies_1d)
    band_energies_1d = band_energies_1d[sort_indices]
    stacked_1d = _prepare_monotonic_stacked_data(
        isf_1d["data"].reshape(isf_1d["basis"].shape)[sort_indices, :],
    )
    times_axis = isf_1d["basis"][1].times

    colors_1d = [cmap(norm(energy)) for energy in band_energies_1d]

    polygons = ax1.stackplot(
        times_axis,
        stacked_1d,
        colors=colors_1d,
        rasterized=True,
        edgecolors="none",
    )
    for poly, color in zip(polygons, colors_1d, strict=False):
        poly.set_edgecolor(color)
        poly.set_linewidth(0.5)

    ax1.text(0.85, 0.9, "1D", transform=ax1.transAxes)
    ax1.set_xlim(times_axis[0], times_axis[-1])
    ax1.set_xlabel("Time / s")
    ax1.set_ylabel(r"$|I(\Delta k, t)|$")
    ax2.text(0.85, 0.9, "2D", transform=ax2.transAxes)
    ax2.set_xlim(times_axis[0], times_axis[-1])
    ax2.set_xlabel("Time / s")
    if barrier_energy == 1:
        ax2.set_ylim(0.7, 1.0)
        ax1.set_ylim(0.7, 1.0)
    else:
        ax2.set_ylim(0, 1.0)
        ax1.set_ylim(0, 1.0)

    format_axis_scientific(ax1.yaxis)

    fig.set_facecolor((0, 0, 0, 0))
    cbar = fig.colorbar(sm, ax=ax2)
    cbar.set_label(
        r"Band Energy $(E - E_b) / k_b T$",
    )
    fig.savefig(
        f"scripts/perturbation/split_band.{barrier_energy}.pdf",
        bbox_inches="tight",
        dpi=1200,
    )


if __name__ == "__main__":
    plot_split_band_isf()
    plot_split_band_isf(barrier_energy=0)

from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.figure import Figure
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.state_vector.eigenstate_list import (
    StatisticalValueList,
    ValueList,
)
from surface_potential_analysis.util.decorators import cached
from surface_potential_analysis.util.plot import get_figure
from surface_potential_analysis.util.util import Measure, get_measured_data

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import get_default_isf_times
from coherent_rates.isf import (
    get_band_resolved_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import CAM_BLUE, CAM_SLATE_1


def get_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])
_B0 = TypeVar("_B0", bound=BasisLike[int, int])


def plot_data_shadow(
    data: np.ndarray[tuple[int], np.dtype[np.complex128]],
    coordinates: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, FillBetweenPolyCollection]:
    """Plot data in 1d.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.complex128]]
    coordinates : np.ndarray[tuple[int], np.dtype[np.float64]]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]

    """
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(data, measure)

    fill = ax.fill_between(
        coordinates,
        0,
        measured_data,
        alpha=0.25,
        linewidth=0,
    )

    return fig, ax, fill


def plot_split_value_list_against_time(
    values: ValueList[TupleBasisLike[_B0, _BT0]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes]:
    """Plot the data against time, split by _B0.

    Parameters
    ----------
    values : ValueList[_AX0Inv]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]

    """
    fig, ax = get_figure(ax)

    stacked = values["data"].reshape(values["basis"].shape)
    cumulative = np.cumsum(stacked, axis=0)
    for i, band_data in enumerate(cumulative[::-1]):
        fig, ax, fill = plot_data_shadow(
            band_data,
            values["basis"][1].times,
            measure=measure,
            ax=ax,
        )
        fill.set_color(CAM_BLUE.warm)
        frac = i / cumulative.shape[0]
        fill.set_alpha(0.2 + 0.8 * (1 - frac))

    ax.set_xlabel("Times /s")
    ax.set_xlim(0, values["basis"][1].times[-1])
    return fig, ax


@cached(Path("data/band.resolved.isf"))
def _get_resolved_data() -> StatisticalValueList[
    TupleBasisLike[BasisLike[Any, Any], EvenlySpacedTimeBasis[Any, Any, Any]]
]:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(0)

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=100,
    )

    times = get_default_isf_times(system=system, config=config)

    return get_band_resolved_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )


def plot_band_resolved_demonstration() -> None:
    resolved_data = _get_resolved_data()
    fig, ax = plt.subplots(
        figsize=get_fig_size(),
        layout="constrained",
    )

    fig, ax = plot_split_value_list_against_time(resolved_data, measure="real", ax=ax)
    ax.set_ylim(0, 1)

    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/boltzmann_isf.resolved.pdf")


if __name__ == "__main__":
    plot_band_resolved_demonstration()

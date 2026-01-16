from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.plot import (
    plot_boltzmann_rate_against_momentum,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    FreeSystem,
    System,
)
from scripts.thesis.bandstructure_plot import CAM_DARK_BLUE
from scripts.thesis.util import (
    CAM_WARM_BLUE,
    format_axis_scientific,
    get_fancy_figure,
    setup_rc_params,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from coherent_rates.fit import FitMethod


setup_rc_params()


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
    line.set_color(CAM_DARK_BLUE)

    _, _, line = plot_boltzmann_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=free_fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label("Free system")
    line.set_color(CAM_WARM_BLUE)

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
        (20, 20),
        (35, 35),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D
    print(system.barrier_energy)  # noqa: T201
    directions = [(i, i) for i in [1, 2, *list(range(5, 100, 5))]]

    fig = _compare_rate_against_free_surface(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(measure="abs"),
        free_fit_method=GaussianMethod(measure="abs"),
    )
    fig.savefig("scripts/thesis/boltzmann_mass.2d.pdf")

from __future__ import annotations

from typing import TYPE_CHECKING

from surface_potential_analysis.basis.momentum_basis_like import MomentumBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_momentum,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    DoubleGaussianMethod,
    GaussianMethod,
)
from coherent_rates.isf import get_boltzmann_rate_against_momentum_data
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    FreeSystem,
)
from scripts.thesis.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_PURPLE,
    format_axis_scientific,
    get_fancy_figure,
    setup_rc_params,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

setup_rc_params()


def select_idx(
    rates: ValueList[MomentumBasis],
    idx: list[int],
) -> ValueList[MomentumBasis]:
    selected_momenta = rates["data"][idx]
    return {
        "basis": MomentumBasis(rates["basis"].k_points[idx]),
        "data": selected_momenta,
    }


if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 2),
        truncation=625,
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_fancy_figure()

    directions = [(i, i) for i in range(1, 7)]
    data_111_double = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=DoubleGaussianMethod(measure="abs", ty="Fast"),
        directions=directions,
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
    fig.savefig("scripts/thesis/boltzmann_mass.2d.pdf")

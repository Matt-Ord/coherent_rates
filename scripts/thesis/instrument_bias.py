from __future__ import annotations

import dataclasses

from scipy.constants import electron_volt
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import ExponentialInstrumentFunction, PeriodicSystemConfig
from coherent_rates.isf import (
    get_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from scripts.thesis.util import (
    CAM_DARK_BLUE,
    CAM_WARM_BLUE,
    get_fancy_figure,
    setup_rc_params,
)

setup_rc_params()

if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (400,),
        (100,),
        truncation=50,
        direction=(50,),
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    times = EvenlySpacedTimeBasis(100, 1, 0, 4.0e-12)

    n_repeats = 100

    data = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )

    fig, ax = get_fancy_figure()
    fig, ax, line = plot_value_list_against_time(data, ax=ax)
    line.set_label("Ideal")
    line.set_color(CAM_DARK_BLUE)

    config = dataclasses.replace(
        config,
        instrument_function=ExponentialInstrumentFunction(
            width=8.03 * 10**-3 * electron_volt,
            optimal_energy_out=7.7 * 10**-3 * electron_volt,
            incoming_energy=8 * 10**-3 * electron_volt,
        ),
    )
    data = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_time(data, ax=ax)
    line.set_label("Corrected")
    line.set_color(CAM_WARM_BLUE)
    legend = ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    ax.set_xlabel(r"Time / $\mathrm{s}$")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")
    fig.savefig("scripts/thesis/instrument_bias.isf.pdf")

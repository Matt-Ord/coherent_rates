from __future__ import annotations

import dataclasses

from scipy.constants import electron_volt
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import ExponentialInstrumentFunction, PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod, get_scattered_momentum
from coherent_rates.isf import (
    get_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
)
from coherent_rates.util import (
    CAM_BLUE,
    get_thesis_figure,
)

# We could also look at
# ! config = PeriodicSystemConfig(
# !     (390,),
# !     (100,),
# !     direction=(26,),
# !     truncation=50,
# !     temperature=155,
# ! )
# ! and for 2D
# ! config = PeriodicSystemConfig(
# !     (20, 20),
# !     (35, 35),
# !     direction=(2, 0),
# !     truncation=625,
# !     temperature=155,
# ! )


def _plot_instrument_bias() -> None:
    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(66,),
        truncation=50,
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )

    n_repeats = 100

    data = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )

    fig, ax = get_thesis_figure()
    fig, ax, line = plot_value_list_against_time(data, ax=ax)
    line.set_label("Ideal")
    line.set_color(CAM_BLUE.warm)

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
    line.set_color(CAM_BLUE.dark)
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    ax.set_xlabel(r"Time / $\mathrm{s}$")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")
    ax.set_ylim(0.7, 1.0)
    fig.savefig("scripts/thesis/instrument_bias.isf.1d.pdf")


def _plot_instrument_bias_2d() -> None:
    system = SODIUM_COPPER_SYSTEM_2D
    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )

    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k: {delta_k:0.3e}")  # noqa: T201
    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )

    data = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=20,
    )

    fig, ax = get_thesis_figure()
    fig, ax, line = plot_value_list_against_time(data, ax=ax)
    line.set_label("Ideal")
    line.set_color(CAM_BLUE.warm)

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
        n_repeats=20,
    )
    fig, ax, line = plot_value_list_against_time(data, ax=ax)
    line.set_label("Corrected")
    line.set_color(CAM_BLUE.dark)
    legend = ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    ax.set_xlabel(r"Time / $\mathrm{s}$")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")
    ax.set_ylim(0.7, 1.0)
    fig.savefig("scripts/thesis/instrument_bias.isf.2d.pdf")


if __name__ == "__main__":
    _plot_instrument_bias()
    _plot_instrument_bias_2d()

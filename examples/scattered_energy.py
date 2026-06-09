from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_momentum,
)
from surface_potential_analysis.state_vector.state_vector import StateVector
from surface_potential_analysis.util.squared_scale import SquaredScale

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.isf import (
    get_conditions_at_temperatures,
    get_scattered_energy_change_against_k,
    get_thermal_scattered_energy_change_against_k,
)
from coherent_rates.plot import (
    plot_occupation_against_energy_change_with_contition,
)
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    FreeSystem,
    System,
)


def plot_scattered_energy_change_state(
    system: System,
    config: PeriodicSystemConfig,
    state: StateVector[Any],
    *,
    directions: list[tuple[int, ...]] | None = None,
) -> tuple[Figure, Axes]:
    bound_data = get_scattered_energy_change_against_k(
        system,
        config,
        state,
        directions=directions,
    )
    fig, ax, _ = plot_value_list_against_momentum(bound_data)
    ax.set_xscale(SquaredScale(axis=None))  # type: ignore library type
    ax.set_title("Quadratic")  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type
    fig.show()

    fig, ax, _ = plot_value_list_against_momentum(bound_data)
    ax.set_title("Linear")  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type
    return (fig, ax)


def plot_occupation_against_energy_change_comparison_mass(
    system: System,
    config: PeriodicSystemConfig,
    mass_ratio: float,
) -> tuple[Figure, Axes]:
    conditions = [
        (system, config, "Normal Mass"),
        (
            system.with_mass(mass_ratio * system.mass),
            config,
            f"{mass_ratio}$\\times$ mass",
        ),
    ]

    fig, ax = plot_occupation_against_energy_change_with_contition(
        conditions,
    )

    ax.axvline(system.barrier_energy, color="black", ls="--")  # type: ignore library type

    ax.set_xlim(0, 10 * system.barrier_energy)
    ax.set_ylim(0)
    ax.legend()  # type: ignore library type
    return (fig, ax)


def plot_occupation_against_energy_change_comparison_temperature(
    system: System,
    config: PeriodicSystemConfig,
    temperatures: tuple[float, float],
) -> tuple[Figure, Axes]:
    conditions = get_conditions_at_temperatures(system, config, temperatures)
    fig, ax = plot_occupation_against_energy_change_with_contition(
        conditions,
    )

    line = ax.axvline(system.barrier_energy, color="black", ls="--")  # type: ignore library type
    line.set_label("Barrier Energy")
    ax.set_xlim(0, 10 * system.barrier_energy)
    ax.set_ylim(0)
    ax.legend()  # type: ignore library type
    return (fig, ax)


def plot_thermal_scattered_energy_change_comparison(
    system: System,
    config: PeriodicSystemConfig,
    *,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> tuple[Figure, Axes]:
    bound_data = get_thermal_scattered_energy_change_against_k(
        system,
        config,
        directions=directions,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_momentum(bound_data)
    line.set_label("Bound")

    free_system = FreeSystem(system)
    free_data = get_thermal_scattered_energy_change_against_k(
        free_system,
        config,
        directions=directions,
        n_repeats=1,
    )
    fig, ax, line1 = plot_value_list_against_momentum(free_data, ax=ax)
    line1.set_label("Free")

    ax.legend()  # type: ignore library type
    ax.set_xscale(SquaredScale(axis=None))  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type

    return (fig, ax)


if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        direction=(50,),
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    fig, ax = plot_occupation_against_energy_change_comparison_temperature(
        system,
        config,
        (100, 155),
    )
    fig.savefig("energy_change.against_t.pdf")

    config = PeriodicSystemConfig((20,), (50,), temperature=155)

    # Shows that scattered energy is lower for a larger initial mass
    fig, ax = plot_occupation_against_energy_change_comparison_mass(system, config, 3)
    fig.savefig("energy_change.against_mass.pdf")

    system = HYDROGEN_NICKEL_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)
    n = 0
    b = 1
    directions = [(n + b * i,) for i in range(10)]

    # # For low state k, the dE vs dk plot is quadratic
    # state = get_state_vector(wavefunctions, 0)
    # fig, ax = plot_scattered_energy_change_state(
    #     system,
    #     config,
    #     state,
    #     directions=directions,
    # )
    # fig.savefig("scattered_energy_change.small_k.pdf")

    # # For high state k, the dE vs dk plot is linear
    # state = get_state_vector(wavefunctions, 230)
    # fig, ax = plot_scattered_energy_change_state(
    #     system,
    #     config,
    #     state,
    #     directions=directions,
    # )
    # fig.savefig("scattered_energy_change.large_k.pdf")

    # Since dE is proportional to (k+dk)^2 - k^2 = 2k*dk +(dk)^2,
    # for low bands, k is small so dE~(dk)^2
    # for high bands, k is large so dE~dk

    # For a thermal state, we have <k> = 0 when averaging across experiments,
    # so the cross term averages out and we get dE proportional to (dk)^2
    fig, ax = plot_thermal_scattered_energy_change_comparison(
        system,
        config,
        directions=directions,
    )
    fig.savefig("energy_change.against_t.1.pdf")

import numpy as np
from matplotlib.scale import LogScale
from scipy.constants import Boltzmann
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod, get_free_particle_isf
from coherent_rates.isf import get_momentum_squared_per_state, get_weak_boltzmann_isf
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, System
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    format_axis_scientific,
    get_thesis_figure,
)


def _plot_momentum_squared() -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    fig, ax = get_thesis_figure()

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(10,),
        truncation=50,
        temperature=155,
        # offset=(0.01,),  # Breaks some of the symmetry  # noqa: ERA001
    )

    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    n_bands = hamiltonian["basis"][0].wavefunctions["basis"][0].shape[0]
    momentum = get_momentum_squared_per_state(
        hamiltonian,
        config.direction,
    )["data"].reshape(n_bands, -1)

    energy_per_state = hamiltonian["data"]

    energies_2d = energy_per_state.reshape((n_bands, -1))
    energies_2d = np.real_if_close(energies_2d)
    scaled_momentum = (momentum / (2 * system.mass * energies_2d)).reshape(
        (n_bands, -1),
    )
    for b in range(n_bands):
        sort_idx = np.argsort(energies_2d[b, :])
        (line,) = ax.plot(
            (energies_2d[b, sort_idx] - system.barrier_energy)
            / (Boltzmann * config.temperature),
            np.real_if_close(scaled_momentum[b, sort_idx]),
            label=f"Band {b}",
        )
        if b % 2 == 0:
            line.set_color(CAM_BLUE.dark)
        else:
            line.set_color(CAM_BLUE.warm)

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"$\langle\hat{p}\rangle^2$ / $2mE$")
    ax.set_ylim(1e-5, 1)
    ax.set_yscale(LogScale(None, base=10))
    ax.set_xlim(-1, 2)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        handles=[line],
        labels=["$E_b$"],
    )

    fig.savefig("scripts/perturbation/expect_p.pdf", bbox_inches="tight")


def _get_momentum_threshold_effective_mass(
    system: System,
    config: PeriodicSystemConfig,
    *,
    threshold: float = 0.01,
) -> tuple[float, float]:
    hamiltonian = get_hamiltonian(system, config)
    momentum = get_momentum_squared_per_state(
        hamiltonian,
        config.direction,
    )["data"]
    energy_per_state = hamiltonian["data"]
    scaled_momentum = momentum / (2 * system.mass * energy_per_state)
    sorted_idx = np.argsort(scaled_momentum)[::-1]

    # From the largest to smallest (low mass to high mass)
    scaled_momentum = scaled_momentum[sorted_idx]
    momentum = np.real_if_close(momentum[sorted_idx])
    energy_per_state = np.real_if_close(energy_per_state[sorted_idx])

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    cut_idx = np.argmax(scaled_momentum < threshold)
    thermal_factors = thermal_factors[:cut_idx]
    momentum = momentum[:cut_idx]

    total_occupation = np.sum(thermal_factors)
    prefactor = 1 / (config.temperature * Boltzmann * system.mass**2)
    inverse_mass = np.sum(thermal_factors * momentum * prefactor)
    inverse_mass /= total_occupation

    return total_occupation, 1 / inverse_mass


def _plot_best_fit_mass(
    *,
    temperature: float = 155,
    mass_factor: float = 1,
    energy_factor: float = 1,
) -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_barrier_energy(energy_factor * system.barrier_energy)
    system = system.with_mass(mass_factor * system.mass)

    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=temperature,
    )

    times = GaussianMethod(measure="abs").get_fit_times(
        system=system,
        config=config,
    )

    isf = get_weak_boltzmann_isf(system, config, times)
    fig, ax = get_thesis_figure()

    fig, ax, line_first_order = plot_value_list_against_time(isf, measure="abs", ax=ax)
    line_first_order.set_label("First Order")
    line_first_order.set_color(CAM_CHERRY.base)

    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"$|I(\Delta k, t)|$")

    format_axis_scientific(ax.yaxis)

    get_hamiltonian.load_or_call_cached(system, config)

    total_occupation, effective_mass = _get_momentum_threshold_effective_mass(
        system,
        config,
    )

    ax.axhline(1 - total_occupation, color=CAM_CHERRY.dark, linestyle="--")

    (line_effective_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system.with_mass(effective_mass),
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_CHERRY.dark,
        linestyle="--",
        label="Effective Mass",
    )

    (line_real_mass,) = ax.plot(
        times.times,
        get_free_particle_isf(
            system,
            config,
            times.times,
            offset=1 - total_occupation,
        ),
        color=CAM_BLUE.warm,
        linestyle="--",
        label="Actual Mass",
    )

    ax.legend(
        loc="upper right",
        handles=[line_real_mass, line_effective_mass, line_first_order],
        fontsize=9,
    )

    ax.set_ylim(((1 - 1.1 * total_occupation), 1))

    fig.savefig("scripts/perturbation/expect_p.best_fit_mass.pdf")


if __name__ == "__main__":
    _plot_momentum_squared()
    _plot_best_fit_mass()

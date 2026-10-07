from typing import Any

import numpy as np
import scipy
import scipy.special
from scipy.constants import Boltzmann, atomic_mass, hbar

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.isf import (
    get_momentum_squared_per_state,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    PeriodicSystem1d,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
)


def _get_classical_momentum_squared_integral(
    barrier_energy: float,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.zeros_like(energies, dtype=np.float64)
    mask = energies > barrier_energy

    if np.any(mask):
        eps = energies[mask] / barrier_energy
        m = 1.0 / eps
        k_val = scipy.special.ellipk(m)
        result[mask] = eps / (4.0 * k_val**2)

    return result


def _get_classical_momentum_squared(
    system: PeriodicSystem1d,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    prefactor = 2 * system.mass * system.barrier_energy * np.pi**2
    integral = _get_classical_momentum_squared_integral(system.barrier_energy, energies)
    return prefactor * integral


def _get_classical_crossing_time_integral(
    barrier_energy: float,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.full(energies.shape, np.inf, dtype=np.float64)

    # Above barrier: un-trapped crossing time
    mask_above = energies > barrier_energy
    if np.any(mask_above):
        eps_above = energies[mask_above] / barrier_energy
        m_above = 1.0 / eps_above
        result[mask_above] = scipy.special.ellipk(m_above) / np.sqrt(eps_above)

    # Below barrier: bound oscillation / attempt time (between classical turning points)
    mask_below = (energies >= 0.0) & (energies < barrier_energy)
    if np.any(mask_below):
        eps_below = energies[mask_below] / barrier_energy
        result[mask_below] = scipy.special.ellipk(eps_below)

    return result


def _get_classical_crossing_time(
    system: PeriodicSystem1d,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:

    # Assumes unit cell length L = 1. If system has a length attribute (e.g. system.length),
    # multiply the prefactor by system.length.
    prefactor = (system.lattice_constant / np.pi) * np.sqrt(
        2.0 * system.mass / system.barrier_energy,
    )
    integral = _get_classical_crossing_time_integral(system.barrier_energy, energies)
    return prefactor * integral


def _get_kemble_formula_probability(
    barrier_energy: float,
    barrier_omega: float,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate the Kemble transmission probability T(E) across all energies."""
    arg = (2.0 * np.pi * (energies - barrier_energy)) / (hbar * barrier_omega)
    return scipy.special.expit(arg)


def _get_barrier_omega(
    system: PeriodicSystem1d,
) -> float:
    return np.sqrt(2 * system.barrier_energy / system.mass) * (
        np.pi / system.lattice_constant
    )


def _plot_crossing_time() -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_mass(system.mass)

    fig, ax = get_fancy_figure()

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
    momentum_squared = get_momentum_squared_per_state(hamiltonian, config.direction)[
        "data"
    ].reshape(n_bands, -1)

    energy_per_state = hamiltonian["data"]

    energies_2d = energy_per_state.reshape((n_bands, -1))
    energies_2d = np.real(energies_2d)
    momentum_squared = momentum_squared.reshape((n_bands, -1))
    for b in range(n_bands):
        sort_idx = np.argsort(energies_2d[b, :])
        energies = energies_2d[b, sort_idx]
        quantum_crossing_time = (system.mass * system.lattice_constant) / np.sqrt(
            np.real_if_close(momentum_squared[b, sort_idx]),
        )

        (line,) = ax.plot(
            (energies - system.barrier_energy) / (Boltzmann * config.temperature),
            quantum_crossing_time,
            label=f"Band {b}",
        )
        line.set_color(CAM_BLUE.warm)
    quantum_line = ax.plot([], [], color=CAM_BLUE.warm, label="Quantum")[0]

    classical_energies = np.linspace(-1, 2, 1000)
    (classical_line,) = ax.plot(
        classical_energies,
        _get_classical_crossing_time(
            system,
            system.barrier_energy
            + (classical_energies * Boltzmann * config.temperature),
        ),
    )
    classical_line.set_label("Classical")
    classical_line.set_color(CAM_BLUE.dark)

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"Crossing time $\tau(E)$")
    ax.set_ylim(0, 2e-12)
    ax.set_xlim(-0.5, 1.0)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper right",
        handles=[line, quantum_line, classical_line],
        labels=["$E_b$", "Actual", "Classical"],
    )

    fig.savefig("scripts/perturbation/quantum_crossing_time.pdf", bbox_inches="tight")


def _plot_quantum_tunnelling_correction() -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    system = system.with_mass(4.00260 * atomic_mass)

    fig, ax = get_fancy_figure()

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
    momentum_squared = get_momentum_squared_per_state(
        hamiltonian,
        config.direction,
    )["data"].reshape(n_bands, -1)

    energy_per_state = hamiltonian["data"]

    energies_2d = energy_per_state.reshape((n_bands, -1))
    energies_2d = np.real(energies_2d)
    momentum_squared = momentum_squared.reshape((n_bands, -1))
    for b in range(n_bands):
        sort_idx = np.argsort(energies_2d[b, :])
        energies = energies_2d[b, sort_idx]

        classical_crossing_time = _get_classical_crossing_time(
            system,
            energies,
        )
        quantum_crossing_time = (system.mass * system.lattice_constant) / np.sqrt(
            np.real_if_close(momentum_squared[b, sort_idx]),
        )

        (line,) = ax.plot(
            (energies - system.barrier_energy) / (Boltzmann * config.temperature),
            (classical_crossing_time / quantum_crossing_time) ** 2,
            label=f"Band {b}",
        )
        line.set_color(CAM_BLUE.warm)
    quantum_line = ax.plot([], [], color=CAM_BLUE.warm, label="Quantum")[0]

    energies = np.linspace(-1, 2, 1000)
    barrier_omega = _get_barrier_omega(system)
    kemble_probability = _get_kemble_formula_probability(
        system.barrier_energy,
        barrier_omega,
        (energies * Boltzmann * config.temperature) + system.barrier_energy,
    )
    (kemble_probability_line,) = ax.plot(
        energies,
        kemble_probability,
        color=CAM_BLUE.dark,
        label="Kemble",
    )

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"Tunneling probability $P(E)$")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(-0.5, 1.0)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper right",
        handles=[line, quantum_line, kemble_probability_line],
        labels=["$E_b$", "Actual", "Kemble"],
    )

    fig.savefig("scripts/perturbation/quantum_tunnelling_rate.pdf")


if __name__ == "__main__":
    _plot_crossing_time()
    _plot_quantum_tunnelling_correction()

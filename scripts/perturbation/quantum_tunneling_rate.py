from typing import Any

import numpy as np
import scipy
import scipy.special
from scipy.constants import Boltzmann, hbar

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


def _plot_quantum_tunnelling_correction() -> None:

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
    momentum = get_momentum_squared_per_state(
        hamiltonian,
        config.direction,
    )["data"].reshape(n_bands, -1)

    energy_per_state = hamiltonian["data"]

    energies_2d = energy_per_state.reshape((n_bands, -1))
    energies_2d = np.real(energies_2d)
    momentum = momentum.reshape(
        (n_bands, -1),
    )
    for b in range(n_bands):
        sort_idx = np.argsort(energies_2d[b, :])
        energies = energies_2d[b, sort_idx]

        classical_momentum = _get_classical_momentum_squared(system, energies)

        (line,) = ax.plot(
            (energies - system.barrier_energy) / (Boltzmann * config.temperature),
            np.real_if_close(momentum[b, sort_idx] / classical_momentum),
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
    ax.set_xlim(-0.1, 1.0)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper right",
        handles=[line, quantum_line, kemble_probability_line],
        labels=["$E_b$", "Actual", "Kemble"],
    )

    fig.savefig("scripts/perturbation/quantum_tunneling_rate.pdf", bbox_inches="tight")


if __name__ == "__main__":
    _plot_quantum_tunnelling_correction()

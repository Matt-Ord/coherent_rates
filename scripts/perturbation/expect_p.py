from typing import Any

import numpy as np
import scipy
from matplotlib.scale import LogScale
from scipy.constants import Boltzmann, hbar

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.isf import (
    get_momentum_squared_per_state,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystem1d,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
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

    # Assumes unit cell length L = 1. If system has a length attribute
    # (example system.length) multiply the prefactor by system.length.
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


def _get_semi_classical_momentum_squared(
    system: PeriodicSystem1d,
    energies: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate the semi-classical momentum squared using the crossing time."""
    barrier_omega = _get_barrier_omega(system)
    tunneling_prob = _get_kemble_formula_probability(
        system.barrier_energy,
        barrier_omega,
        energies,
    )
    crossing_time = _get_classical_crossing_time(system, energies)

    with np.errstate(divide="ignore", invalid="ignore"):
        unweighted_momentum_squared = (
            system.mass * system.lattice_constant / crossing_time
        ) ** 2

    unweighted_momentum_squared = np.nan_to_num(
        unweighted_momentum_squared,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return tunneling_prob * unweighted_momentum_squared


def _plot_momentum_squared_with_classical(*, semi_classical: bool = False) -> None:

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

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
        line.set_color(CAM_BLUE.base)
    quantum_line = ax.plot([], [], color=CAM_BLUE.warm, label="Quantum")[0]

    energies = np.linspace(-1, 2, 1000)

    if semi_classical:
        semi_classical_momentum = _get_semi_classical_momentum_squared(
            system,
            (energies * Boltzmann * config.temperature) + system.barrier_energy,
        )
        semi_classical_momentum /= (2 * system.mass) * (
            energies * Boltzmann * config.temperature + system.barrier_energy
        )
        (classical_line,) = ax.plot(
            energies,
            semi_classical_momentum,
            color=CAM_BLUE.dark,
        )
    else:
        classical_momentum = _get_classical_momentum_squared(
            system,
            (energies * Boltzmann * config.temperature) + system.barrier_energy,
        )
        classical_momentum /= (2 * system.mass) * (
            energies * Boltzmann * config.temperature + system.barrier_energy
        )
        (classical_line,) = ax.plot(energies, classical_momentum, color=CAM_BLUE.dark)

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"$\langle p_d^2\rangle$ / $2mE$")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-1, 2)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper left",
        fontsize=8,
        handles=[line, classical_line, quantum_line],
        labels=["$E_b$", "Semiclassical", "Quantum"]
        if semi_classical
        else ["$E_b$", "Classical", "Quantum"],
    )

    fig.savefig("scripts/perturbation/expect_p.classical.pdf", bbox_inches="tight")


def _plot_momentum_squared_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
        # offset=(0.01, 0.01),  # Breaks some of the symmetry  # noqa: ERA001
    )

    fig, ax = get_thesis_figure()

    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    n_bands = hamiltonian["basis"][0].wavefunctions["basis"][0].shape[0]
    momentum = get_momentum_squared_per_state(
        hamiltonian,
        config.direction,
    )["data"].reshape(n_bands, -1)

    energy_per_state = hamiltonian["data"]

    energies_2d = energy_per_state.reshape((n_bands, -1))
    energies_2d = np.real_if_close(energies_2d)
    scaled_momentum = (
        momentum / (system.mass * config.temperature * Boltzmann)
    ).reshape((n_bands, -1))

    scaled_energy = (energies_2d - system.barrier_energy) / (
        Boltzmann * config.temperature
    )

    for b in range(100):
        sort_idx = np.argsort(energies_2d[b, :])
        (line,) = ax.plot(
            scaled_energy[b, sort_idx],
            np.real_if_close(scaled_momentum[b, sort_idx]),
            label=f"Band {b}",
        )
        if b % 2 == 0:
            line.set_color(CAM_BLUE.dark)
        else:
            line.set_color(CAM_BLUE.warm)

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"$\langle\hat{p}\rangle^2$ / $2mE$")
    ax.set_xlim(-1, 2)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(loc="upper left", handles=[line], labels=["$E_b$"])

    fig.savefig("scripts/perturbation/expect_p.2d.pdf", bbox_inches="tight")
    fig.show()
    input()


if __name__ == "__main__":
    _plot_momentum_squared()
    _plot_momentum_squared_with_classical(semi_classical=True)
    _plot_momentum_squared_2d()

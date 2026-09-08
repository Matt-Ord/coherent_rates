from typing import Any

import numpy as np
import scipy
from matplotlib.scale import LogScale
from scipy.constants import Boltzmann

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


def _plot_momentum_squared_with_classical() -> None:

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
        line.set_color(CAM_BLUE.warm)
    quantum_line = ax.plot([], [], color=CAM_BLUE.warm, label="Quantum")[0]

    energies = np.linspace(-1, 2, 1000)
    classical_momentum = _get_classical_momentum_squared(
        system,
        (energies * Boltzmann * config.temperature) + system.barrier_energy,
    )
    classical_momentum /= (2 * system.mass) * (
        energies * Boltzmann * config.temperature + system.barrier_energy
    )
    (classical_line,) = ax.plot(energies, classical_momentum, color=CAM_BLUE.dark)

    ax.set_xlabel("$(E - E_b)$/ $k_b T$")
    ax.set_ylabel(r"$\langle\hat{p}\rangle^2$ / $2mE$")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-1, 2)

    line = ax.axvline(0)
    line.set_linestyle("--")
    line.set_color(CAM_CHERRY.dark)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        handles=[line, classical_line, quantum_line],
        labels=["$E_b$", "Classical", "Quantum"],
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

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        handles=[line],
        labels=["$E_b$"],
    )

    fig.savefig("scripts/perturbation/expect_p.2d.pdf", bbox_inches="tight")
    fig.show()
    input()


if __name__ == "__main__":
    _plot_momentum_squared()
    _plot_momentum_squared_with_classical()
    _plot_momentum_squared_2d()

import numpy as np
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
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
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


def _plot_momentum_squared_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
        # offset=(0.01, 0.01),  # Breaks some of the symmetry
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
    average_energy = np.mean(scaled_energy, axis=1)
    _u = average_energy < 2
    print(np.argmin(average_energy < 2))
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
    _plot_momentum_squared_2d()

"""Plot the thermal distribution of quantum translational momentum.

The translational ISF, which is the cosine transform of the distribution, is
also compared with the Gaussian decay of a free particle with the effective
mass, m_eff / m = m k_B T / <p_tr^2>.

The translational momentum of each Bloch eigenstate is the magnitude of its
diagonal momentum, |<n|p|n>|, and each eigenstate is weighted by its Boltzmann
factor. The distribution is compared with the exact classical distribution on
the same potential, which in units of sqrt(m k_B T) is independent of mass.

The figures mirror translational_momentum_distribution.{flat,corrugated}.pdf in
thesis_calculations, which show the classical distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Patch
from scipy.constants import Boltzmann, atomic_mass, hbar
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import ellipk

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, PeriodicSystem1d
from coherent_rates.util import CAM_BLUE, get_thesis_figure

HELIUM_MASS = 4.00260 * atomic_mass
TEMPERATURE = 155
P_RANGE = (0.0, 4.0)
N_GRID = 4000
BROADENING = 0.01
N_K_INTERPOLATED = 100000
# The momentum transfer of the classical ISF figures
DELTA_K = 0.43e10


def _get_quantum_translational_momentum(
    system: PeriodicSystem1d,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return |<n|p|n>| / sqrt(m k_B T) and the thermal weight of each state.

    Both are returned with shape (n_bands, n_k).

    In 1D, the diagonal momentum of a Bloch state is <n|p|n> = (m / hbar)
    dE_n / dk. Each band E_n(k) is interpolated with a periodic spline, and
    sampled on a dense grid of k, which removes the noise from sampling a
    finite number of k points. The van Hove singularities at the maximum
    speed of each band remain, as they are a property of the band structure.
    """
    config = PeriodicSystemConfig(
        (2000,),
        (100,),
        direction=(10,),
        truncation=50,
        temperature=TEMPERATURE,
    )
    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    n_bands = hamiltonian["basis"][0].wavefunctions["basis"][0].shape[0]
    energies = np.real(hamiltonian["data"]).reshape(n_bands, -1)

    # The k points are stored in FFT order, so the band is sorted into
    # [-pi / dx, pi / dx) and closed at pi / dx for the periodic spline
    n_k = energies.shape[1]
    k_scale = 2 * np.pi / system.lattice_constant
    order = np.argsort(np.fft.fftfreq(n_k))
    k = np.append(np.sort(np.fft.fftfreq(n_k)), 0.5) * k_scale
    sorted_energies = energies[:, order]
    spline = CubicSpline(
        k,
        np.concatenate([sorted_energies, sorted_energies[:, :1]], axis=1),
        axis=1,
        bc_type="periodic",
    )

    k_dense = np.linspace(-0.5, 0.5, N_K_INTERPOLATED, endpoint=False) * k_scale
    energies_dense = spline(k_dense)
    momentum = np.abs(system.mass * spline(k_dense, 1) / hbar)

    kbt = Boltzmann * TEMPERATURE
    weights = np.exp(-(energies_dense - np.min(energies_dense)) / kbt)
    weights /= np.sum(weights)
    return momentum / np.sqrt(system.mass * kbt), weights


def _get_classical_translational_momentum(
    barrier_energy_ratio: float,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return the continuous part of the classical rho(p_tr), p_tr / sqrt(m k_B T).

    The weight of each energy E = epsilon E_b is e^(-beta E) I(epsilon), where
    I(epsilon) is the dimensionless crossing time above the barrier and the
    half period below it. The trapped adsorbates form a delta function at zero,
    which is not included.
    """
    u0 = barrier_energy_ratio

    def trapped(epsilon: float) -> float:
        return np.exp(-u0 * epsilon) * ellipk(epsilon)

    def escaping(epsilon: float) -> float:
        return np.exp(-u0 * epsilon) * ellipk(1 / epsilon) / np.sqrt(epsilon)

    partition = quad(trapped, 0, 1, limit=500)[0] + quad(escaping, 1, np.inf)[0]

    epsilon = 1 + np.geomspace(1e-12, 20, 4000)
    p_tr = np.pi * np.sqrt(u0 / 2) * np.sqrt(epsilon) / ellipk(1 / epsilon)
    weight = escaping(epsilon) / partition  # type: ignore[arg-type]
    return p_tr, weight / np.gradient(p_tr, epsilon)


def _get_quantum_density(
    p_tr: np.ndarray[Any, np.dtype[np.float64]],
    weights: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return the contribution of each band to rho(p_tr) on a fine grid.

    The density of each band diverges at its maximum speed, so the density is
    broadened by a Gaussian of width BROADENING sqrt(m k_B T). Since p_tr is a
    magnitude, the density is reflected at p_tr = 0 before broadening.
    """
    edges = np.linspace(*P_RANGE, N_GRID + 1)
    width = edges[1] - edges[0]
    rho = np.array(
        [
            np.histogram(p_band, bins=edges, weights=w_band)[0] / width
            for p_band, w_band in zip(p_tr, weights, strict=True)
        ],
    )
    rho = gaussian_filter1d(rho, BROADENING / width, axis=1, mode="reflect")
    return 0.5 * (edges[1:] + edges[:-1]), rho


def _plot_translational_momentum_distribution(
    system: PeriodicSystem1d,
    name: str,
) -> None:
    p_quantum, weights = _get_quantum_translational_momentum(system)
    kbt = Boltzmann * TEMPERATURE
    p_classical, rho_classical = _get_classical_translational_momentum(
        system.barrier_energy / kbt,
    )

    fig, ax = get_thesis_figure()
    # The contribution of each band is stacked on the bands above it in
    # energy, with alternating colours to distinguish neighbouring bands
    p_grid, rho_bands = _get_quantum_density(p_quantum, weights)
    rho_stacked = np.cumsum(rho_bands[::-1], axis=0)[::-1]
    rho_below = np.zeros_like(p_grid)
    for band in reversed(range(rho_stacked.shape[0])):
        rho_total = rho_stacked[band]
        color = CAM_BLUE.warm if band % 2 == 0 else CAM_BLUE.base
        ax.fill_between(p_grid, rho_below, rho_total, color=color, lw=0)
        rho_below = rho_total
    quantum_patch = Patch(color=CAM_BLUE.warm, label="quantum")

    (line,) = ax.plot(p_classical, rho_classical, color=CAM_BLUE.dark)
    line.set_linestyle("--")
    line.set_label("classical")

    # The weight close to zero is far taller than the continuous part, so
    # the axis is cut off to show the shape of the continuous part
    ax.set_ylim(0, 1.4 * float(np.max(rho_classical)))
    ax.set_xlim(*P_RANGE)

    ax.set_xlabel(r"$p_\mathrm{tr} / \sqrt{m k_B T}$")
    ax.set_ylabel(r"$\rho(p_\mathrm{tr})$")
    ax.legend(handles=[quantum_patch, line], loc="upper right")
    fig.savefig(f"scripts/thesis/translational_momentum_distribution.{name}.pdf")


def _get_quantum_translational_isf(
    system: PeriodicSystem1d,
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return the translational ISF, with the quantum recoil phase removed."""
    p_tr, weights = _get_quantum_translational_momentum(system)
    # The momenta are binned finely enough that the phase error within a
    # bin is negligible over the times of interest
    counts, edges = np.histogram(p_tr, bins=40000, range=(0, 8), weights=weights)
    p_centres = 0.5 * (edges[1:] + edges[:-1])
    velocity = p_centres * np.sqrt(system.mass * Boltzmann * TEMPERATURE) / system.mass
    return np.cos(DELTA_K * np.outer(times, velocity)) @ counts


def _get_effective_mass_isf(
    system: PeriodicSystem1d,
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return the ISF of a free particle with the quantum effective mass.

    In units of sqrt(m k_B T), k_B T / m_eff = <p_tr^2> k_B T / m, so the
    free particle ISF exp(-k_B T dk^2 t^2 / 2 m_eff) depends only on <p_tr^2>.
    """
    p_tr, weights = _get_quantum_translational_momentum(system)
    kbt = Boltzmann * TEMPERATURE
    effective_mass = system.mass / np.sum(weights * p_tr**2)
    return np.exp(-kbt * DELTA_K**2 * times**2 / (2 * effective_mass))


def _plot_translational_isf(system: PeriodicSystem1d, name: str) -> None:
    # Time is measured in units of the decay rate alpha of a free particle
    # with the true mass, so that the free particle decays as exp(-(alpha t)^2)
    alpha = np.sqrt(Boltzmann * TEMPERATURE / (2 * system.mass)) * DELTA_K
    times = np.linspace(0, 4 / alpha, 800)
    fig, ax = get_thesis_figure()
    (line,) = ax.plot(
        times * alpha,
        _get_quantum_translational_isf(system, times),
        color=CAM_BLUE.warm,
    )
    line.set_label("quantum")
    (mass_line,) = ax.plot(
        times * alpha,
        _get_effective_mass_isf(system, times),
        color=CAM_BLUE.dark,
    )
    mass_line.set_linestyle("--")
    mass_line.set_label("effective mass")

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\alpha t$")
    ax.set_ylabel(r"$I_\mathrm{tr}(\Delta k, t)$")
    ax.legend(handles=[line, mass_line], loc="upper right")
    fig.savefig(f"scripts/thesis/translational_isf.{name}.pdf")


if __name__ == "__main__":
    _plot_translational_momentum_distribution(SODIUM_COPPER_BRIDGE_SYSTEM_1D, "sodium")
    _plot_translational_momentum_distribution(
        SODIUM_COPPER_BRIDGE_SYSTEM_1D.with_mass(HELIUM_MASS),
        "helium",
    )
    _plot_translational_isf(SODIUM_COPPER_BRIDGE_SYSTEM_1D, "sodium")
    _plot_translational_isf(
        SODIUM_COPPER_BRIDGE_SYSTEM_1D.with_mass(HELIUM_MASS),
        "helium",
    )

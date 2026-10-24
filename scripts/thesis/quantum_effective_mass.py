"""Compare the quantum effective mass of an adsorbate with the classical limit.

On the 1D cosine potential V(x) = E_b / 2 (1 - cos(2 pi x / dx)), the ratio
m_eff / m depends only on two dimensionless numbers, the barrier energy
E_b / k_B T and the quantum of barrier energy hbar omega_b / k_B T, where
omega_b = (pi / dx) sqrt(2 E_b / m) is the barrier frequency. In the
classical limit hbar omega_b / k_B T -> 0, and the ratio depends only on
E_b / k_B T.

Three models of the effective mass m_eff / m = m k_B T / <p_tr^2> are compared

- The exact result, from the Bloch states of the cosine potential, which is
  the Mathieu equation. In a plane wave basis the Hamiltonian is tridiagonal,
  and the diagonal momentum of each Bloch state follows from the
  Hellmann-Feynman theorem without numerical differentiation.
- A phase-integral model, which replaces the Bloch states by the
  semiclassical band condition
      cos(k dx) = cos(Theta(E) + phi(eps)) / |t(E)|,
  where Theta is the action across the well, |t|^2 = P is the Kemble
  transmission probability through a parabolic barrier, eps is the energy
  below the barrier in units of hbar omega_b, and phi is the phase correction
  of the parabolic barrier.
- A closed-form semiclassical model, which averages the band condition over
  each band. The average of p_tr^2 over a band at energy E is the classical
  p_tr^2 multiplied by 1 - sqrt(1 - P(E)) = 1 - |r(E)|, where |r| is the
  amplitude of reflection from the barrier.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter
from scipy.constants import Boltzmann, atomic_mass, hbar
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.special import ellipe, ellipk, expit, gamma, i0, i0e, i1e, loggamma
from surface_potential_analysis.util.decorators import cached

from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, PeriodicSystem1d
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_SLATE_1,
    CAM_SLATE_2,
    CAM_SLATE_4,
    get_fancy_figure,
    get_thesis_fig_size,
    get_thesis_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

TEMPERATURE = 155
N_K = 256
# Plane waves are included up to a kinetic energy of E_b + MAX_KINETIC k_B T
MAX_KINETIC = 30

ADSORBATE_MASSES = {
    "H": 1.00784 * atomic_mass,
    "D": 2.01410 * atomic_mass,
    "He": 4.00260 * atomic_mass,
    "Li": 6.94100 * atomic_mass,
    "Na": SODIUM_COPPER_BRIDGE_SYSTEM_1D.mass,
    "K": 39.0983 * atomic_mass,
    "Cs": 132.905 * atomic_mass,
}


def _get_dimensionless_parameters(system: PeriodicSystem1d) -> tuple[float, float]:
    """Return E_b / k_B T and hbar omega_b / k_B T for a system."""
    kbt = Boltzmann * TEMPERATURE
    omega_b = (np.pi / system.lattice_constant) * np.sqrt(
        2 * system.barrier_energy / system.mass,
    )
    return system.barrier_energy / kbt, hbar * omega_b / kbt


def _get_bloch_states(
    barrier_ratio: float,
    omega_ratio: float,
    *,
    include_edges: bool = False,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return E / k_B T and p_tr^2 / (m k_B T) of each Bloch state.

    Both are returned with shape (n_k, n_bands). Energies are measured in
    units of k_B T, and wavevectors in units of 2 pi / dx. The kinetic energy
    of a plane wave q is then e_R q^2, where e_R = (hbar omega_b)^2 /
    (E_b k_B T). The momentum of a plane wave is 2 pi hbar q / dx, so
    p^2 / (m k_B T) = 2 e_R q^2.

    Since E(k) = E(-k), only the half of the Brillouin zone with k > 0 is
    sampled, and only the bands below E_b + MAX_KINETIC k_B T are kept.
    By default k is sampled at the midpoints of N_K // 2 intervals, which
    gives the uniform average over the zone. With include_edges, k runs from
    the centre to the edge of the zone, where the momentum vanishes.
    """
    recoil = omega_ratio**2 / barrier_ratio
    n_g = int(np.sqrt((barrier_ratio + MAX_KINETIC) / recoil)) + 12
    g = np.arange(-n_g, n_g + 1)
    off_diagonal = np.full(g.size - 1, -barrier_ratio / 4)

    def _diagonal(k: float) -> np.ndarray[Any, np.dtype[np.float64]]:
        return (recoil * (k + g) ** 2 + barrier_ratio / 2).astype(np.float64)

    band_energies = np.asarray(
        eigvalsh_tridiagonal(_diagonal(0), off_diagonal),
        dtype=np.float64,
    )
    n_bands = int(
        np.sum(band_energies < barrier_ratio + MAX_KINETIC),
    )
    if include_edges:
        k = np.linspace(0, 0.5, N_K // 2)
    else:
        k = 0.5 * (np.arange(N_K // 2) + 0.5) / (N_K // 2)
    energies = np.empty((k.size, n_bands))
    p_squared = np.empty((k.size, n_bands))
    for i, k_i in enumerate(k):
        energies[i], vectors = eigh_tridiagonal(
            _diagonal(k_i),
            off_diagonal,
            select="i",
            select_range=(0, n_bands - 1),
        )
        # The diagonal momentum of each Bloch state, by Hellmann-Feynman
        q_expect = np.einsum("gn,g->n", vectors**2, k_i + g)
        p_squared[i] = 2 * recoil * q_expect**2
    if include_edges:
        # By symmetry, the momentum vanishes at the centre and edge of the zone
        p_squared[[0, -1]] = 0
    return energies, p_squared


def get_exact_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return the exact m_eff / m = m k_B T / <p_tr^2> on the cosine potential."""
    energies, p_squared = _get_bloch_states(barrier_ratio, omega_ratio)
    weights = np.exp(-(energies - np.min(energies)))
    return float(np.sum(weights) / np.sum(weights * p_squared))


def _get_crossing_time_integral(
    u: float | np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return I(u), the crossing time in units of (dx / pi) sqrt(m / 2 E_b).

    Above the barrier I(u) is the time to cross the unit cell, and below the
    barrier it is half the period of oscillation within the well.
    """
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    below = u < 1
    out[below] = 2 * ellipk(u[below])
    out[~below] = 2 * ellipk(1 / u[~below]) / np.sqrt(u[~below])
    return out


def _get_kemble_probability(
    u: np.ndarray[Any, np.dtype[np.float64]],
    barrier_ratio: float,
    omega_ratio: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return the Kemble probability P(E) of transmission, E = u E_b."""
    return expit(2 * np.pi * barrier_ratio * (u - 1) / omega_ratio)


def _get_semiclassical_mass_ratio_from_weight(
    barrier_ratio: float,
    weight: Any,  # noqa: ANN401
) -> float:
    """Return m_eff / m, with p_tr^2 = weight(u) times the classical value.

    In units of E_b, with the crossing time I(u) of the classical motion

        m_eff / m = (1 / (2 pi^2 e_b)) int e^(-e_b u) I(u) du
                                      / int e^(-e_b u) weight(u) / I(u) du.

    With weight(u) = theta(u - 1), the result is the exact classical ratio.
    """

    def numerator(u: float) -> float:
        return float(np.exp(-barrier_ratio * u) * _get_crossing_time_integral(u))

    def denominator(u: float) -> float:
        return float(
            np.exp(-barrier_ratio * u) * weight(u) / _get_crossing_time_integral(u),
        )

    u_max = 1 + 80 / barrier_ratio
    partition = quad(numerator, 0, 1, limit=500)[0]
    partition += quad(numerator, 1, u_max, limit=500)[0]
    flux = quad(denominator, 0, 1, limit=500)[0]
    flux += quad(denominator, 1, u_max, limit=500)[0]
    return partition / (2 * np.pi**2 * barrier_ratio * flux)


def get_classical_mass_ratio(barrier_ratio: float) -> float:
    """Return the classical m_eff / m, which depends only on E_b / k_B T."""
    return _get_semiclassical_mass_ratio_from_weight(
        barrier_ratio,
        lambda u: float(u > 1),
    )


def get_band_averaged_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return m_eff / m with the band averaged weight 1 - sqrt(1 - P)."""
    return _get_semiclassical_mass_ratio_from_weight(
        barrier_ratio,
        lambda u: (
            1 - np.sqrt(1 - _get_kemble_probability(u, barrier_ratio, omega_ratio))
        ),
    )


def get_tunnelling_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return m_eff / m with the weight P of each band centre."""
    return _get_semiclassical_mass_ratio_from_weight(
        barrier_ratio,
        lambda u: _get_kemble_probability(u, barrier_ratio, omega_ratio),
    )


def get_exact_partition_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return Z / Z_cl, the exact quantum partition function of the Bloch states.

    Per unit cell, with wavevectors in units of 2 pi / dx, the classical
    partition function is sqrt(pi / e_R) e^(-a) I_0(a), where a = e_b / 2.
    """
    energies, _ = _get_bloch_states(barrier_ratio, omega_ratio)
    recoil = omega_ratio**2 / barrier_ratio
    quantum = np.mean(np.sum(np.exp(-energies), axis=1))
    a = barrier_ratio / 2
    return float(quantum / (np.sqrt(np.pi / recoil) * np.exp(-a) * i0(a)))


def get_wigner_partition_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return Z / Z_cl to first order in hbar^2, the Wigner-Kirkwood expansion.

    For a general potential, Z / Z_cl = 1 - (beta hbar)^2 <V''(x)>_cl / 24 m,
    where the average is taken over the classical Boltzmann distribution. On
    the cosine potential V''(x) = m omega_b^2 cos(2 pi x / dx), and the
    classical average of the cosine is I_1(a) / I_0(a), with a = E_b / 2 k_B T.
    In a deep well I_1(a) / I_0(a) -> 1, and the result reduces to the first
    term of the harmonic (w / 2) / sinh(w / 2).
    """
    a = barrier_ratio / 2
    return float(1 - omega_ratio**2 / 24 * i1e(a) / i0e(a))


def get_semiclassical_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return m_eff / m with the band averaged flux and the Wigner-Kirkwood Z.

    Since m_eff / m is proportional to Z, the band averaged ratio is scaled by
    the first-order correction to the partition function.
    """
    return get_wigner_partition_ratio(
        barrier_ratio,
        omega_ratio,
    ) * get_band_averaged_mass_ratio(barrier_ratio, omega_ratio)


def get_rate_theory_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return m_eff / m with a transmission coefficient, as in quantum rate theory.

    The crossing rate 1 / tau is taken to be the same at all energies, so the
    band averaged weight 1 - sqrt(1 - P) reduces the classical flux by

        kappa = beta int e^(-beta (E - E_b)) (1 - sqrt(1 - P)) dE
              = Gamma(1 - s) Gamma(1 / 2 + s) / sqrt(pi),

    where s = hbar omega_b / (2 pi k_B T), the analogue of the Bell
    coefficient for the weight P. The partition function takes the
    Wigner-Kirkwood correction.
    """
    s = omega_ratio / (2 * np.pi)
    kappa = gamma(1 - s) * gamma(0.5 + s) / np.sqrt(np.pi)
    return float(
        get_classical_mass_ratio(barrier_ratio)
        * get_wigner_partition_ratio(barrier_ratio, omega_ratio)
        / kappa,
    )


def get_series_mass_ratio(barrier_ratio: float, omega_ratio: float) -> float:
    """Return m_eff / m from the second-order expansion of the band averaged flux.

    The weight 1 - sqrt(1 - P) differs from the classical step only within a
    few hbar omega_b of the barrier top, where tau(E) varies only
    logarithmically. The crossing time there is taken to be the classical
    Lambda = ln(16 E_b / k_B T) + gamma_E with k_B T replaced by hbar omega_b.
    Expanding the Boltzmann factor across the barrier region, the Boltzmann
    average of the weight is

        eta = 1 - (ln 2 / pi) w + (1 / 12 + (ln 2)^2 / (2 pi^2)) w^2,

    with w = hbar omega_b / k_B T. Relative to the classical result,

        m / m_eff = (m / m_eff)_cl (1 - A (1 - eta) + delta_zpe),

    where A = omega_b / (<1 / tau>_xi (Lambda - ln w)), and the last term is
    the Wigner-Kirkwood correction to the partition function.
    """
    flux = quad(
        lambda u: float(
            np.exp(-barrier_ratio * (u - 1)) / _get_crossing_time_integral(u),
        ),
        1,
        1 + 80 / barrier_ratio,
        limit=500,
    )[0]
    # omega_b / <1 / tau>_xi, from the classical flux in units of E_b
    crossing = 1 / (barrier_ratio * flux)
    prefactor = crossing / (np.log(16 * barrier_ratio / omega_ratio) + np.euler_gamma)
    reflection = np.log(2) / np.pi
    tunnelling = 1 / 12 + reflection**2 / 2
    correction = (
        -prefactor * (reflection * omega_ratio - tunnelling * omega_ratio**2)
        + 1
        - get_wigner_partition_ratio(barrier_ratio, omega_ratio)
    )
    return float(get_classical_mass_ratio(barrier_ratio) / (1 + correction))


def _get_well_action(
    u: np.ndarray[Any, np.dtype[np.float64]],
    barrier_ratio: float,
    omega_ratio: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return Theta(E) / hbar, the action across the well, for E = u E_b.

    Below the barrier, Theta is the action between the classical turning
    points. Above the barrier, the turning points move into the complex
    plane, and the real part of the action is taken across the unit cell.
    """
    prefactor = 4 * barrier_ratio / omega_ratio
    out = np.empty_like(u)
    below = u < 1
    out[below] = ellipe(u[below]) - (1 - u[below]) * ellipk(u[below])
    out[~below] = np.sqrt(u[~below]) * ellipe(1 / u[~below])
    return prefactor * out


def _get_parabolic_phase(
    eps: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return the phase correction of a parabolic barrier.

    phi(eps) = arg Gamma(1 / 2 + i eps) - eps ln|eps| + eps, which removes the
    logarithmic divergence of the crossing time at the top of the barrier.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.imag(loggamma(0.5 + 1j * eps)) - eps * np.log(np.abs(eps)) + eps
    return np.where(eps == 0, 0.0, phase)


def _get_phase_integral_slope_squared(
    phase: np.ndarray[Any, np.dtype[np.float64]],
    probability: np.ndarray[Any, np.dtype[np.float64]],
    phase_slope: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return (dE / d(k dx))^2 of the states with total phase S.

    Differentiating the band condition cos(k dx) = cos(S) / sqrt(P), and
    neglecting the variation of P across a band, gives
    dE / d(k dx) = sqrt(P - cos^2 S) / (|sin S| dS / dE), which vanishes at
    the edges of each band.
    """
    return np.clip(probability - np.cos(phase) ** 2, 0, None) / (
        phase_slope**2 * np.sin(phase) ** 2
    )


def get_phase_integral_mass_ratio(
    barrier_ratio: float,
    omega_ratio: float,
    *,
    n_energy: int = 400000,
    n_per_band: int = 3000,
) -> float:
    """Return m_eff / m from the semiclassical band condition.

    The total phase S(E) = Theta(E) + phi(eps) increases monotonically with
    energy, and band n is centred at S = (n + 1 / 2) pi. Within each band,
    cos(k dx) = cos(S) / |t|, and the momentum of each state is
    p = (m dx / hbar) dE / d(k dx), with dE / d(k dx) from
    _get_phase_integral_slope_squared. States are sampled on a grid of S which
    is concentrated at the centre of each band, so that the narrow bands
    deep in the well are resolved.
    """
    u_max = 1 + MAX_KINETIC / barrier_ratio
    u = np.linspace(1e-8, u_max, n_energy)
    eps = barrier_ratio * (1 - u) / omega_ratio
    phase = _get_well_action(u, barrier_ratio, omega_ratio) + _get_parabolic_phase(eps)
    # dS / dE, with energies in units of k_B T
    phase_slope = np.gradient(phase, barrier_ratio * u)

    offset = np.geomspace(1e-15, 1, n_per_band) * np.pi / 2
    offset = np.concatenate([-offset[::-1], offset])

    partition = 0.0
    flux = 0.0
    for n in range(int(phase[-1] / np.pi) - 1):
        band_phase = (n + 0.5) * np.pi + offset
        # Energies in units of k_B T
        energy = barrier_ratio * np.interp(band_phase, phase, u)
        probability = expit((energy - barrier_ratio) * 2 * np.pi / omega_ratio)
        cos_k = np.clip(np.cos(band_phase) / np.sqrt(probability), -1, 1)
        k_dx = np.arccos(cos_k)

        d_k = np.abs(np.diff(k_dx))
        mid_phase = 0.5 * (band_phase[1:] + band_phase[:-1])
        mid_energy = 0.5 * (energy[1:] + energy[:-1])
        slope_squared = _get_phase_integral_slope_squared(
            mid_phase,
            0.5 * (probability[1:] + probability[:-1]),
            np.interp(mid_energy, barrier_ratio * u, phase_slope),
        )
        weight = np.exp(-(mid_energy - barrier_ratio))
        partition += np.sum(weight * d_k) / np.pi
        flux += np.sum(weight * slope_squared * d_k) / np.pi

    # In these units, p^2 / (m k_B T) = (2 pi^2 e_b / w^2) (dE / d(k dx))^2
    p_squared = 2 * np.pi**2 * barrier_ratio / omega_ratio**2 * flux / partition
    return 1 / p_squared


def _plot_mass_ratio_against_mass() -> None:
    base = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    masses = np.geomspace(0.8 * atomic_mass, 1000 * atomic_mass, 60)
    parameters = [_get_dimensionless_parameters(base.with_mass(m)) for m in masses]
    barrier_ratio = parameters[0][0]

    exact = np.array([get_exact_mass_ratio(*p) for p in parameters])
    semiclassical = np.array([get_semiclassical_mass_ratio(*p) for p in parameters])
    classical = get_classical_mass_ratio(barrier_ratio)

    fig, ax = get_fancy_figure()
    mass_u = masses / atomic_mass
    (exact_line,) = ax.plot(mass_u, exact, color=CAM_BLUE.warm)
    (semiclassical_line,) = ax.plot(mass_u, semiclassical, color=CAM_CHERRY.dark)
    semiclassical_line.set_linestyle("--")
    classical_line = ax.axhline(classical, color=CAM_BLUE.dark)
    classical_line.set_linestyle("--")

    for name, mass in ADSORBATE_MASSES.items():
        ratio = get_exact_mass_ratio(
            *_get_dimensionless_parameters(base.with_mass(mass)),
        )
        ax.scatter(mass / atomic_mass, ratio, color=CAM_BLUE.dark, marker="x", zorder=5)
        ax.annotate(
            name,
            (mass / atomic_mass, ratio),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.8, mass_u[-1])
    ax.set_ylim(2.2, 2.8)
    ax.set_xlabel(r"Mass / $\mathrm{u}$")
    ax.set_ylabel(r"$m_\mathrm{eff} / m$")
    ax.legend(
        handles=[
            exact_line,
            semiclassical_line,
            classical_line,
        ],
        labels=["exact", "semiclassical", "classical"],
        loc="upper right",
    )
    fig.savefig("scripts/thesis/quantum_effective_mass.mass.pdf")

    print(f"E_b / k_B T = {barrier_ratio:.4f}")  # noqa: T201
    print(f"classical m_eff / m = {classical:.4f}")  # noqa: T201
    for name, mass in ADSORBATE_MASSES.items():
        p = _get_dimensionless_parameters(base.with_mass(mass))
        print(  # noqa: T201
            f"{name}: hbar w_b / kT = {p[1]:.3f}, "
            f"exact = {get_exact_mass_ratio(*p):.4f}, "
            f"phase integral = {get_phase_integral_mass_ratio(*p):.4f}, "
            f"band averaged = {get_band_averaged_mass_ratio(*p):.4f}, "
            f"semiclassical = {get_semiclassical_mass_ratio(*p):.4f}, "
            f"rate theory = {get_rate_theory_mass_ratio(*p):.4f}, "
            f"series = {get_series_mass_ratio(*p):.4f}, "
            f"Z exact = {get_exact_partition_ratio(*p):.4f}, "
            f"Z Wigner = {get_wigner_partition_ratio(*p):.4f}, "
            f"band centre = {get_tunnelling_mass_ratio(*p):.4f}",
        )


class _CorrectionTerms(TypedDict):
    reflection: float
    tunnelling: float
    zero_point: float
    total: float


def get_correction_terms(barrier_ratio: float, omega_ratio: float) -> _CorrectionTerms:
    """Return the relative change in m_eff from each quantum effect.

    The flux integral of the semiclassical mass, relative to its classical
    value int_(E_b) e^(-beta E) / tau dE, is 1 + delta_r + delta_t, where

        delta_r = int_(E_b)^inf e^(-beta E) (w - 1) / tau dE,
        delta_t = int_0^(E_b) e^(-beta E) w / tau dE,

    in units of the classical flux, with w = 1 - sqrt(1 - P). Since
    m / m_eff = (m / m_eff)_cl (Z_cl / Z) (1 + delta_r + delta_t), and Z / Z_cl
    takes the Wigner-Kirkwood form, each term is the change in m_eff with the other
    two switched off, and the total includes all three.
    """

    def missing_transmission(u: float) -> float:
        # sqrt(1 - P), written so that it does not overflow far from E_b
        return float(
            np.sqrt(expit(-2 * np.pi * barrier_ratio * (u - 1) / omega_ratio)),
        )

    def flux(u: float, weight: float) -> float:
        return float(
            np.exp(-barrier_ratio * (u - 1)) * weight / _get_crossing_time_integral(u),
        )

    u_max = 1 + 80 / barrier_ratio
    classical = quad(lambda u: flux(u, 1), 1, u_max, limit=500)[0]
    reflection = (
        quad(lambda u: flux(u, -missing_transmission(u)), 1, u_max, limit=500)[0]
        / classical
    )
    tunnelling = (
        quad(lambda u: flux(u, 1 - missing_transmission(u)), 0, 1, limit=500)[0]
        / classical
    )
    partition = get_wigner_partition_ratio(barrier_ratio, omega_ratio)
    return {
        "reflection": float(1 / (1 + reflection) - 1),
        "tunnelling": float(1 / (1 + tunnelling) - 1),
        "zero_point": partition - 1,
        "total": float(partition / (1 + reflection + tunnelling) - 1),
    }


# Sequential map for the barrier energy, from light to dark blue
CAM_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "CAM_SEQUENTIAL_CMAP",
    [CAM_BLUE.warm, CAM_BLUE.dark],
)


# Both correction figures share the range of hbar omega_b / k_B T
CORRECTION_OMEGA_RATIOS = np.geomspace(0.05, 3, 60)


def _plot_correction_terms_against_omega(ax: Axes) -> None:
    """Plot each correction on the Cu(111) barrier against hbar omega_b / k_B T.

    At fixed barrier, lattice constant and temperature, hbar omega_b / k_B T
    falls as m^(-1 / 2), so heavier adsorbates lie further to the left.
    """
    barrier_ratio, _ = _get_dimensionless_parameters(SODIUM_COPPER_BRIDGE_SYSTEM_1D)
    omega_ratios = CORRECTION_OMEGA_RATIOS
    terms = [get_correction_terms(barrier_ratio, w) for w in omega_ratios]

    ax.axhline(0, color=CAM_SLATE_4, linewidth=0.8)
    (total_line,) = ax.plot(
        omega_ratios,
        [t["total"] for t in terms],
        color=CAM_CHERRY.dark,
    )
    (reflection_line,) = ax.plot(
        omega_ratios,
        [t["reflection"] for t in terms],
        color=CAM_BLUE.dark,
    )
    reflection_line.set_linestyle("-.")
    (tunnelling_line,) = ax.plot(
        omega_ratios,
        [t["tunnelling"] for t in terms],
        color=CAM_BLUE.warm,
    )
    tunnelling_line.set_linestyle(":")
    (zero_point_line,) = ax.plot(
        omega_ratios,
        [t["zero_point"] for t in terms],
        color=CAM_BLUE.base,
    )
    zero_point_line.set_linestyle("--")

    for name in ["H", "He", "Na"]:
        system = SODIUM_COPPER_BRIDGE_SYSTEM_1D.with_mass(ADSORBATE_MASSES[name])
        _, omega_ratio = _get_dimensionless_parameters(system)
        ax.axvline(omega_ratio, color=CAM_SLATE_2, linewidth=0.8, zorder=0)
        ax.annotate(
            name,
            (omega_ratio, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlim(omega_ratios[0], omega_ratios[-1])
    ax.set_ylim(-0.25, 0.6)
    ax.set_xlabel(r"$\hbar \omega_b / k_B T$")
    ax.set_ylabel("Correction")
    ax.legend(
        handles=[
            total_line,
            reflection_line,
            tunnelling_line,
            zero_point_line,
        ],
        labels=["total", "reflection", "tunnelling", "zero-point"],
        loc="upper left",
    )


def _plot_reflection_against_omega(ax: Axes) -> ScalarMappable:
    """Plot the reflection correction against hbar omega_b / k_B T.

    One line is drawn for each barrier energy E_b / k_B T. The lines
    collapse onto a single curve, since the reflection correction depends
    almost entirely on hbar omega_b / k_B T.
    """
    barrier_ratios = np.array([0.5, 1, 2, 3, 5, 8])
    omega_ratios = CORRECTION_OMEGA_RATIOS
    norm = Normalize(0, barrier_ratios[-1])
    ax.axhline(0, color=CAM_SLATE_4, linewidth=0.8)
    for barrier_ratio in barrier_ratios:
        reflection = [
            get_correction_terms(barrier_ratio, w)["reflection"] for w in omega_ratios
        ]
        ax.plot(
            omega_ratios,
            reflection,
            color=CAM_SEQUENTIAL_CMAP(norm(barrier_ratio)),
        )

    ax.set_xscale("log")
    ax.set_xlim(omega_ratios[0], omega_ratios[-1])
    ax.set_ylim(-0.25, 0.6)
    ax.set_xlabel(r"$\hbar \omega_b / k_B T$")
    ax.set_ylabel("Correction")
    return ScalarMappable(norm=norm, cmap=CAM_SEQUENTIAL_CMAP)


def _plot_correction_terms() -> None:
    fig, ax = _get_square_figure()
    _plot_correction_terms_against_omega(ax)
    fig.savefig("scripts/thesis/quantum_effective_mass.corrections.terms.pdf")

    fig, ax = _get_square_figure()
    mappable = _plot_reflection_against_omega(ax)
    colorbar_ax = ax.inset_axes((0.08, 0.78, 0.45, 0.05))
    colorbar = fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(r"$E_b / k_B T$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig("scripts/thesis/quantum_effective_mass.corrections.reflection.pdf")


def _get_square_figure() -> tuple[Figure, Axes]:
    width = get_thesis_fig_size()[0]
    return get_thesis_figure(fig_size=(width, width))


def _get_classical_momentum_squared(
    energy: np.ndarray[Any, np.dtype[np.float64]],
    barrier_ratio: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return p_cl^2 / (m k_B T) = 2 pi^2 e_b / I(u)^2 at E / k_B T = energy.

    Below the barrier, p_cl = m dx / tau(E), with tau half the period in
    the well.
    """
    crossing = _get_crossing_time_integral(energy / barrier_ratio)
    return 2 * np.pi**2 * barrier_ratio / crossing**2


def get_phase_integral_bands(
    barrier_ratio: float,
    omega_ratio: float,
    *,
    n_k: int = 512,
    n_energy: int = 400000,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Return E / k_B T and p_tr^2 / (m k_B T) from the band condition.

    Both are returned with shape (n_k, n_bands), sampled uniformly in k dx
    over [0, pi], including the edges of each band. Writing the total phase of
    band n as S = (n + 1 / 2) pi + delta, the band condition
    cos(k dx) = cos(S) / |t| becomes sin(delta) = (-1)^(n + 1) |t| cos(k dx).
    Since |t| depends on the energy, and so on delta, the condition is
    solved by fixed-point iteration. The momentum of each state then follows
    from _get_phase_integral_slope_squared.
    """
    u_max = 1 + MAX_KINETIC / barrier_ratio
    u = np.linspace(1e-8, u_max, n_energy)
    eps = barrier_ratio * (1 - u) / omega_ratio
    phase = _get_well_action(u, barrier_ratio, omega_ratio) + _get_parabolic_phase(eps)
    # dS / dE, with energies in units of k_B T
    phase_slope = np.gradient(phase, barrier_ratio * u)

    k_dx = np.linspace(0, np.pi, n_k)
    n_bands = int(phase[-1] / np.pi) - 1
    energies = np.empty((n_k, n_bands))
    p_squared = np.empty((n_k, n_bands))
    for n in range(n_bands):
        sign = (-1) ** (n + 1)
        delta = np.zeros_like(k_dx)
        for _ in range(200):
            energy = barrier_ratio * np.interp((n + 0.5) * np.pi + delta, phase, u)
            transmission = np.sqrt(
                expit((energy - barrier_ratio) * 2 * np.pi / omega_ratio),
            )
            delta = np.arcsin(sign * transmission * np.cos(k_dx))
        energies[:, n] = energy
        slope_squared = _get_phase_integral_slope_squared(
            (n + 0.5) * np.pi + delta,
            expit((energy - barrier_ratio) * 2 * np.pi / omega_ratio),
            np.interp(energy, barrier_ratio * u, phase_slope),
        )
        # In these units, p^2 / (m k_B T) = (2 pi^2 e_b / w^2) (dE / d(k dx))^2
        p_squared[:, n] = 2 * np.pi**2 * barrier_ratio / omega_ratio**2 * slope_squared
    return energies, p_squared


def _plot_state_momentum(name: str, x_range: tuple[float, float]) -> None:
    """Plot (tau(E) p_tr / m dx)^2 of each Bloch state, with the semiclassical bands.

    The exact Bloch states are compared with the phase-integral band
    condition, where tau(E) is the classical crossing time, which below the
    barrier is half the period in the well.
    """
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D.with_mass(ADSORBATE_MASSES[name])
    barrier_ratio, omega_ratio = _get_dimensionless_parameters(system)

    fig, ax = _get_square_figure()
    energies, p_squared = _get_bloch_states(
        barrier_ratio,
        omega_ratio,
        include_edges=True,
    )
    classical = _get_classical_momentum_squared(energies, barrier_ratio)
    for band in range(energies.shape[1]):
        ax.plot(
            energies[:, band] - barrier_ratio,
            p_squared[:, band] / classical[:, band],
            color=CAM_BLUE.warm if band % 2 == 0 else CAM_BLUE.base,
        )
    quantum_line = ax.plot([], [], color=CAM_BLUE.warm)[0]

    energies, p_squared = get_phase_integral_bands(barrier_ratio, omega_ratio)
    classical = _get_classical_momentum_squared(energies, barrier_ratio)
    for band in range(energies.shape[1]):
        (line,) = ax.plot(
            energies[:, band] - barrier_ratio,
            p_squared[:, band] / classical[:, band],
            color=CAM_CHERRY.dark,
        )
        line.set_linestyle("--")
    semiclassical_line = ax.plot([], [], color=CAM_CHERRY.dark)[0]
    semiclassical_line.set_linestyle("--")

    ax.set_xlim(*x_range)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"$(E - E_b) / k_B T$")
    ax.set_ylabel(r"$\tau(E)^2 p_\mathrm{tr}^2 / (m \Delta x)^2$")
    ax.legend(
        handles=[quantum_line, semiclassical_line],
        labels=["quantum", "phase integral"],
        loc="upper left",
    )
    fig.savefig(f"scripts/thesis/quantum_effective_mass.momentum.{name}.pdf")


class _MassRatioMap(TypedDict):
    barrier_ratios: np.ndarray[Any, np.dtype[np.float64]]
    omega_ratios: np.ndarray[Any, np.dtype[np.float64]]
    deviation: np.ndarray[Any, np.dtype[np.float64]]


def _get_mass_ratio_deviation(
    barrier_ratio: float,
    omega_ratios: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return m_eff / m_eff^cl - 1 at one barrier energy, for each omega_ratio."""
    classical = get_classical_mass_ratio(barrier_ratio)
    return np.array(
        [get_exact_mass_ratio(barrier_ratio, w) / classical - 1 for w in omega_ratios],
    )


@cached(Path("scripts/thesis/quantum_effective_mass.map.npz"))
def get_mass_ratio_map() -> _MassRatioMap:
    """Return m_eff / m_eff^cl - 1 against E_b / k_B T and hbar omega_b / k_B T.

    Each barrier energy is independent, so the columns are computed in
    parallel.
    """
    barrier_ratios = np.linspace(1, 5, 100)
    omega_ratios = np.geomspace(0.1, 3, 100)
    with ProcessPoolExecutor() as executor:
        columns = list(
            executor.map(
                _get_mass_ratio_deviation,
                barrier_ratios,
                [omega_ratios] * barrier_ratios.size,
            ),
        )
    return {
        "barrier_ratios": barrier_ratios,
        "omega_ratios": omega_ratios,
        "deviation": np.stack(columns, axis=1),
    }


# Diverging map from blue (m_eff lowered) through the axes background to cherry
CAM_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "CAM_DIVERGING_CMAP",
    [
        CAM_BLUE.warm,
        CAM_BLUE.base,
        CAM_BLUE.light,
        CAM_SLATE_1,
        CAM_CHERRY.light,
        CAM_CHERRY.warm,
        CAM_CHERRY.base,
    ],
)


def _plot_mass_ratio_map() -> None:
    data = get_mass_ratio_map()
    barrier_ratios = data["barrier_ratios"]
    omega_ratios = data["omega_ratios"]
    deviation = data["deviation"]

    fig, ax = get_fancy_figure()
    # The scale is set by the largest rise in m_eff, and the stronger fall at
    # the lightest masses saturates
    limit = 1.2 * float(np.max(deviation))
    mesh = ax.pcolormesh(
        barrier_ratios,
        omega_ratios,
        deviation,
        shading="nearest",
        cmap=CAM_DIVERGING_CMAP,
        norm=TwoSlopeNorm(0, -limit, limit),
        rasterized=True,
    )
    for name in ["H", "He", "Na"]:
        system = SODIUM_COPPER_BRIDGE_SYSTEM_1D.with_mass(ADSORBATE_MASSES[name])
        barrier_ratio, omega_ratio = _get_dimensionless_parameters(system)
        # At fixed mass, lattice constant and temperature, omega_b grows as
        # sqrt(E_b), so a change of barrier moves the adsorbate along a curve
        path = omega_ratio * np.sqrt(barrier_ratios / barrier_ratio)
        (path_line,) = ax.plot(
            barrier_ratios,
            path,
            color=CAM_SLATE_4,
            linewidth=1,
        )
        path_line.set_linestyle(":")
        ax.annotate(
            name,
            (barrier_ratio, omega_ratio),
            xytext=(6, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
        )

    ax.set_yscale("log")
    ax.set_ylim(omega_ratios[0], omega_ratios[-1])
    # Plain labels on a few round values, rather than every power of ten
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 3]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(barrier_ratios[0], barrier_ratios[-1])
    ax.set_xlabel(r"$E_b / k_B T$")
    ax.set_ylabel(r"$\hbar \omega_b / k_B T$")
    colorbar = fig.colorbar(mesh, ax=ax, extend="min")
    colorbar.set_label("Correction")
    fig.savefig("scripts/thesis/quantum_effective_mass.map.pdf")


if __name__ == "__main__":
    _plot_mass_ratio_against_mass()
    _plot_state_momentum("Na", (-0.5, 2))
    _plot_state_momentum("He", (-0.5, 2))
    _plot_mass_ratio_map()
    _plot_correction_terms()

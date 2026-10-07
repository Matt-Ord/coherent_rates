from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quad
from scipy.special import ellipk

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod
from coherent_rates.isf import get_ordered_momentum, get_weak_boltzmann_isf
from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, System
from coherent_rates.util import CAM_BLUE, CAM_CHERRY, cached, get_fancy_figure


def _get_single_exact_effective_mass_ratio(
    dimensionless_barrier_energy: float,
) -> float:
    u0 = dimensionless_barrier_energy

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    def integrand_partition(epsilon: float) -> float:
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    partition_integral, _ = quad(integrand_partition, 1, np.inf)

    return 2 * partition_integral / (denominator_integral * u0 * np.pi**2)


def _get_exact_effective_mass_ratio(
    barrier_energy_ratio_fine: np.ndarray,
) -> np.ndarray[tuple[int], np.dtype[np.floating[Any]]]:

    return np.array(
        [_get_single_exact_effective_mass_ratio(m) for m in barrier_energy_ratio_fine],
    )


def _get_fixed_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
    *,
    target_occupation: float,
) -> tuple[float, float]:
    momentum, energy_per_state = get_ordered_momentum(system, config)
    prefactor = 1 / (config.temperature * Boltzmann * system.mass)
    momentum *= prefactor

    sort_indices = np.argsort(energy_per_state)[::-1]
    momentum = momentum[sort_indices]
    energy_per_state = energy_per_state[sort_indices]

    thermal_factors = np.exp(-energy_per_state / (Boltzmann * config.temperature))
    thermal_factors /= np.sum(thermal_factors)

    cumsum_thermal_factors = np.cumsum(thermal_factors)
    cumsum_inverse_mass = np.cumsum(thermal_factors * momentum / system.mass)

    # Find the state cutoff index closest to the target occupation
    idx = int(np.argmin(np.abs(cumsum_thermal_factors - target_occupation)))

    actual_occupation = cumsum_thermal_factors[idx]
    inverse_mass = cumsum_inverse_mass[idx] / actual_occupation
    effective_mass = 1 / inverse_mass

    return float(actual_occupation), float(effective_mass / system.mass)


def _get_classical_above_barrier_occupation(
    system: System,
    config: PeriodicSystemConfig,
) -> float:

    u0 = system.barrier_energy / (config.temperature * Boltzmann)

    def integrand_below(epsilon: float) -> float:
        # Trapped states (0 <= E < U0)
        return ellipk(epsilon) * np.exp(-u0 * epsilon)

    def integrand_above(epsilon: float) -> float:
        # Running states (E >= U0)
        return 1 / np.sqrt(epsilon) * ellipk(1 / epsilon) * np.exp(-u0 * epsilon)

    z_below, _ = quad(integrand_below, 0, 1)
    z_above, _ = quad(integrand_above, 1, np.inf)

    return z_above / (z_below + z_above)


def _get_classical_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[float, float]:
    return _get_fixed_threshold_mass_ratio(
        system,
        config,
        target_occupation=_get_classical_above_barrier_occupation(system, config),
    )


def _get_long_time_threshold_mass_ratio(
    system: System,
    config: PeriodicSystemConfig,
    *,
    t_factor: float = 8,
) -> tuple[float, float]:
    times = GaussianMethod(measure="abs", t_factor=t_factor).get_fit_times(
        system=system,
        config=config,
    )
    target_occupation = (
        1
        - np.abs(
            get_weak_boltzmann_isf.call_uncached(system, config, times)["data"],
        )[-1]
    )
    return _get_fixed_threshold_mass_ratio(
        system,
        config,
        target_occupation=target_occupation,
    )


@cached(Path("scripts/perturbation/classical_effective_mass.cache.npz"))
def _get_quamtum_effective_mass_ratio(
    barrier_energy_ratio: np.ndarray[tuple[int], np.dtype[np.floating[Any]]],
) -> np.ndarray[tuple[int], np.dtype[np.floating[Any]]]:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    config = PeriodicSystemConfig(
        (50,),
        (150,),
        direction=(1,),
        truncation=100,
        temperature=155,
    )
    return np.array(
        [
            _get_classical_threshold_mass_ratio(
                system.with_barrier_energy(
                    float(point) * config.temperature * Boltzmann,
                ),
                config,
            )[1]
            for point in barrier_energy_ratio
        ],
    )


def _load_classical_simulated_ratios() -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.load(
            "scripts/perturbation/effective_mass_trend_simulated_ratios.npz",
        )
        return data["barrier_energy_ratio"], data["simulated_effective_mass_ratio"]

    except FileNotFoundError:
        return np.array([]), np.array([])


def _plot_effective_mass_ratio() -> None:

    barrier_energy_ratio = np.linspace(0, 3, 1000)

    fig, ax = get_fancy_figure()

    (exact_line,) = ax.plot(
        barrier_energy_ratio,
        _get_exact_effective_mass_ratio(barrier_energy_ratio),
    )
    exact_line.set_label("Analytical")
    exact_line.set_color(CAM_BLUE.warm)

    points = np.linspace(0.1, 3, 30, endpoint=True)
    (quantum_line,) = ax.plot(points, _get_quamtum_effective_mass_ratio(points))
    quantum_line.set_label("Quantum")
    quantum_line.set_color(CAM_CHERRY.dark)
    quantum_line.set_marker("x")
    quantum_line.set_linestyle("")

    points, simulated_effective_mass_ratio = _load_classical_simulated_ratios()

    (classical_line,) = ax.plot(points, simulated_effective_mass_ratio)
    classical_line.set_label("Classical")
    classical_line.set_color(CAM_BLUE.dark)
    classical_line.set_marker("x")
    classical_line.set_linestyle("")

    ax.legend(handles=[exact_line, quantum_line, classical_line])

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$E_b / k_B T$")
    ax.set_ylabel(r"$m_{\text{eff}} / m$")
    fig.savefig("scripts/perturbation/classical_effective_mass.pdf")


if __name__ == "__main__":
    _plot_effective_mass_ratio()

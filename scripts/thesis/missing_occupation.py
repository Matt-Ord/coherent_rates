import numpy as np
from scipy.constants import Boltzmann, hbar
from scipy.special import logsumexp

from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D, PeriodicSystem1d
from coherent_rates.util import CAM_BLUE, get_paper_figure


def _plot_missing_occupation(system: PeriodicSystem1d, temperature: float) -> None:
    band_idx = np.arange(200)

    band_k = 2 * np.ceil(band_idx / 2) * np.pi / system.lattice_constant
    band_energy = (hbar * band_k) ** 2 / (2 * system.mass)

    # Calculate the log of the unnormalized weights first
    # This prevents values from hitting 0.0 prematurely
    log_weights = -(band_energy / (Boltzmann * temperature))

    # Log-Sum-Exp trick for stable normalization
    log_occupation = log_weights - logsumexp(log_weights)

    # To find 1 - cumsum(occ), we sum the tail: sum_{i=n+1}^{N} occ_i
    # We do this in the original space only for the suffix to maintain precision
    occupation = np.exp(log_occupation)

    # Calculate missing occupation by summing from the end backwards
    # This is much more stable than 1 - cumsum(forward)
    missing_occupation = np.zeros_like(occupation)
    for i in range(len(occupation)):
        missing_occupation[i] = np.sum(occupation[i + 1 :])

    fig, ax = get_paper_figure()
    # Use ax.semilogy or set_yscale("log")
    (line,) = ax.plot(band_idx[:26], missing_occupation[:26])
    line.set_color(CAM_BLUE.warm)
    ax.set_yscale("log")
    ax.set_xlim(0, 25)

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Missing Occupation")
    fig.savefig("scripts/thesis/missing_occupation.pdf")


def _get_thermal_distance(
    system: PeriodicSystem1d,
    temperature: float,
    time: float,
) -> None:

    thermal_velocity = np.sqrt(Boltzmann * temperature / system.mass)
    thermal_distance = thermal_velocity * time

    print(thermal_distance / system.lattice_constant)  # noqa: T201


if __name__ == "__main__":
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    _plot_missing_occupation(system, temperature=155)
    _get_thermal_distance(system, temperature=155, time=1e-12)

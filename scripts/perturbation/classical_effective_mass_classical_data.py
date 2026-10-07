# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "scipy",
#     "classical-diffusion @ git+https://github.com/Matt-Ord/classical_diffusion.git@374ea0b62492dbcdd5abc1c16cf0e05ab8018cbc",
#     "jax",
#     "tqdm"
# ]
# ///


# uv run --no-project ./scripts/perturbation/classical_effective_mass_classical_data.py
from pathlib import Path

# cspell: disable-next-line spelling
import jax.random as jrandom  # type: ignore[import]
import numpy as np
from classical_diffusion.langevin import (  # type: ignore[import]
    SODIUM_COPPER_SYSTEM_1D,
    PeriodicSystem1D,
    get_effective_mass,
    solve_ensemble_ballistic,
)
from classical_diffusion.simulation import TimeSpan  # type: ignore[import]
from classical_diffusion.util import (  # type: ignore[import]
    cache_base_path,
    disabled_timing,
)
from tqdm import tqdm  # type: ignore[import]


def _with_barrier_energy(
    system: PeriodicSystem1D,
    barrier_energy: float,
) -> PeriodicSystem1D:
    """Return a copy of the system with a new barrier energy."""
    return PeriodicSystem1D(
        gamma=system.gamma,
        temperature=system.temperature,
        m=system.m,
        delta_x=system.delta_x,
        barrier_energy=barrier_energy,
        units=system.units,
        n_dim=system.n_dim,
    )


def _get_simulated_effective_mass(
    system: PeriodicSystem1D,
    barrier_energy_ratio: np.ndarray,
    n_samples: np.ndarray,
) -> np.ndarray:
    # cspell: disable-next-line spelling
    keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy_ratio.size)
    out = np.zeros_like(barrier_energy_ratio)

    barrier_energy = barrier_energy_ratio * system.kbt

    with disabled_timing():
        for idx, _ in enumerate(
            # cspell: disable-next-line spelling
            tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size),
        ):
            result = solve_ensemble_ballistic.call_uncached(
                _with_barrier_energy(system, barrier_energy=barrier_energy[idx]),
                TimeSpan(t_end=8 / system.gamma, n_steps=1000),
                energy_range=(barrier_energy[idx], np.inf),
                n_samples=n_samples[idx],
                _key=keys[idx],
            )

            mass = get_effective_mass(result, filter_timescale=1 / system.gamma)
            out[idx] = mass.item() / system.m

        return out


def _generate_classical_simulated_ratios() -> None:
    points = np.linspace(0.1, 3, 30, endpoint=True)
    out = _get_simulated_effective_mass(
        system=SODIUM_COPPER_SYSTEM_1D,
        barrier_energy_ratio=points,
        n_samples=10000 * np.ones_like(points, dtype=int),
    )
    # cspell: disable-next-line spelling
    np.savez(
        "scripts/perturbation/effective_mass_trend_simulated_ratios.npz",
        barrier_energy_ratio=points,
        simulated_effective_mass_ratio=out,
    )


if __name__ == "__main__":
    with cache_base_path(Path("scripts/perturbation")):
        _generate_classical_simulated_ratios()

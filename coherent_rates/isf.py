from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from scipy.constants import Boltzmann, electron_volt, hbar  # type: ignore library type
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedBasis,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.momentum_basis_like import MomentumBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    apply_operator_to_state,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation_diagonal,
)
from surface_potential_analysis.state_vector.eigenstate_list import (
    ValueList,
)
from surface_potential_analysis.state_vector.plot import (
    get_periodic_x_operator,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)
from surface_potential_analysis.util.decorators import cached, timed
from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

from coherent_rates.config import (
    PeriodicSystemConfig,
    SimpleInstrumentFunction,
)
from coherent_rates.fit import (
    FitMethod,
    GaussianMethod,
    GaussianPlusExponentialMethod,
    get_free_particle_rate,
    get_scattered_momentum,
)
from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    apply_scattering_operator_to_state,
    apply_scattering_operator_to_states,
    get_instrument_biased_periodic_x_from_hamiltonian,
    get_k_operator_sparse,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.state import (
    LocalizationStrategy,
    _get_coherent_state_for_basis,
    get_local_boltzmann_state_from_hamiltonian,
    get_random_boltzmann_state_from_hamiltonian,
    get_random_coherent_coordinates,
)
from coherent_rates.system import (
    System,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisLike,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import (
        StatisticalValueList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])

_BV0 = TypeVar("_BV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])

_ESB0 = TypeVar("_ESB0", bound=BlochBasis[Any])
_ESB1 = TypeVar("_ESB1", bound=BlochBasis[Any])


def _get_isf_pair_states_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> tuple[StateVectorList[_BT0, _ESB0], StateVectorList[_BT0, _B0]]:
    state_evolved = solve_schrodinger_equation_diagonal(
        initial_state,
        times,
        hamiltonian,
    )

    state_evolved_scattered = apply_scattering_operator_to_states(
        operator,
        state_evolved,
    )

    state_scattered = apply_scattering_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation_diagonal(
        state_scattered,
        times,
        hamiltonian,
    )

    return (state_evolved_scattered, state_scattered_evolved)


def get_isf_pair_states(
    system: System,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B1],
    times: _BT0,
) -> tuple[
    StateVectorList[_BT0, StackedBasisWithVolumeLike[Any, Any, Any]],
    StateVectorList[_BT0, StackedBasisWithVolumeLike[Any, Any, Any]],
]:
    hamiltonian = get_hamiltonian(system, config)
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    return _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )


@timed
def _get_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[_BT0]:
    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )
    return calculate_inner_products_elementwise(
        state_scattered_evolved,
        state_evolved_scattered,
    )


def get_isf(
    system: System,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[_BT0]:
    hamiltonian = get_hamiltonian(system, config)
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    return _get_isf_from_hamiltonian(hamiltonian, operator, initial_state, times)


def get_analytical_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
) -> ValueList[_BT0]:
    k = get_scattered_momentum(system, config, [config.direction])[0]

    # ISF(k, t) = exp(-(kTt^2 - i hbar t)* energy)
    energy = k**2 / (2 * system.mass)
    boltzmann_energy = Boltzmann * config.temperature
    data = np.exp(
        (-1 * boltzmann_energy * times.times**2 + 1j * hbar * times.times) * energy,
    )
    return {"data": data, "basis": times}


def _get_states_per_band(
    states: StateVectorList[
        _B1,
        BlochBasis[_B0],
    ],
) -> StateVectorList[
    TupleBasis[_B0, _B1],
    TupleBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]],
]:
    basis = states["basis"][1].wavefunctions["basis"][0]

    data = states["data"].reshape(-1, *basis.shape).swapaxes(0, 1)
    return {
        "basis": TupleBasis(TupleBasis(basis[0], states["basis"][0]), basis[1]),
        "data": data.ravel(),
    }


def _get_band_resolved_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB1],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[TupleBasis[Any, _BT0]]:
    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )
    per_band_scattered_evolved = _get_states_per_band(
        state_scattered_evolved,
    )
    per_band_evolved_scattered = _get_states_per_band(
        state_evolved_scattered,
    )

    return calculate_inner_products_elementwise(
        per_band_scattered_evolved,
        per_band_evolved_scattered,
    )


def get_band_resolved_boltzmann_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[TupleBasisLike[BasisLike[Any, Any], _BT0]]:
    hamiltonian = get_hamiltonian(system, config)
    bands = hamiltonian["basis"][0].wavefunctions["basis"][0][0]
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    isf_data = np.zeros((n_repeats, bands.n * times.n), dtype=np.complex128)

    def _calculate_isf(i: int) -> None:
        state = get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_band_resolved_isf_from_hamiltonian(
            hamiltonian,
            operator,
            state,
            times,
        )
        isf_data[i, :] = data["data"]

    for i in range(n_repeats):
        _calculate_isf(i)

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": TupleBasis(bands, times),
        "standard_deviation": sd,
    }


def _get_coherent_isf_from_hamiltonian(  # noqa: PLR0913
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 1,
    sigma_0: tuple[float, ...] | None = None,
) -> StatisticalValueList[_BT0]:
    if sigma_0 is None:
        sigma_0 = tuple(system.lattice_constant / 10 for _ in config.resolution)

    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    isf_data = np.zeros((2 * n_repeats, times.n), dtype=np.complex128)

    def _calculate_isf(i: int) -> None:
        x0, k0 = get_random_coherent_coordinates(system, config)

        state = _get_coherent_state_for_basis(hamiltonian["basis"][1], x0, k0, sigma_0)
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i, :] = data["data"]

        k0 = tuple([-j for j in k0])
        state = _get_coherent_state_for_basis(hamiltonian["basis"][1], x0, k0, sigma_0)
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i + n_repeats, :] = data["data"]

    for i in range(n_repeats):
        _calculate_isf(i)

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {"data": mean, "basis": times, "standard_deviation": sd}


def _get_coherent_isf_data_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    *,
    n_repeats: int = 10,
) -> Path:
    return Path(f"data/{hash((system, config))}.{hash(times)}.{n_repeats}.coherent.isf")


@cached(_get_coherent_isf_data_path)
def get_coherent_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 10,
    sigma_0: tuple[float, ...] | None = None,
) -> StatisticalValueList[_BT0]:
    """Get the isf with n_repeats coherent wavepackets.

    Parameters
    ----------
    system : PeriodicSystem
    config : PeriodicSystemConfig
    times : _BT0
    direction : tuple[int, ...] | None, optional
        direction, by default None
    n_repeats : int, optional
        n_repeats, by default 10
    sigma_0 : float | None, optional
        sigma_0, by default None

    Returns
    -------
    StatisticalValueList[_BT0]

    """
    hamiltonian = get_hamiltonian(system, config)

    return _get_coherent_isf_from_hamiltonian(
        hamiltonian,
        system,
        config,
        times,
        n_repeats=n_repeats,
        sigma_0=sigma_0,
    )


def _get_coherent_rate_against_momentum_data_path(  # noqa: PLR0913
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
    sigma_0: tuple[float, ...] | None = None,
) -> Path:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions
    sigma_0 = (-1,) if sigma_0 is None else sigma_0
    return Path(
        f"data/{hash((system, config))}.{hash(fit_method)}"
        f".{hash((directions[0], directions[-1], len(directions)))}"
        f".{hash((n_repeats, sigma_0))}"
        ".coherent.rates",
    )


def get_coherent_rate(
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    n_repeats: int = 10,
) -> float:
    times = fit_method.get_fit_times(system=system, config=config)
    isf = get_coherent_isf(system, config, times, n_repeats=n_repeats)
    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


def _get_coherent_rate_from_hamiltonian(  # noqa: PLR0913
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    n_repeats: int = 10,
    sigma_0: tuple[float, ...] | None = None,
) -> float:
    times = fit_method.get_fit_times(system=system, config=config)
    isf = _get_coherent_isf_from_hamiltonian(
        hamiltonian,
        system,
        config,
        times,
        n_repeats=n_repeats,
        sigma_0=sigma_0,
    )

    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


@cached(_get_coherent_rate_against_momentum_data_path)
def get_coherent_rate_against_momentum_data(  # noqa: PLR0913
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
    sigma_0: tuple[float, ...] | None = None,
) -> ValueList[MomentumBasis]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions

    rates = np.zeros(len(directions), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(directions):
        rates[i] = _get_coherent_rate_from_hamiltonian(
            hamiltonian,
            system=system,
            config=config.with_direction(direction),
            fit_method=fit_method,
            n_repeats=n_repeats,
            sigma_0=sigma_0,
        )

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {"data": rates.ravel(), "basis": basis}


def _get_default_directions(config: PeriodicSystemConfig) -> list[tuple[int, ...]]:
    return list(
        zip(
            *tuple(
                cast("list[int]", (s * np.arange(1, r)).tolist())
                for (s, r) in zip(config.shape, config.resolution, strict=True)
            ),
            strict=False,
        ),
    )


def _get_boltzmann_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[_BT0]:
    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    def _calculate_isf(i: int) -> None:
        state = get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i, :] = data["data"]

    for i in range(n_repeats):
        _calculate_isf(i)

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": times,
        "standard_deviation": sd,
    }


def _get_boltzmann_isf_data_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    *,
    n_repeats: int = 10,
) -> Path:
    return Path(
        f"data/{hash((system, config))}.{hash(times)}.{n_repeats}.boltzmann.isf",
    )


@cached(_get_boltzmann_isf_data_path)
@timed
def get_boltzmann_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 10,
) -> StatisticalValueList[_BT0]:
    hamiltonian = get_hamiltonian(system, config)
    return _get_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config,
        times,
        n_repeats=n_repeats,
    )


def _get_k_diagonal(
    operator: SparseScatteringOperator[_ESB0, _ESB0],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    band_basis = operator["basis"][0].wavefunctions["basis"][0][0]
    basis = operator["basis"][0].wavefunctions["basis"][0][1]
    n_bands = band_basis.n
    n_k = basis.n
    if not all(x == 0 for x in operator["direction"]):
        return np.zeros((n_bands, n_k), dtype=np.complex128)

    data = operator["data"].reshape(n_bands, n_bands, n_k)
    return np.einsum("aaj->aj", data)


def _get_k_scatter(
    operator: SparseScatteringOperator[_ESB0, _ESB0],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get |<n|delta k hat(p) / m |m>|^2.

    returns an array of shape (n_bands, n_bands, n_k)
    where the second index is the band index of m

    """
    band_basis = operator["basis"][0].wavefunctions["basis"][0][0]
    basis = operator["basis"][0].wavefunctions["basis"][0][1]
    n_bands = band_basis.n
    n_k = basis.n
    if not all(x == 0 for x in operator["direction"]):
        return np.zeros((n_bands, n_k), dtype=np.float64)

    data = operator["data"].reshape(n_bands, n_bands, n_k)
    square_norm = np.abs(data) ** 2
    n_idx = np.arange(n_bands)
    square_norm[n_idx, n_idx, :] = 0.0
    return square_norm.astype(np.float64)


def _get_time_factor_first_order(
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
    friction: float = 0,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    gamma_t = friction * times
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (1 - np.exp(-gamma_t)) / friction
    is_almost_zero = np.isclose(gamma_t, 0)
    out[is_almost_zero] = times[is_almost_zero] * (1 - (gamma_t[is_almost_zero] / 2))
    return out


def _get_scatter_omega(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    *,
    band_idx: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get |<n|delta k hat(p) / m |m>|^2.

    returns an array of shape (n_bands, n_k)
    where the n_bands index is the band m

    """
    energies = hamiltonian["data"]
    n_bands, n_k = hamiltonian["basis"][0].wavefunctions["basis"][0].shape
    energies = energies.reshape(n_bands, n_k) / hbar
    return np.real(energies[band_idx, None, :] - energies[:, :])


def _get_time_factor_second_order(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
    friction: float = 0,
    *,
    band_idx: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    scatter_omega = _get_scatter_omega(hamiltonian, band_idx=band_idx)
    scatter_omega = scatter_omega.reshape((*scatter_omega.shape, 1))
    times = times.reshape(1, 1, -1)
    omega_t = scatter_omega * times

    if np.isclose(friction, 0):
        return 0.5 * times**2 * np.sinc(omega_t / (2 * np.pi)) ** 2

    gamma_t = friction * times
    numerator = 1 + np.exp(-2 * gamma_t) - (2 * np.exp(-gamma_t) * np.cos(omega_t))
    denominator = 2 * (scatter_omega**2 + friction**2)
    return numerator / denominator


def _get_decay_per_state(  # noqa: PLR0913
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    direction: tuple[int, ...],
    times: np.ndarray[tuple[int], np.dtype[np.float64]],
    mass: float,
    friction: float = 0,
    *,
    second_order: bool = False,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:

    state_basis = hamiltonian["basis"][0]
    dk = BasisUtil(state_basis).fundamental_dk_stacked
    direction_k = np.einsum("i,ij->j", direction, dk)
    scatter_operator = get_k_operator_sparse(
        state_basis,
        direction_k * (hbar / mass),
    )
    recoil = np.sum(np.square(direction_k)) * hbar / (2 * mass)

    diagonal_k = (recoil - _get_k_diagonal(scatter_operator)).reshape(-1, 1)
    diagonal_time_factor = (
        _get_time_factor_first_order(times, friction) if second_order else times
    )
    diagonal_phase = (1j) * (diagonal_time_factor)[np.newaxis, :]
    out = diagonal_phase * diagonal_k

    if second_order:
        n_bands = state_basis.wavefunctions["basis"][0][0].n
        out = out.reshape((n_bands, -1, times.size))
        scatter_k = _get_k_scatter(scatter_operator)
        scatter_k = scatter_k.reshape((*scatter_k.shape, 1))
        for band_idx in range(n_bands):
            time_factor = _get_time_factor_second_order(
                hamiltonian,
                times,
                friction,
                band_idx=band_idx,
            )
            out[band_idx, :] -= np.einsum(
                "bkt->kt",  # cspell: disable-line
                time_factor * scatter_k[band_idx],
            ).reshape(-1, times.size)
        out = out.reshape(-1, times.size)
    return out


def _get_weak_boltzmann_isf_data_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    friction: float = 0,
    *,
    second_order: bool = False,
) -> Path:
    prefix = f"{hash((system, config))}.{hash(times)}.{second_order}.{friction}"
    return Path(f"data/{prefix}.weak_boltzmann.isf")


def _get_weak_boltzmann_isf_from_hamiltonian(  # noqa: PLR0913
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    direction: tuple[int, ...],
    temperature: float,
    times: _BT0,
    mass: float,
    friction: float = 0,
    *,
    second_order: bool = False,
) -> ValueList[_BT0]:
    isf_per_state = np.exp(
        _get_decay_per_state(
            hamiltonian,
            direction,
            times.times,
            second_order=second_order,
            friction=friction,
            mass=mass,
        ),
    )

    kb_t = Boltzmann * temperature
    occupations = np.exp(-hamiltonian["data"] / kb_t)
    occupations /= np.sum(occupations)

    isf = np.einsum("nj,n->j", isf_per_state, occupations)
    return {"data": isf, "basis": times}


@cached(_get_weak_boltzmann_isf_data_path)
@timed
def get_weak_boltzmann_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    friction: float = 0,
    *,
    second_order: bool = False,
) -> ValueList[_BT0]:
    hamiltonian = get_hamiltonian(system, config)

    return _get_weak_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config.direction,
        config.temperature,
        times,
        second_order=second_order,
        mass=system.mass,
        friction=friction,
    )


def _get_local_boltzmann_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 1,
    strategy: LocalizationStrategy | None = None,
) -> StatisticalValueList[_BT0]:
    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        direction=config.direction,
        instrument_function=config.instrument_function,
    )

    def _calculate_isf(i: int) -> None:
        state = get_local_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
            strategy=strategy,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i, :] = data["data"]

    for i in range(n_repeats):
        _calculate_isf(i)

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": times,
        "standard_deviation": sd,
    }


def _get_local_boltzmann_isf_data_path(
    system: System,
    config: PeriodicSystemConfig,
    times: Any,  # noqa: ANN401
    *,
    n_repeats: int = 10,
    strategy: LocalizationStrategy | None = None,
) -> Path:
    strategy_hash = hash(strategy) if strategy is not None else -1
    return Path(
        f"data/{hash((system, config))}.{hash(times)}.{n_repeats}.{strategy_hash}"
        ".local.boltzmann.isf",
    )


@cached(_get_local_boltzmann_isf_data_path)
@timed
def get_local_boltzmann_isf(
    system: System,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    n_repeats: int = 10,
    strategy: LocalizationStrategy | None = None,
) -> StatisticalValueList[_BT0]:
    hamiltonian = get_hamiltonian(system, config)
    return _get_local_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config,
        times,
        n_repeats=n_repeats,
        strategy=strategy,
    )


def _get_boltzmann_rate_against_momentum_data_path(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> Path:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions
    return Path(
        f"data/{hash((system, config))}.{hash(fit_method)}"
        f".{hash((directions[0], directions[-1], len(directions), n_repeats))}.rates",
    )


def get_boltzmann_rate(
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    n_repeats: int = 10,
) -> float:
    times = fit_method.get_fit_times(
        system=system,
        config=config,
    )

    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )

    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


def _get_boltzmann_rate_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    n_repeats: int = 10,
) -> float:
    times = fit_method.get_fit_times(system=system, config=config)

    isf = _get_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config,
        times,
        n_repeats=n_repeats,
    )

    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


@cached(_get_boltzmann_rate_against_momentum_data_path)
def get_boltzmann_rate_against_momentum_data(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> ValueList[MomentumBasis]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions

    rates = np.zeros(len(directions), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(directions):
        rates[i] = _get_boltzmann_rate_from_hamiltonian(
            hamiltonian,
            system=system,
            config=config.with_direction(direction),
            fit_method=fit_method,
            n_repeats=n_repeats,
        )

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {"data": rates.ravel(), "basis": basis}


@timed
def _get_weak_boltzmann_rate_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    second_order: bool = False,
) -> float:
    times = fit_method.get_fit_times(system=system, config=config)

    isf = _get_weak_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config.direction,
        config.temperature,
        times,
        second_order=second_order,
        mass=system.mass,
    )

    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


def get_weak_boltzmann_rate(
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    second_order: bool = False,
) -> float:
    hamiltonian = get_hamiltonian(system, config)
    return _get_weak_boltzmann_rate_from_hamiltonian(
        hamiltonian,
        system,
        config,
        fit_method,
        second_order=second_order,
    )


def _get_weak_boltzmann_rate_against_momentum_data_path(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> Path:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions
    return Path(
        f"data/{hash((system, config))}.{hash(fit_method)}"
        f".{hash((directions[0], directions[-1], len(directions)))}.weak.rates",
    )


@cached(_get_weak_boltzmann_rate_against_momentum_data_path)
def get_weak_boltzmann_rate_against_momentum_data(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> ValueList[MomentumBasis]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions

    rates = np.zeros(len(directions), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(directions):
        rates[i] = _get_weak_boltzmann_rate_from_hamiltonian(
            hamiltonian,
            system=system,
            config=config.with_direction(direction),
            fit_method=fit_method,
        )

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {"data": rates.ravel(), "basis": basis}


def _get_local_boltzmann_rate_from_hamiltonian(  # noqa: PLR0913
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    system: System,
    config: PeriodicSystemConfig,
    fit_method: FitMethod[Any],
    *,
    n_repeats: int = 10,
    strategy: LocalizationStrategy | None = None,
) -> float:
    times = fit_method.get_fit_times(
        system=system,
        config=config,
    )

    isf = _get_local_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config,
        times,
        n_repeats=n_repeats,
        strategy=strategy,
    )

    return fit_method.get_rate_from_isf(
        isf,
        system=system,
        config=config,
    )


def _get_local_boltzmann_rate_against_momentum_data_path(  # noqa: PLR0913
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
    strategy: LocalizationStrategy | None = None,
) -> Path:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions
    strategy_hash = hash(strategy) if strategy is not None else -1
    return Path(
        f"data/{hash((system, config))}.{hash(fit_method)}"
        f".{hash((directions[0], directions[-1], len(directions), n_repeats))}"
        f".{strategy_hash}"
        ".local.rates",
    )


@cached(_get_local_boltzmann_rate_against_momentum_data_path)
def get_local_boltzmann_rate_against_momentum_data(  # noqa: PLR0913
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
    strategy: LocalizationStrategy | None = None,
) -> ValueList[MomentumBasis]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    directions = _get_default_directions(config) if directions is None else directions

    rates = np.zeros(len(directions), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(directions):
        rates[i] = _get_local_boltzmann_rate_from_hamiltonian(
            hamiltonian,
            system=system,
            config=config.with_direction(direction),
            fit_method=fit_method,
            n_repeats=n_repeats,
            strategy=strategy,
        )

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {"data": rates.ravel(), "basis": basis}


@timed
def get_boltzmann_rate_against_condition_and_momentum_data(
    conditions: list[tuple[System, PeriodicSystemConfig]],
    directions: list[tuple[int, ...]] | None = None,
    *,
    fit_method: FitMethod[Any] | None = None,
) -> ValueList[TupleBasis[FundamentalBasis[int], MomentumBasis]]:
    fit_method = (
        GaussianPlusExponentialMethod("Gaussian") if fit_method is None else fit_method
    )
    directions = (
        _get_default_directions(conditions[0][1]) if directions is None else directions
    )

    data = np.zeros(
        (len(conditions), len(directions)),
        dtype=np.complex128,
    )

    rate_data = None
    for j, (system, config) in enumerate(conditions):
        rate_data = get_boltzmann_rate_against_momentum_data(
            system,
            config,
            fit_method=fit_method,
            directions=directions,
        )
        data[j, :] = rate_data["data"]

    if rate_data is None:
        msg = "Must have at least one rate!"
        raise ValueError(msg)

    return {
        "data": data.ravel(),
        "basis": TupleBasis(
            FundamentalBasis(len(conditions)),
            rate_data["basis"],
        ),
    }


def calculate_effective_mass_from_gradient(
    temperature: float,
    gradient: float,
) -> float:
    return Boltzmann * temperature / (gradient**2)


@dataclass
class RateAgainstMomentumFitData:
    """Stores data from linear fit with calculated effective mass."""

    gradient: float
    intercept: float


def get_rate_against_momentum_linear_fit(
    values: ValueList[MomentumBasis],
) -> RateAgainstMomentumFitData:
    k_points = values["basis"].k_points
    rates = np.real(values["data"])
    fit = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
        np.polynomial.Polynomial.fit(  # type: ignore bad library type
            k_points,
            rates,
            deg=[0, 1],
            domain=(0, np.max(k_points)),
            window=(0, np.max(k_points)),
        ).coef,
    )
    return RateAgainstMomentumFitData(fit[1], fit[0])


def get_effective_mass_data_from_linear_fit(
    data: ValueList[MomentumBasis],
    temperature: float,
) -> float:
    return calculate_effective_mass_from_gradient(
        temperature,
        get_rate_against_momentum_linear_fit(data).gradient,
    )


def get_linear_fit_effective_mass_data(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> float:
    rate_data = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
    )

    return get_effective_mass_data_from_linear_fit(rate_data, config.temperature)


SimulationCondition = tuple[System, PeriodicSystemConfig, str]


def get_free_particle_rate_for_conditions(
    conditions: Iterable[SimulationCondition],
) -> list[float]:
    return [get_free_particle_rate(s, c) for (s, c, _) in conditions]


def get_free_rate_against_momentum(
    system: System,
    config: PeriodicSystemConfig,
    *,
    directions: list[tuple[int, ...]] | None = None,
) -> ValueList[MomentumBasis]:
    directions = _get_default_directions(config) if directions is None else directions

    conditions = get_conditions_at_directions(system, config, directions)
    rates = get_free_particle_rate_for_conditions(conditions)

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {
        "basis": basis,
        "data": np.array(rates, dtype=np.complex128),
    }


@timed
def get_linear_fit_effective_mass_against_condition_data(
    conditions: list[SimulationCondition],
    directions: list[tuple[int, ...]] | None = None,
    *,
    fit_method: FitMethod[Any] | None = None,
) -> ValueList[FundamentalBasis[int]]:
    n_conditions = len(conditions)
    data = np.zeros(
        (n_conditions),
        dtype=np.complex128,
    )

    for j, (system, config, _) in enumerate(conditions):
        data[j] = get_linear_fit_effective_mass_data(
            system,
            config,
            fit_method=fit_method,
            directions=directions,
        )

    return {
        "data": data.ravel(),
        "basis": FundamentalBasis(len(conditions)),
    }


def get_rate_against_condition_data(
    conditions: list[SimulationCondition],
    *,
    fit_method: FitMethod[Any] | None = None,
) -> ValueList[FundamentalBasis[int]]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    n_conditions = len(conditions)
    data = np.zeros(
        (n_conditions),
        dtype=np.complex128,
    )

    for j, (system, config, _) in enumerate(conditions):
        data[j] = get_boltzmann_rate(system, config, fit_method)

    return {
        "data": data.ravel(),
        "basis": FundamentalBasis(len(conditions)),
    }


def get_effective_mass_against_condition_data(
    conditions: list[SimulationCondition],
    *,
    fit_method: FitMethod[Any] | None = None,
) -> ValueList[FundamentalBasis[int]]:
    rates = get_rate_against_condition_data(conditions, fit_method=fit_method)
    free_rates = get_free_particle_rate_for_conditions(conditions)
    data = np.array(free_rates) / rates["data"]

    return {
        "basis": rates["basis"],
        "data": data.ravel(),
    }


def get_effective_mass_against_momentum_data(
    system: System,
    config: PeriodicSystemConfig,
    *,
    directions: list[tuple[int, ...]] | None = None,
    fit_method: FitMethod[Any] | None = None,
) -> ValueList[MomentumBasis]:
    rates = get_boltzmann_rate_against_momentum_data(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
    )
    free_rates = get_free_rate_against_momentum(
        system,
        config,
        directions=directions,
    )
    data = free_rates["data"] / rates["data"]

    return {
        "basis": rates["basis"],
        "data": data.ravel(),
    }


def get_conditions_for_config(
    systems: Iterable[tuple[System, str]],
    config: PeriodicSystemConfig,
) -> list[SimulationCondition]:
    return [(system, config, label) for (system, label) in systems]


def get_conditions_at_mass(
    system: System,
    config: PeriodicSystemConfig,
    masses: Iterable[float],
) -> list[SimulationCondition]:
    return get_conditions_for_config(
        ((system.with_mass(mass), f"{mass} Kg") for mass in masses),
        config,
    )


def get_conditions_at_barrier_energy(
    system: System,
    config: PeriodicSystemConfig,
    barrier_energies: Iterable[float],
) -> list[SimulationCondition]:
    return get_conditions_for_config(
        (
            (system.with_barrier_energy(energy), f"{energy}")
            for energy in barrier_energies
        ),
        config,
    )


def get_conditions_at_temperatures(
    system: System,
    config: PeriodicSystemConfig,
    temperatures: Iterable[float],
) -> list[SimulationCondition]:
    return [(system, config.with_temperature(t), f"{t} K") for t in temperatures]


def get_conditions_at_directions(
    system: System,
    config: PeriodicSystemConfig,
    directions: Iterable[tuple[int, ...]],
) -> list[SimulationCondition]:
    return [
        (system, config.with_direction(d), f"({', '.join(str(x) for x in d)},)")
        for d in directions
    ]


def _energy_to_mev(energy: float) -> float:
    return energy * 10**3 / electron_volt


def get_conditions_at_energy_range(
    system: System,
    config: PeriodicSystemConfig,
    scattered_energy_ranges: Iterable[float],
) -> list[SimulationCondition]:
    return [
        (
            system,
            dataclasses.replace(
                config,
                instrument_function=SimpleInstrumentFunction(energy_range=(-r, r)),
            ),
            rf"$\pm$ {_energy_to_mev(r):.1f} meV",
        )
        for r in scattered_energy_ranges
    ]


def _get_scattered_energy_change(
    hamiltonian: SingleBasisDiagonalOperator[_BV0],
    state: StateVector[Any],
    direction: tuple[int, ...] | None = None,
) -> float:
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    energy = np.real(calculate_expectation_diagonal(hamiltonian, state))
    scattered_state = apply_operator_to_state(operator, state)
    scattered_energy = calculate_expectation_diagonal(
        hamiltonian,
        scattered_state,
    )

    return np.real(scattered_energy - energy)


def _get_thermal_scattered_energy_change(
    hamiltonian: SingleBasisDiagonalOperator[_BV0],
    temperature: float,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 10,
) -> ValueList[FundamentalBasis[int]]:
    energy_change = np.zeros(n_repeats, dtype=np.complex128)
    for i in range(n_repeats):
        state = get_random_boltzmann_state_from_hamiltonian(hamiltonian, temperature)
        energy_change[i] = _get_scattered_energy_change(hamiltonian, state, direction)

    return {"basis": FundamentalBasis(n_repeats), "data": energy_change}


def get_thermal_scattered_energy_change_against_k(
    system: System,
    config: PeriodicSystemConfig,
    *,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> StatisticalValueList[MomentumBasis]:
    directions = _get_default_directions(config) if directions is None else directions
    hamiltonian = get_hamiltonian(system, config)
    energy_change = np.zeros(len(directions), dtype=np.complex128)
    standard_deviation = np.zeros(len(directions), dtype=np.float64)
    for i, k_point in enumerate(directions):
        data = _get_thermal_scattered_energy_change(
            hamiltonian,
            config.temperature,
            k_point,
            n_repeats=n_repeats,
        )["data"]
        energy_change[i] = np.average(data)
        standard_deviation[i] = np.std(data)

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {
        "data": energy_change,
        "basis": basis,
        "standard_deviation": standard_deviation,
    }


def get_scattered_energy_change_against_k(
    system: System,
    config: PeriodicSystemConfig,
    state: StateVector[Any],
    *,
    directions: list[tuple[int, ...]] | None = None,
) -> ValueList[MomentumBasis]:
    directions = _get_default_directions(config) if directions is None else directions
    hamiltonian = get_hamiltonian(system, config)
    energy_change = np.zeros(len(directions), dtype=np.complex128)
    for i, k_point in enumerate(directions):
        energy_change[i] = _get_scattered_energy_change(hamiltonian, state, k_point)

    basis = MomentumBasis(get_scattered_momentum(system, config, directions))
    return {"data": energy_change, "basis": basis}

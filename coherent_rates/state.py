from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import scipy.optimize
from scipy.constants import (  # type: ignore bad types
    Boltzmann,
    hbar,
)
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.build import get_displacements_x_stacked
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
    get_continuous_potential,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    BlochBasis,
)
from surface_potential_analysis.wavepacket.localization._operator import (
    convert_operator_to_basis,
)

from coherent_rates.solve import get_hamiltonian

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.state_vector.state_vector import StateVector

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.system import System


def _get_coherent_state_for_basis[SBV0: StackedBasisWithVolumeLike[Any, Any, Any]](
    basis: SBV0,
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> StateVector[SBV0]:
    basis_x = stacked_basis_as_fundamental_position_basis(basis)

    displacements = get_displacements_x_stacked(basis, x_0)

    # stores distance from x0
    distance = np.linalg.norm(
        [d["data"] / s for d, s in zip(displacements, sigma_0, strict=False)],
        axis=0,
    )

    # i k.(x - x')
    phi = np.einsum(  # type: ignore unknown lib type
        "ij,i->j",
        [d["data"] for d in displacements],
        k_0,
    )
    data = np.exp(1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return convert_state_vector_to_basis({"basis": basis_x, "data": data / norm}, basis)


@overload
def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[float, ...],
) -> float: ...


@overload
def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> np.ndarray[Any, np.dtype[np.float64]]: ...


def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[float, ...] | tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    potential = get_continuous_potential(
        system.get_potential(config.shape, config.resolution),
    )
    return np.abs(
        np.exp(-potential(cast("Any", x_point)) / (config.temperature * Boltzmann)),
    )


def get_thermal_occupation_x(
    system: System,
    config: PeriodicSystemConfig,
) -> ValueList[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    x_probability = get_thermal_probability_x(
        system,
        config,
        tuple(BasisUtil(potential["basis"]).x_points_stacked),
    )
    return {
        "basis": potential["basis"],
        "data": x_probability.astype(np.complex128) / np.sum(x_probability),
    }


@overload
def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[float, ...],
) -> float: ...


@overload
def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> np.ndarray[Any, np.dtype[np.float64]]: ...


def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[float, ...] | tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    return np.abs(
        np.exp(
            -np.square(hbar * np.linalg.norm(k_point, axis=0))
            / (2 * system.mass * config.temperature * Boltzmann),
        ),
    )


def get_thermal_occupation_k(
    system: System,
    config: PeriodicSystemConfig,
) -> ValueList[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    basis = system.get_potential_basis(config.shape, config.resolution)
    k_basis = stacked_basis_as_fundamental_momentum_basis(basis)
    util = BasisUtil(k_basis)
    k_probability = get_thermal_probability_k(
        system,
        config,
        tuple(util.fundamental_stacked_k_points),
    )
    return {
        "basis": k_basis,
        "data": k_probability.astype(np.complex128) / np.sum(k_probability),
    }


def get_random_coherent_x(
    system: System,
    config: PeriodicSystemConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng

    basis = stacked_basis_as_fundamental_position_basis(
        system.get_potential_basis(config.shape, config.resolution),
    )
    util = BasisUtil(basis)

    while True:
        x0 = tuple[float, ...](
            np.einsum("i,ik->k", rng.random(basis.ndim), util.delta_x_stacked),  # type: ignore lib
        )
        if rng.random() > get_thermal_probability_x(system, config, x0):
            continue
        return x0


def get_random_coherent_k(
    system: System,
    config: PeriodicSystemConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng

    basis = stacked_basis_as_fundamental_position_basis(
        system.get_potential_basis(config.shape, config.resolution),
    )
    util = BasisUtil(basis)

    while True:
        k0 = tuple[float, ...](
            np.einsum("i,ik->k", (0.5 - rng.random(basis.ndim)), util.delta_k_stacked),  # type: ignore lib
        )
        if rng.random() > get_thermal_probability_k(system, config, k0):
            continue
        return k0


def get_random_coherent_coordinates(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rng = np.random.default_rng()

    # position probabilities
    x0 = get_random_coherent_x(
        system,
        config,
        rng=rng,
    )

    # momentum probabilities
    k0 = get_random_coherent_k(
        system,
        config,
        rng=rng,
    )

    return (x0, k0)


def get_coherent_state(
    system: System,
    config: PeriodicSystemConfig,
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    basis = potential["basis"]

    return _get_coherent_state_for_basis(basis, x_0, k_0, sigma_0)


def get_random_coherent_state(
    system: System,
    config: PeriodicSystemConfig,
    sigma_0: tuple[float, ...],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """Generate a Gaussian state.

    x0,k0 are given approximately by a thermal distribution.

    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        sigma_0 (float): width of the state

    Returns:
    -------
        StateVector[...]: random coherent state

    """
    x0, k0 = get_random_coherent_coordinates(system, config)
    return get_coherent_state(
        system,
        config,
        x0,
        k0,
        sigma_0,
    )


def get_boltzmann_state_from_hamiltonian[B0: BasisLike[Any, Any]](
    hamiltonian: SingleBasisDiagonalOperator[B0],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> StateVector[B0]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = (
        boltzmann_distribution / normalization
        if phase is None
        else boltzmann_distribution * np.exp(1j * phase) / normalization
    )
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def get_random_boltzmann_state_from_hamiltonian[B0: BasisLike[Any, Any]](
    hamiltonian: SingleBasisDiagonalOperator[B0],
    temperature: float,
) -> StateVector[B0]:
    rng = np.random.default_rng()
    phase = 2 * np.pi * rng.random(len(hamiltonian["data"]))
    return get_boltzmann_state_from_hamiltonian(hamiltonian, temperature, phase)


def get_random_boltzmann_state(
    system: System,
    config: PeriodicSystemConfig,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    """Generate a random Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)
    return get_random_boltzmann_state_from_hamiltonian(hamiltonian, config.temperature)


def _get_error_operator[SBV0: StackedBasisWithVolumeLike[Any, Any, Any]](
    basis: SBV0,
    x_0: tuple[float, ...],
) -> SingleBasisOperator[SBV0]:
    util = BasisUtil(basis)
    # We only get the location in the x0 direction here
    x_points = util.fundamental_x_points_stacked - np.array(x_0)[:, np.newaxis]
    locations = np.linalg.norm(x_points, axis=0)
    locations /= np.linalg.norm(util.delta_x_stacked, axis=1)[0]

    basis_position = stacked_basis_as_fundamental_position_basis(basis)
    operator: SingleBasisOperator[Any] = {
        "basis": TupleBasis(basis_position, basis_position),
        "data": np.diag(locations**2),
    }
    return convert_operator_to_basis(operator, TupleBasis(basis, basis))


def _get_error_between_states[BB: BlochBasis[Any]](
    hamiltonian: SingleBasisDiagonalOperator[BB],
    error_operator: SingleBasisOperator[BB],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    state = get_boltzmann_state_from_hamiltonian(
        hamiltonian,
        temperature,
        phase,
    )
    prod = calculate_expectation(error_operator, state)
    return np.real(prod)


def _get_local_boltzmann_state_from_hamiltonian[BB: BlochBasis[Any]](
    hamiltonian: SingleBasisDiagonalOperator[BB],
    temperature: float,
    *,
    strategy: LocalizationStrategy | None = None,
) -> StateVector[BB]:
    delta_x_repeat = hamiltonian["basis"][0].wavefunctions["basis"][1].delta_x_stacked
    strategy = (
        UniformXLocalizationStrategy(
            delta_x_repeat,
            tuple(a / 10 for a in np.linalg.norm(delta_x_repeat, axis=1)),
        )
        if strategy is None
        else strategy
    )
    params = strategy.generate_params()

    coherent_state = _get_coherent_state_for_basis(
        hamiltonian["basis"][0],
        params.x_0,
        params.k_0,
        params.sigma_0,
    )

    initial_phase = np.angle(coherent_state["data"])

    if strategy.optimized:
        # This is a very crude way to minimise the width (x-x0)^2
        # It also does not take into account periodic boundaries!
        # Using an external tool like Wannier90 would be much better
        # but more complicated to set up.
        error_operator = _get_error_operator(hamiltonian["basis"][0], params.x_0)

        def _error(phase_vector: np.ndarray) -> float:
            return _get_error_between_states(
                hamiltonian,
                error_operator,
                temperature,
                phase_vector,
            )

        res = scipy.optimize.minimize(
            _error,
            initial_phase,
            method="L-BFGS-B",
            options={
                "disp": True,
                "maxiter": 250,
                "maxfun": np.inf,
                "ftol": 1e-16,
                "gtol": 1e-18,
            },
        )
        phase = res.x

    else:
        phase = initial_phase
    return get_boltzmann_state_from_hamiltonian(hamiltonian, temperature, phase)


def _get_random_k0(
    temperature: float,
    mass: float,
    *,
    n_dim: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng
    stddev = np.sqrt(mass * Boltzmann * temperature) / hbar
    return tuple(rng.normal(0, stddev, n_dim))


@dataclass(frozen=True, kw_only=True)
class LocalizationParams:
    """A set of params specifying localization strategy."""

    x_0: tuple[float, ...]
    k_0: tuple[float, ...]
    sigma_0: tuple[float, ...]

    def __post_init__(self) -> None:
        assert len(self.x_0) == len(self.k_0)  # noqa: S101
        assert len(self.k_0) == len(self.sigma_0)  # noqa: S101

    def __hash__(self) -> int:
        return hash((self.x_0, self.k_0, self.sigma_0))


class LocalizationStrategy(ABC):
    """A strategy to generate localization parameters."""

    @abstractmethod
    def generate_params(self) -> LocalizationParams: ...

    @property
    @abstractmethod
    def optimized(self) -> bool: ...


class UniformXLocalizationStrategy(LocalizationStrategy):
    """A strategy to generate localization parameters with uniform x0."""

    def __init__(
        self,
        delta_x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        sigma_0: tuple[float, ...],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.delta_x = delta_x
        self.sigma_0 = sigma_0
        self.rng = np.random.default_rng() if rng is None else rng

    def generate_params(self) -> LocalizationParams:
        x0 = tuple[float, ...](
            np.einsum("i,ik->k", self.rng.random(self.delta_x[0].size), self.delta_x),
        )
        k0 = (0.0,) * len(x0)
        return LocalizationParams(x_0=x0, k_0=k0, sigma_0=self.sigma_0)

    @property
    def optimized(self) -> bool:
        return False

    def __hash__(self) -> int:
        return hash((tuple(map(tuple, self.delta_x)), self.sigma_0))


class FixedLocalizationStrategy(LocalizationStrategy):
    """A strategy which always generates the same parameters."""

    def __init__(
        self,
        params: LocalizationParams,
    ) -> None:
        self.params = params

    def generate_params(self) -> LocalizationParams:
        return self.params

    @property
    def optimized(self) -> bool:
        return False

    def __hash__(self) -> int:
        return hash(self.params)


class ThermalLocalizationStrategy(LocalizationStrategy):
    """A strategy to generate localization parameters from thermal distributions."""

    def __init__(
        self,
        system: System,
        config: PeriodicSystemConfig,
        sigma_0: tuple[float, ...],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.system = system
        self.config = config
        self.sigma_0 = sigma_0
        self.rng = np.random.default_rng() if rng is None else rng

    def generate_params(self) -> LocalizationParams:
        rng = np.random.default_rng()
        self.system.get_potential(self.config.shape, self.config.resolution)
        x0 = get_random_coherent_x(
            system=self.system,
            config=self.config,
            rng=rng,
        )
        k0 = _get_random_k0(
            self.config.temperature,
            self.system.mass,
            n_dim=len(x0),
            rng=rng,
        )
        return LocalizationParams(x_0=x0, k_0=k0, sigma_0=self.sigma_0)

    @property
    def optimized(self) -> bool:
        return False

    def __hash__(self) -> int:
        return hash((self.system, self.config, self.sigma_0))


def get_local_boltzmann_state_from_hamiltonian[BB: BlochBasis[Any]](
    hamiltonian: SingleBasisDiagonalOperator[BB],
    temperature: float,
    *,
    strategy: LocalizationStrategy | None = None,
) -> StateVector[BB]:
    return _get_local_boltzmann_state_from_hamiltonian(
        hamiltonian,
        temperature,
        strategy=strategy,
    )


def get_local_boltzmann_state(
    system: System,
    config: PeriodicSystemConfig,
    *,
    strategy: LocalizationStrategy | None = None,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    """Generate a local Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system
        x_0 (tuple[float, ...]): center position

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)

    return _get_local_boltzmann_state_from_hamiltonian(
        hamiltonian,
        config.temperature,
        strategy=strategy,
    )

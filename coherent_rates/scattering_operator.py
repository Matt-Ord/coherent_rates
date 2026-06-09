from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator
from surface_potential_analysis.util.decorators import cached, timed
from surface_potential_analysis.wavepacket.get_eigenstate import (
    BlochBasis,
    get_fundamental_wavepacket,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
)

from coherent_rates.config import IdealInstrumentFunction, PeriodicSystemConfig
from coherent_rates.solve import get_hamiltonian

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import TruncatedBasis
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from coherent_rates.config import InstrumentFunction
    from coherent_rates.system import System

    _B0 = TypeVar("_B0", bound=BlochBasis[Any])

    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BlochBasis[Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BlochBasis[Any], covariant=True)


class SparseScatteringOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the (sparse) scattering operator basis."""

    basis: TupleBasisLike[_B0_co, _B1_co]
    """The original basis of the operator."""

    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """The operator stored in sparse form.

    The data is stored in band (out), band (in), bloch k such that the full
    original data can be found by doing

    ```python
    full = np.einsum("ijk->ikjk", operator["data"].reshape(
            operator["basis"][0].vectors["basis"][0].n,
            operator["basis"][1].vectors["basis"][0].n,
            operator["basis"][1].vectors["basis"][1].n,
        ),)
    stacked = full.reshape(
        operator["basis"][0].vectors["basis"][0].n,
        *operator["basis"][0].vectors["basis"][1].shape,
        operator["basis"][1].vectors["basis"][0].n,
        *operator["basis"][1].vectors["basis"][1].shape,
    )
    data = np.roll(
        stacked,
        direction,
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).ravel()
    ```
    """

    direction: tuple[int, ...]
    """The direction of scattering"""


def as_operator_from_sparse_scattering_operator(
    operator: SparseScatteringOperator[_B0, _B0],
) -> SingleBasisOperator[_B0]:
    # Basis of the bloch wavefunction list [band basis, list basis]
    basis = operator["basis"][0].wavefunctions["basis"][0]
    # in shape (band, band, list)
    stacked = operator["data"].reshape(basis[0].n, basis[0].n, -1)

    # in shape (band, list, band, list)
    rolled = np.einsum("ijk,kl->ikjl", stacked, np.eye(stacked.shape[-1])).reshape(  # type: ignore bad types
        basis[0].n,
        *basis[1].shape,
        basis[0].n,
        *basis[1].shape,
    )
    # Roll the lhs basis.
    data = np.roll(
        rolled,
        operator["direction"],
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).ravel()

    return {"basis": operator["basis"], "data": data}


def as_sparse_scattering_operator_from_operator(
    operator: SingleBasisOperator[_B0],
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    # Basis of the bloch wavefunction list
    basis = operator["basis"][0].wavefunctions["basis"][0]
    stacked = operator["data"].reshape(
        basis[0].n,
        *basis[1].shape,
        basis[0].n,
        *basis[1].shape,
    )

    rolled = np.roll(
        stacked,
        tuple(-d for d in direction),
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).reshape(*basis.shape, *basis.shape)

    data = np.einsum("ijkj->ikj", rolled).ravel()  # type: ignore bad types
    return {"basis": operator["basis"], "direction": direction, "data": data}


def apply_scattering_operator_to_state(
    operator: SparseScatteringOperator[_B0, _B0],
    state: StateVector[_B2],
) -> StateVector[_B0]:
    converted = convert_state_vector_to_basis(state, operator["basis"][1])
    data = np.einsum(  # type: ignore bad types
        "ijk,jk->ik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][1].n,
        ),
        # band (in), bloch k
        converted["data"].reshape(
            operator["basis"][1].wavefunctions["basis"][0].shape,
        ),
    )
    stacked = data.reshape(
        operator["basis"][0].wavefunctions["basis"][0][0].n,
        *operator["basis"][0].wavefunctions["basis"][0][1].shape,
    )
    rolled = np.roll(
        stacked,
        operator["direction"],
        axis=tuple(range(1, stacked.ndim)),
    ).ravel()

    return {"basis": operator["basis"][0], "data": rolled}


def apply_scattering_operator_to_states(
    operator: SparseScatteringOperator[_B0, _B0],
    states: StateVectorList[_B2, _B1],
) -> StateVectorList[_B2, _B0]:
    converted = convert_state_vector_list_to_basis(states, operator["basis"][1])
    data = np.einsum(  # type: ignore bad types
        "ijk,ljk->lik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][1].n,
        ),
        # list, band, bloch k
        converted["data"].reshape(
            converted["basis"].shape[0],
            *operator["basis"][1].wavefunctions["basis"][0].shape,
        ),
    )

    stacked = data.reshape(
        converted["basis"].shape[0],
        operator["basis"][0].wavefunctions["basis"][0][0].n,
        *operator["basis"][0].wavefunctions["basis"][0][1].shape,
    )
    rolled = np.roll(
        stacked,
        operator["direction"],
        axis=tuple(range(2, stacked.ndim)),
    ).ravel()

    return {
        "basis": TupleBasis(states["basis"][0], operator["basis"][0]),
        "data": rolled,
    }


@timed
def get_periodic_x_operator_sparse(
    basis: _B0_co,
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0_co, _B0_co]:
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    stacked_nk_points = BasisUtil(bloch_phase_basis).stacked_nk_points
    for i, nk_in in enumerate(zip(*stacked_nk_points, strict=False)):
        # Find the bloch k of the scattered state
        util = BasisUtil(basis.wavefunctions["basis"][0][1])
        idx_out = util.get_flat_index(
            tuple(j + s for (j, s) in zip(nk_in, direction, strict=False)),
            mode="wrap",
        )
        nk_out = tuple(k[idx_out] for k in stacked_nk_points)

        # Direction in the wavefunction space
        # Not this is not the same for all idx_in, idx_out
        bloch_wavefunction_basis = basis.wavefunctions["basis"][1]
        bloch_wavefunction_direction = tuple(
            (d - (out - in_)) // s
            for (d, in_, out, s) in zip(
                direction,
                nk_in,
                nk_out,
                bloch_phase_basis.shape,
                strict=True,
            )
        )

        # Get the periodic x in the basis of bloch wavefunction u(x)
        basis_in = basis.basis_at_bloch_k(nk_in)
        basis_out = basis.basis_at_bloch_k(nk_out)
        periodic_x = convert_operator_to_basis(
            get_periodic_x_operator(
                bloch_wavefunction_basis,
                bloch_wavefunction_direction,
            ),
            TupleBasis(basis_out, basis_in),
        )

        out[:, :, i] = periodic_x["data"].reshape(band_basis.n, band_basis.n)
    return {
        "basis": TupleBasis(basis, basis),
        "data": out.ravel(),
        "direction": direction,
    }


def get_energy_change_operator_sparse(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    """Get the energy of the outgoing state - the energy of the incoming state.

    This is the total increse of energy of the system after scattering
    in the given direction.
    """
    basis = hamiltonian["basis"][0]
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    stacked_nk_points = BasisUtil(bloch_phase_basis).stacked_nk_points

    eigenvalues = hamiltonian["data"].reshape(band_basis.n, bloch_phase_basis.n)
    for i, nk_in in enumerate(zip(*stacked_nk_points, strict=False)):
        # Find the bloch k of the scattered state
        util = BasisUtil(basis.wavefunctions["basis"][0][1])
        j = util.get_flat_index(
            tuple(j + s for (j, s) in zip(nk_in, direction, strict=False)),
            mode="wrap",
        )

        out[:, :, i] = eigenvalues[:, np.newaxis, j] - eigenvalues[np.newaxis, :, i]

    return {
        "basis": hamiltonian["basis"],
        "data": out.ravel(),
        "direction": direction,
    }


@timed
def get_instrument_biased_periodic_x_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    direction: tuple[int, ...],
    instrument_function: InstrumentFunction,
) -> SparseScatteringOperator[_B0, _B0]:
    periodic_x = get_periodic_x_operator_sparse(hamiltonian["basis"][0], direction)
    if isinstance(instrument_function, IdealInstrumentFunction):
        return periodic_x

    scattered_energy = get_energy_change_operator_sparse(hamiltonian, direction)
    instrument_factor = instrument_function.evaluate(
        np.real(scattered_energy["data"]),
    )
    periodic_x["data"] *= instrument_factor

    return periodic_x


def _get_instrument_biased_periodic_x(
    system: System,
    config: PeriodicSystemConfig,
) -> Path:
    config = dataclasses.replace(config, temperature=0)
    return Path(
        f"data/{hash((system, config))}.periodic_x",
    )


@cached(_get_instrument_biased_periodic_x)
def get_instrument_biased_periodic_x(
    system: System,
    config: PeriodicSystemConfig,
) -> SparseScatteringOperator[
    BlochBasis[TruncatedBasis[int, int]],
    BlochBasis[TruncatedBasis[int, int]],
]:
    hamiltonian = get_hamiltonian(system, config)
    return get_instrument_biased_periodic_x_from_hamiltonian(
        hamiltonian,
        tuple(int(x) for x in config.direction),
        config.instrument_function,
    )


@timed
def get_k_operator_sparse(
    basis: _B0_co,
    direction: tuple[float, ...],
) -> SparseScatteringOperator[_B0_co, _B0_co]:
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    wavepacket = get_fundamental_wavepacket(basis.wavefunctions)
    n_bands = band_basis.n
    n_k_crystal = wavepacket["basis"][0][1].n
    n_k_state = wavepacket["basis"][1].n
    wavepacket_data = wavepacket["data"].reshape(
        n_bands,
        n_k_crystal,
        n_k_state,
    )
    # The intrinsic momentum of the state.
    state_k_points = BasisUtil(wavepacket["basis"][1]).fundamental_stacked_k_points
    fundamental_dk = BasisUtil(wavepacket["basis"][1]).fundamental_dk_stacked
    bloch_fractions = get_wavepacket_sample_fractions(wavepacket["basis"][0][1])
    for i, bloch_fraction in enumerate(bloch_fractions.T):
        # The state in momentum basis at the given bloch k
        wavepackets_at_k = wavepacket_data[:, i]
        # The crystal momentum of the state
        k_crystal = np.tensordot(fundamental_dk, bloch_fraction, axes=(0, 0))

        # Calculate direction_j * k_(ji) for each state i
        operator_in_k = np.einsum(
            "ji,j->i",
            state_k_points + k_crystal[:, np.newaxis],
            direction,
        )
        # Convert from momentum basis to eigen-basis
        out[:, :, i] = np.einsum(
            "ai,i,bi->ab",
            np.conj(wavepackets_at_k),
            operator_in_k,
            wavepackets_at_k,
            optimize=True,
        )

    return {
        "basis": TupleBasis(basis, basis),
        "data": out.ravel(),
        "direction": (0, 0, 0),
    }


@timed
def get_k_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    direction: tuple[float, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    return get_k_operator_sparse(hamiltonian["basis"][0], direction)

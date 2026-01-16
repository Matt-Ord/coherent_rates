from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

_DEFAULT_DIRECTION = ()


class InstrumentFunction(ABC):
    """Instrument function for scattered energy selection."""

    @abstractmethod
    def evaluate(
        self: Self,
        energy: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Evaluate the instrument sensitivity at a given energy.

        The energy, is the energy gained by the system during scattering.
        This is the same as the energy lost by the scattered particle.
        """


@dataclass(kw_only=True, frozen=True)
class SimpleInstrumentFunction(InstrumentFunction):
    """Simple instrument function that is constant within a range and zero outside."""

    energy_range: tuple[float, float] = field(
        default=(-np.inf, np.inf),
    )

    def evaluate(
        self: Self,
        energy: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.where(
            (self.energy_range[0] <= energy) & (energy <= self.energy_range[1]),
            1.0,
            0.0,
        )

    def __hash__(self) -> int:
        return hash(self.energy_range)


class IdealInstrumentFunction(SimpleInstrumentFunction):
    """Ideal instrument function that is constant for all energies."""

    def __init__(self: Self) -> None:
        super().__init__(energy_range=(-np.inf, np.inf))


@dataclass(kw_only=True, frozen=True)
class ExponentialInstrumentFunction(InstrumentFunction):
    """An instrument who's response is an exponential decay."""

    width: float
    optimal_energy_out: float
    incoming_energy: float

    def evaluate(
        self: Self,
        energy: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        energy_out = self.incoming_energy - energy

        return np.where(
            energy_out > 0,
            np.exp(
                -0.5 * (np.abs(energy_out - self.optimal_energy_out) / self.width),
            ),
            0.0,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.width,
                self.optimal_energy_out,
                self.incoming_energy,
            ),
        )


@dataclass(frozen=True)
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    truncation: int | None = None
    temperature: float = field(default=150, kw_only=True)
    instrument_function: InstrumentFunction = field(
        default_factory=IdealInstrumentFunction,
        kw_only=True,
    )
    direction: tuple[int, ...] = field(default=_DEFAULT_DIRECTION, kw_only=True)

    def __post_init__(self: Self) -> None:
        if self.direction is _DEFAULT_DIRECTION:
            object.__setattr__(self, "direction", tuple(0 for _ in self.shape))

    def with_direction(self: Self, direction: tuple[int, ...]) -> Self:
        return dataclasses.replace(self, direction=direction)

    def with_temperature(self: Self, temperature: float) -> Self:
        return dataclasses.replace(self, temperature=temperature)

    def with_resolution(self: Self, resolution: tuple[int, ...]) -> Self:
        return dataclasses.replace(self, resolution=resolution)

    def with_shape(self: Self, shape: tuple[int, ...]) -> Self:
        return dataclasses.replace(self, shape=shape)

    def with_truncation(self: Self, truncation: int | None) -> Self:
        return dataclasses.replace(self, truncation=truncation)

    @property
    def n_bands(self: Self) -> int:
        """Total number of bands.

        Parameters
        ----------
        self : Self

        Returns
        -------
        int

        """
        return (
            np.prod(self.resolution).item()
            if self.truncation is None
            else self.truncation
        )

    def __hash__(self: Self) -> int:
        return hash(
            (
                self.shape,
                self.resolution,
                self.n_bands,
                self.temperature,
                self.direction,
                self.instrument_function,
            ),
        )

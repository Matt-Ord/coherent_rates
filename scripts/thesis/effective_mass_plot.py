import numpy as np
from matplotlib.scale import SymmetricalLogScale
from scipy.constants import hbar
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_transformed_basis,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_with_eigenvalues_to_basis,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_transformed_energy_1d,
    plot_wavepacket_transformed_energy_effective_mass_against_energy,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from scripts.thesis.bandstructure_plot import CAM_DARK_BLUE
from scripts.thesis.util import CAM_WARM_BLUE, get_fancy_figure, setup_rc_params

setup_rc_params()


def plot_rates() -> None:
    config = PeriodicSystemConfig((60,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)
    converted = convert_wavepacket_with_eigenvalues_to_basis(
        wavefunctions,
        list_basis=stacked_basis_as_fundamental_transformed_basis(
            wavefunctions["basis"][0][1],
        ),
    )

    data = converted["eigenvalue"].reshape(
        converted["basis"][0][0].n,
        -1,
    )[list(range(converted["basis"][0][0].n)), 0]

    fig, ax = get_fancy_figure()
    fig, ax, (line, free_line) = plot_wavepacket_transformed_energy_1d(
        wavefunctions,
        free_mass=system.mass,
        measure="abs",
        ax=ax,
        scale_factor=system.lattice_constant / hbar,
    )
    if free_line is not None:
        free_line.set_color(CAM_DARK_BLUE)
    line.set_color(CAM_WARM_BLUE)
    line.set_linestyle("-")
    barrier_energy = system.barrier_energy

    transition = np.argwhere(data > barrier_energy)[0][0]
    print(f"Barrier energy: {barrier_energy:0.2e} J")  # noqa: T201
    ax.set_ylim(None, 6e3)
    ax.set_xlim(0, 3 * transition)
    ax.set_ylabel(r"$R_n(\Delta x)$ / $\mathrm{s}^{-1}$")

    legend = ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
        handles=[free_line],
        labels=["Free Particle"],
    )
    legend.get_frame().set_alpha(0)

    fig.savefig("scripts/thesis/effective_mass_plot.rates.pdf")


def plot_effective_mass() -> None:
    config = PeriodicSystemConfig((60,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, ax = get_fancy_figure()
    fig, ax, line0 = plot_wavepacket_transformed_energy_effective_mass_against_energy(
        wavefunctions,
        true_mass=system.mass,
        ax=ax,
    )
    line0.set_color(CAM_WARM_BLUE)

    line = ax.axvline(system.barrier_energy)  # type: ignore library type
    ax.set_yscale(
        SymmetricalLogScale(None, linthresh=1e-1),
    )
    ax.set_ylabel(r"Effective Mass \quad $\frac{m_\mathrm{eff} }{m}-1$")
    ax.set_xlabel("Average Energy / $J$")
    ax.set_xlim(0, 3 * system.barrier_energy)
    line.set_color(CAM_DARK_BLUE)
    line.set_linestyle("--")

    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handles=[line],
        labels=["Barrier Energy"],
    )
    legend.get_frame().set_alpha(0)

    fig.savefig("scripts/thesis/effective_mass_plot.mass.pdf")


if __name__ == "__main__":
    plot_effective_mass()
    plot_rates()

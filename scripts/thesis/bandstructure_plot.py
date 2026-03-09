from surface_potential_analysis.wavepacket.plot import plot_wavepacket_eigenvalues_1d_k

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    get_paper_figure,
    get_thesis_figure,
)


def _plot_thesis_figure() -> None:
    config = PeriodicSystemConfig((20,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)

    # 2. Set the size
    fig, ax = get_thesis_figure()
    fig, ax, lines = plot_wavepacket_eigenvalues_1d_k(wavefunctions, ax=ax)

    barrier_energy = system.barrier_energy
    print(f"Barrier energy: {barrier_energy:0.2e} J")  # noqa: T201
    ax.set_ylim(None, 2 * barrier_energy)
    barrier_line = ax.axhline(barrier_energy, linestyle="--", label="Barrier Energy")
    barrier_line.set_color(CAM_BLUE.dark)
    barrier_line.set_linewidth(1)
    ax.set_xlabel("Crystal Momentum $k_c$ / $m^{-1}$")
    ax.tick_params(axis="both", which="major", labelsize=8)

    for line in lines:
        line.set_marker("")
        line.set_linestyle("-")
        line.set_color(CAM_BLUE.warm)
        line.set_linewidth(1)

    legend = ax.legend(
        frameon=False,
        loc="upper center",
        fontsize=9,
        handles=[barrier_line],
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/bandstructure_plot.pdf")


def _plot_paper_figure() -> None:
    config = PeriodicSystemConfig((20,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, ax = get_paper_figure()
    fig, ax, lines = plot_wavepacket_eigenvalues_1d_k(wavefunctions, ax=ax)

    barrier_energy = system.barrier_energy
    print(f"Barrier energy: {barrier_energy:0.2e} J")  # noqa: T201
    ax.set_ylim(None, 2 * barrier_energy)
    barrier_line = ax.axhline(barrier_energy, linestyle="--", label="Barrier Energy")
    barrier_line.set_color(CAM_CHERRY.dark)
    barrier_line.set_linewidth(1)
    ax.set_xlabel("Crystal Momentum $k_c$ / $m^{-1}$")
    ax.tick_params(axis="both", which="major", labelsize=8)

    for line in lines:
        line.set_marker("")
        line.set_linestyle("-")
        line.set_color(CAM_BLUE.warm)
        line.set_linewidth(1)

    legend = ax.legend(
        frameon=False,
        loc="upper center",
        fontsize=9,
        handles=[barrier_line],
    )
    legend.get_frame().set_alpha(0)

    fig.savefig("scripts/thesis/bandstructure_plot.pdf")


if __name__ == "__main__":
    _plot_thesis_figure()
    _plot_paper_figure()

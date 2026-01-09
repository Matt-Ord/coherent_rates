from matplotlib import pyplot as plt
from surface_potential_analysis.wavepacket.plot import plot_wavepacket_eigenvalues_1d_k

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

CAM_DARK_BLUE = "#133844"
CAM_WARM_BLUE = "#00BDB6"
CAM_SLATE_1 = "#ECEEF1"

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Utopia"],
        "text.latex.preamble": r"\usepackage{fourier}",
        "font.size": 11,
    },
)


def get_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    wavefunctions = get_bloch_wavefunctions(system, config)

    # 2. Set the size
    fig, ax = plt.subplots(
        figsize=get_fig_size(),
        layout="constrained",
    )
    fig, ax, lines = plot_wavepacket_eigenvalues_1d_k(wavefunctions, ax=ax)

    barrier_energy = system.barrier_energy
    print(f"Barrier energy: {barrier_energy:0.2e} J")  # noqa: T201
    ax.set_ylim(None, 2 * barrier_energy)
    barrier_line = ax.axhline(barrier_energy, linestyle="--", label="Barrier Energy")
    barrier_line.set_color(CAM_DARK_BLUE)
    barrier_line.set_linewidth(1)
    ax.set_xlabel("Crystal Momentum $k_c$ / $m^{-1}$")
    ax.tick_params(axis="both", which="major", labelsize=8)

    for line in lines:
        line.set_marker("")
        line.set_linestyle("-")
        line.set_color(CAM_WARM_BLUE)
        line.set_linewidth(1)

    legend = ax.legend(
        frameon=False,
        loc="upper center",
        fontsize=9,
        handles=[barrier_line],
    )
    legend.get_frame().set_alpha(0)
    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor((0, 0, 0, 0))
    fig.savefig("scripts/thesis/bandstructure_plot.pdf")

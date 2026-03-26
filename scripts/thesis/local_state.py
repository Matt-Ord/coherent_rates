import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.state_vector.plot import (
    plot_state_1d_k,
    plot_state_1d_x,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.solve import get_hamiltonian
from coherent_rates.state import (
    FixedLocalizationStrategy,
    LocalizationParams,
    get_coherent_state,
    get_local_boltzmann_state,
    get_random_boltzmann_state,
)
from coherent_rates.system import SODIUM_COPPER_BRIDGE_SYSTEM_1D
from coherent_rates.util import (
    CAM_BLUE,
    format_axis_scientific,
    get_paper_figure,
    get_thesis_figure,
)


def get_random_k0(
    temperature: float,
    mass: float,
    *,
    n_dim: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng
    stddev = np.sqrt(mass * Boltzmann * temperature) / hbar
    return tuple(rng.normal(0, stddev, n_dim))


def plot_local_state_comparison() -> None:
    fig, ax = get_thesis_figure()

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(100,),
        truncation=25,
        temperature=155,
    )

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    get_hamiltonian.load_or_call_cached(system, config)

    rng = np.random.default_rng()
    strategy = FixedLocalizationStrategy(
        LocalizationParams(
            x_0=(system.lattice_constant * (1 + 0.05 * rng.uniform(-0.5, 0.5)),),
            sigma_0=(system.lattice_constant / 30,),
            k_0=get_random_k0(
                config.temperature,
                system.mass,
                n_dim=1,
                rng=rng,
            ),
        ),
    )
    initial_state = get_local_boltzmann_state(
        system,
        config,
        strategy=strategy,
    )
    fig, ax, line = plot_state_1d_x(initial_state, ax=ax)
    line.set_color(CAM_BLUE.warm)
    line.set_label("Actual")

    params = strategy.generate_params()
    coherent_state = get_coherent_state(
        system,
        config,
        params.x_0,
        params.k_0,
        params.sigma_0,
    )
    fig, ax, line = plot_state_1d_x(coherent_state, ax=ax)
    line.set_color(CAM_BLUE.dark)
    line.set_label("Target")
    line.set_linestyle("--")
    line.set_alpha(0.7)

    ax.set_xlabel(r"$x$ / $\mathrm{m}$")
    ax.set_ylabel(r"$|\langle x|\psi\rangle|$")
    ax.set_xlim(0, 7 * system.lattice_constant)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="upper right",
        borderpad=1.0,
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0, 0, 1, 1),
    )
    _, _, inset_line = plot_state_1d_k(initial_state, measure="abs", ax=inset_ax)
    inset_line.set_color(CAM_BLUE.warm)

    _, _, inset_line = plot_state_1d_k(coherent_state, measure="abs", ax=inset_ax)
    inset_line.set_color(CAM_BLUE.dark)
    inset_line.set_linestyle("--")
    inset_line.set_alpha(0.7)

    inset_ax.xaxis.get_offset_text().set_fontsize(8)
    inset_ax.yaxis.get_offset_text().set_fontsize(8)
    inset_ax.set_facecolor((0, 0, 0, 0))
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.set_xlim(-5e11, 5e11)
    format_axis_scientific(inset_ax.yaxis)

    inset_ax.set_xlabel(r"k / $\mathrm{m}^{-1}$", fontsize=9, labelpad=-1)
    inset_ax.set_ylabel(r"$|\langle k|\psi\rangle|$", fontsize=9, labelpad=-1)

    legend = ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/local_state.pdf")


def plot_local_state_comparison_paper() -> None:
    fig, ax = get_paper_figure()

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(100,),
        truncation=25,
        temperature=155,
    )

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    get_hamiltonian.load_or_call_cached(system, config)

    rng = np.random.default_rng()
    strategy = FixedLocalizationStrategy(
        LocalizationParams(
            x_0=(system.lattice_constant * (1 + 0.05 * rng.uniform(-0.5, 0.5)),),
            sigma_0=(system.lattice_constant / 25,),
            k_0=get_random_k0(
                config.temperature,
                system.mass,
                n_dim=1,
                rng=rng,
            ),
        ),
    )
    initial_state = get_local_boltzmann_state(
        system,
        config,
        strategy=strategy,
    )
    fig, ax, line = plot_state_1d_x(initial_state, ax=ax)
    line.set_color(CAM_BLUE.warm)
    line.set_label("Actual")

    params = strategy.generate_params()
    coherent_state = get_coherent_state(
        system,
        config,
        params.x_0,
        params.k_0,
        params.sigma_0,
    )
    fig, ax, line = plot_state_1d_x(coherent_state, ax=ax)
    line.set_color(CAM_BLUE.dark)
    line.set_label("Target")
    line.set_linestyle("--")
    line.set_alpha(0.7)

    ax.set_xlabel(r"$x$ / $\mathrm{m}$")
    ax.set_ylabel(r"$|\langle x|\psi\rangle|$")
    ax.set_xlim(0, 7 * system.lattice_constant)

    inset_ax = inset_axes(
        ax,
        width="45%",
        height="45%",
        loc="upper right",
        borderpad=1.0,
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0, 0, 1, 1),
    )
    _, _, inset_line = plot_state_1d_k(initial_state, measure="abs", ax=inset_ax)
    inset_line.set_color(CAM_BLUE.warm)

    _, _, inset_line = plot_state_1d_k(coherent_state, measure="abs", ax=inset_ax)
    inset_line.set_color(CAM_BLUE.dark)
    inset_line.set_linestyle("--")
    inset_line.set_alpha(0.7)

    inset_ax.xaxis.get_offset_text().set_fontsize(8)
    inset_ax.yaxis.get_offset_text().set_fontsize(8)
    inset_ax.set_facecolor((0, 0, 0, 0))
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.tick_params(axis="both", which="major", labelsize=8)
    inset_ax.set_xlim(-5e11, 5e11)
    format_axis_scientific(inset_ax.yaxis)

    inset_ax.set_xlabel(r"k / $\mathrm{m}^{-1}$", fontsize=9, labelpad=-1)
    inset_ax.set_ylabel(r"$|\langle k|\psi\rangle|$", fontsize=9, labelpad=-1)
    inset_ax.yaxis.get_offset_text().set_va("top")
    inset_ax.yaxis.get_offset_text().set_position((0.01, 0))
    legend = ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0)
    fig.savefig("scripts/thesis/local_state.pdf")


def plot_random_state() -> None:
    fig, ax = get_thesis_figure()

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(100,),
        truncation=25,
        temperature=155,
    )

    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    get_hamiltonian.load_or_call_cached(system, config)

    initial_state = get_random_boltzmann_state(
        system,
        config,
    )
    fig, ax, line = plot_state_1d_x(initial_state, ax=ax)
    line.set_color(CAM_BLUE.warm)
    line.set_label("Actual")

    ax.set_xlabel(r"$x$ / $\mathrm{m}$")
    ax.set_ylabel(r"$|\langle x|\psi\rangle|$")
    ax.set_xlim(0, 7 * system.lattice_constant)

    fig.savefig("scripts/thesis/local_state.random.pdf")


if __name__ == "__main__":
    plot_local_state_comparison()
    plot_local_state_comparison_paper()
    plot_random_state()

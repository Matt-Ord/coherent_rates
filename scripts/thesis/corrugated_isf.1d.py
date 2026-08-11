from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
    get_scattered_momentum,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_SLATE_1,
    format_axis_scientific,
    get_paper_isf_figure,
    get_thesis_fig_size,
    setup_fancy_figure,
    setup_rc_params,
)


def get_double_thesis_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params()
    w, h = get_thesis_fig_size()
    fig, (ax1, ax2) = plt.subplots(
        figsize=fig_size or (2 * w, h),
        ncols=2,
        layout="constrained",
    )
    setup_fancy_figure(fig, [ax1, ax2])
    return fig, (ax1, ax2)


def plot_periodic_isf_for_paper() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (400,),
        (100,),
        direction=(18 * 5,),
        truncation=25,
        temperature=155,
    )
    times = EvenlySpacedTimeBasis(1000, 1, 0, 2.0e-12)
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201

    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )

    fig, (ax0, ax1) = get_paper_isf_figure()
    fig, ax0, line = plot_value_list_against_time(isf, measure="real", ax=ax0)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)

    method = GaussianMethod(measure="abs")
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])
    fig, ax0, line = plot_value_list_against_time(
        fitted_data,
        ax=ax0,
        measure="real",
    )
    line.set_label("Fitted")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")
    ax0.set_xlabel("")
    ax0.set_ylabel(r"$\Re{(I(\Delta k, t))}$")
    ax0.set_ylim(0, 1)

    format_axis_scientific(ax0.yaxis)
    format_axis_scientific(ax1.yaxis)

    ax0.legend(frameon=False, loc="lower left", fontsize=9)

    _, _, inset_line = plot_value_list_against_time(isf, measure="imag", ax=ax1)
    inset_line.set_color(CAM_BLUE.warm)
    ax1.set_ylim(-0.01, 0.02)
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_facecolor(CAM_SLATE_1)
    ax1.yaxis.get_offset_text().set_va("top")  # type: ignore[attr-defined]
    ax1.yaxis.get_offset_text().set_ha("right")  # type: ignore[attr-defined]
    ax1.yaxis.get_offset_text().set_position((-0.01, 0))

    fig.canvas.draw()
    ax1.set_xticks(ax0.get_xticks())
    ax0.set_xticks(ax1.get_xticks())
    ax1.set_xlabel(r"Time / $s$")
    ax1.set_ylabel(r"$\Im{(I(\Delta k, t))}$")

    ax0.yaxis.set_label_coords(-0.10, 0.5)
    ax1.yaxis.set_label_coords(-0.10, 0.5)
    fig.savefig("scripts/thesis/corrugated_isf.1d.paper.pdf")


def plot_periodic_isf_for_thesis() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (390,),
        (100,),
        direction=(67,),
        truncation=50,
        temperature=155,
    )
    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201

    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )

    fig, (ax0, ax1) = get_double_thesis_figure()
    fig, ax0, line = plot_value_list_against_time(isf, measure="real", ax=ax0)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)

    method = GaussianMethod(measure="abs")
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])
    fig, ax0, line = plot_value_list_against_time(
        fitted_data,
        ax=ax0,
        measure="real",
    )
    line.set_label("Fitted")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")
    ax0.set_xlabel("")
    ax0.set_ylabel(r"$\Re{(I(\Delta k, t))}$")
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, 2e-12)

    format_axis_scientific(ax0.yaxis)
    format_axis_scientific(ax1.yaxis)

    ax0.legend(frameon=False, loc="lower left", fontsize=9)

    _, _, inset_line = plot_value_list_against_time(isf, measure="imag", ax=ax1)
    inset_line.set_color(CAM_BLUE.warm)
    inset_line.set_label("Simulated")
    ax1.set_ylim(-0.008, 0.016)
    ax1.set_yticks([-0.008, -0.004, 0.0, 0.004, 0.008, 0.012, 0.016])
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_facecolor(CAM_SLATE_1)

    ax1.legend(frameon=False, loc="lower left", fontsize=9)

    fig.canvas.draw()
    ax1.set_xticks(ax0.get_xticks())
    ax0.set_xticks(ax1.get_xticks())
    ax1.set_xlabel(r"Time / $s$")
    ax0.set_xlabel(r"Time / $s$")
    ax1.set_ylabel(r"$\Im{(I(\Delta k, t))}$")

    fig.savefig("scripts/thesis/corrugated_isf.1d.thesis.pdf")


def plot_periodic_isf_for_thesis_large() -> None:
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    config = PeriodicSystemConfig(
        (390,),
        (100,),
        direction=(128,),
        truncation=50,
        temperature=155,
    )
    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k:1 {delta_k:0.3e}")  # noqa: T201

    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=100,
    )

    fig, (ax0, ax1) = get_double_thesis_figure()
    fig, ax0, line = plot_value_list_against_time(isf, measure="real", ax=ax0)
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.warm)

    method = GaussianMethod(measure="abs")
    fit = method.get_fit_from_isf(
        isf,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, isf["basis"])
    fig, ax0, line = plot_value_list_against_time(
        fitted_data,
        ax=ax0,
        measure="real",
    )
    line.set_label("Fitted")
    line.set_color(CAM_BLUE.dark)
    line.set_linestyle("--")
    ax0.set_xlabel("")
    ax0.set_ylabel(r"$\Re{(I(\Delta k, t))}$")
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, 1.1e-12)

    format_axis_scientific(ax0.yaxis)
    format_axis_scientific(ax1.yaxis)

    ax0.legend(frameon=False, loc="lower left", fontsize=9)

    _, _, inset_line = plot_value_list_against_time(isf, measure="imag", ax=ax1)
    inset_line.set_color(CAM_BLUE.warm)
    inset_line.set_label("Simulated")
    ax1.set_ylim(-0.02, 0.04)
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_facecolor(CAM_SLATE_1)

    ax1.legend(frameon=False, loc="lower left", fontsize=9)

    fig.canvas.draw()
    ax1.set_xticks(ax0.get_xticks())
    ax0.set_xticks(ax1.get_xticks())
    ax1.set_xlabel(r"Time / $s$")
    ax0.set_xlabel(r"Time / $s$")
    ax1.set_ylabel(r"$\Im{(I(\Delta k, t))}$")

    fig.savefig("scripts/thesis/corrugated_isf.1d.thesis.large.pdf")


if __name__ == "__main__":
    plot_periodic_isf_for_paper()
    plot_periodic_isf_for_thesis_large()
    plot_periodic_isf_for_thesis()

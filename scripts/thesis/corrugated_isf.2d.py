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
    SODIUM_COPPER_SYSTEM_2D,
)
from coherent_rates.util import (
    CAM_BLUE,
    CAM_CHERRY,
    CAM_SLATE_1,
    format_axis_scientific,
    get_thesis_fig_size,
    get_thesis_figure,
    setup_rc_params_thesis,
)


def get_double_thesis_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params_thesis()
    w, h = get_thesis_fig_size()
    fig, (ax1, ax2) = plt.subplots(
        figsize=fig_size or (2 * w, h),
        ncols=2,
        layout="constrained",
    )
    ax1.set_facecolor(CAM_SLATE_1)
    ax2.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax1.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=9,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )
    ax2.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=9,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax1.xaxis.label.set_fontsize(11)
    ax1.yaxis.label.set_fontsize(11)
    ax2.xaxis.label.set_fontsize(11)
    ax2.yaxis.label.set_fontsize(11)
    return fig, (ax1, ax2)


def plot_periodic_isf_for_thesis() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config_1 = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(1, 0),
        truncation=625,
        temperature=155,
    )
    config_2 = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(2, 0),
        truncation=625,
        temperature=155,
    )
    config_3 = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(3, 0),
        truncation=625,
        temperature=155,
    )
    times_1 = GaussianMethod().get_fit_times(
        system=system,
        config=config_1,
    )
    times_2 = GaussianMethod().get_fit_times(
        system=system,
        config=config_2,
    )
    times_3 = GaussianMethod().get_fit_times(
        system=system,
        config=config_3,
    )
    delta_k_1 = get_scattered_momentum(system, config_1, [config_1.direction])[0]
    delta_k_2 = get_scattered_momentum(system, config_2, [config_2.direction])[0]
    delta_k_3 = get_scattered_momentum(system, config_3, [config_3.direction])[0]
    print(f"Actual delta k:1 {delta_k_1:0.3e}")  # noqa: T201
    print(f"Actual delta k:2 {delta_k_2:0.3e}")  # noqa: T201
    print(f"Actual delta k:3 {delta_k_3:0.3e}")  # noqa: T201

    isf = get_boltzmann_isf(
        system,
        config_1,
        times_1,
        n_repeats=20,
    )

    w, h = get_thesis_fig_size()
    fig, ax0 = get_thesis_figure(fig_size=(1.5 * w, h))
    fig, ax0, line = plot_value_list_against_time(
        {
            "basis": EvenlySpacedTimeBasis(
                100,
                1,
                0,
                GaussianMethod().t_factor,
            ),
            "data": isf["data"],
        },
        measure="real",
        ax=ax0,
    )
    line.set_label(rf"${delta_k_1 * 10**-9:.2} \times 10^9 \mathrm{{m}}^{{-1}}$")
    line.set_color(CAM_BLUE.warm)

    isf = get_boltzmann_isf(
        system,
        config_2,
        times_2,
        n_repeats=20,
    )

    fig, ax0, line = plot_value_list_against_time(
        {
            "basis": EvenlySpacedTimeBasis(
                100,
                1,
                0,
                GaussianMethod().t_factor,
            ),
            "data": isf["data"],
        },
        measure="real",
        ax=ax0,
    )
    line.set_label("Simulated")
    line.set_color(CAM_BLUE.dark)
    line.set_label(rf"${delta_k_2 * 10**-9:.2} \times 10^9 \mathrm{{m}}^{{-1}}$")

    isf = get_boltzmann_isf(
        system,
        config_3,
        times_3,
        n_repeats=20,
    )

    fig, ax0, line = plot_value_list_against_time(
        {
            "basis": EvenlySpacedTimeBasis(
                100,
                1,
                0,
                GaussianMethod().t_factor,
            ),
            "data": isf["data"],
        },
        measure="real",
        ax=ax0,
    )
    line.set_label("Simulated")
    line.set_color(CAM_CHERRY.dark)
    line.set_label(rf"${delta_k_3 * 10**-9:.2} \times 10^9 \mathrm{{m}}^{{-1}}$")

    ax0.set_xlabel("")
    ax0.set_ylabel(r"$\Re{(I(\Delta k, t))}$")
    ax0.set_ylim(0.95, 1)
    ax0.set_xlim(0, 2)

    format_axis_scientific(ax0.yaxis)

    ax0.legend(frameon=False, loc="upper right", fontsize=9)

    fig.canvas.draw()
    ax0.set_xlabel(r"Time / $T_{\mathrm{free}}$")

    fig.savefig("scripts/thesis/corrugated_isf.2d.thesis.pdf")


def plot_periodic_isf_for_thesis_large() -> None:
    system = SODIUM_COPPER_SYSTEM_2D

    config = PeriodicSystemConfig(
        (20, 20),
        (35, 35),
        direction=(5, 0),
        truncation=625,
        temperature=155,
    )
    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    delta_k = get_scattered_momentum(system, config, [config.direction])[0]
    print(f"Actual delta k (large): {delta_k:0.3e}")  # noqa: T201

    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=20,
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
    ax0.set_ylim(0.6, 1)
    ax0.set_xlim(0, 2e-12)

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

    fig.savefig("scripts/thesis/corrugated_isf.2d.thesis.large.pdf")


if __name__ == "__main__":
    plot_periodic_isf_for_thesis_large()
    plot_periodic_isf_for_thesis()

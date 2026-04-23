from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)
CAM_CREST = CamColor(
    light="#FFE2C8",
    warm="#FFC392",
    base="#FD8153",
    dark="#DD3025",
)
CAM_PURPLE = CamColor(
    light="#F2ECF8",
    warm="#D1B7EB",
    base="#A368DF",
    dark="#681FB1",
)
CAM_INDIGO = CamColor(
    light="#EBEDFB",
    warm="#B0B9F1",
    base="#5366E0",
    dark="#29347A",
)
CAM_GREEN = CamColor(
    light="#DFF2EA",
    warm="#AFDFCB",
    base="#4DB78C",
    dark="#13553A",
)


CAM_SLATE_1 = "#ECEEF1"


def setup_rc_params_thesis() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Utopia"],
            "text.latex.preamble": r"\usepackage{fourier}"
            "\n"
            r"\usepackage{amsmath}",
            "font.size": 11,
        },
    )


def get_thesis_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


def get_thesis_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    setup_rc_params_thesis()
    fig, ax = plt.subplots(
        figsize=fig_size or get_thesis_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor((0, 0, 0, 0))
    return fig, ax


def get_paper_fig_size() -> tuple[float, float]:
    return (3.3, 2.5)


def setup_rc_params_paper() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use external LaTeX
            "pgf.rcfonts": False,  # Ignore Matplotlib's internal font settings
            "font.family": "serif",
            "font.serif": ["Charter"],  # Match your \usepackage{charter}
            "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{charter}                    % Main text font
        \usepackage{mathptmx}                   % Math font to match your preamble
        \usepackage{mathtools}                  % For complex math if needed
    """,
            "font.size": 9,  # Matches your 9pt document class
        },
    )


def get_paper_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    setup_rc_params_paper()
    fig, ax = plt.subplots(
        figsize=fig_size or get_paper_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    return fig, ax


def get_paper_isf_figure(
    *,
    fig_size: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    setup_rc_params_paper()
    fig, (ax0, ax1) = plt.subplots(
        nrows=2,
        figsize=fig_size or get_paper_fig_size(),
        layout="constrained",
        sharex=True,
    )
    ax0.set_facecolor(CAM_SLATE_1)
    ax1.set_facecolor(CAM_SLATE_1)
    fig.set_facecolor((0, 0, 0, 0))

    ax0.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )
    ax1.tick_params(
        axis="both",
        direction="in",
        top=True,  # Ticks on top
        right=True,  # Ticks on right
        labelsize=8,  # xtick.labelsize and ytick.labelsize
        which="both",  # Apply to both major and minor ticks if needed
    )

    # 3. Handle Label Sizes
    ax0.xaxis.label.set_fontsize(9)
    ax0.yaxis.label.set_fontsize(9)
    ax1.xaxis.label.set_fontsize(9)
    ax1.yaxis.label.set_fontsize(9)
    return fig, (ax0, ax1)


def format_axis_scientific(ax: Axis) -> None:
    formatter = ticker.ScalarFormatter(useMathText=True)
    # 2. Force scientific notation
    # (0, 0) tells it to use scientific notation for all numbers regardless of size
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.set_major_formatter(formatter)

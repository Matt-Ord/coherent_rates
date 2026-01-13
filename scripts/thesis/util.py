from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure

CAM_DARK_BLUE = "#133844"
CAM_WARM_BLUE = "#00BDB6"
CAM_SLATE_1 = "#ECEEF1"


def setup_rc_params() -> None:
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


def get_fig_size() -> tuple[float, float]:
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27

    # We want half width
    plot_width_in = (total_textwidth_pt / 2) * pt_to_inch

    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.85
    return plot_width_in, plot_height_in


def get_fancy_figure() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(
        figsize=get_fig_size(),
        layout="constrained",
    )
    ax.set_facecolor(CAM_SLATE_1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor((0, 0, 0, 0))
    return fig, ax


def format_axis_scientific(ax: Axis) -> None:
    formatter = ticker.ScalarFormatter(useMathText=True)
    # 2. Force scientific notation
    # (0, 0) tells it to use scientific notation for all numbers regardless of size
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.set_major_formatter(formatter)

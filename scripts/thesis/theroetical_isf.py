import numpy as np
from matplotlib import pyplot as plt

from coherent_rates.util import CAM_BLUE, get_fancy_figure, get_fig_size

x, y = get_fig_size()

wide = False
if wide:
    fig, ax = get_fancy_figure(fig_size=(2 * x, y))
    delta_k = 2
else:
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
            "figure.figsize": (3, 2.5),  # Standard single-column width (~8.5cm)
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )
    fig, ax = get_fancy_figure(fig_size=(3, 2.5))
    delta_k = 1
t = np.linspace(0, 8 / delta_k, 1000)


isf_vals = np.exp(delta_k * (1 - t - np.exp(-t)))
(line,) = ax.plot(t, isf_vals)
line.set_color(CAM_BLUE.warm)
ax.set_xlabel(r"Time $t$ / $\gamma$")
ax.set_ylabel(r"ISF $I(\Delta k, t)$")
line.set_label("Theoretical")

isf_ballistic = np.exp(-delta_k * t**2 / 2)
(line,) = ax.plot(t, isf_ballistic)
line.set_color(CAM_BLUE.dark)
line.set_linestyle("--")
line.set_label("Ballistic limit")

isf_diffusive = np.exp(delta_k * (1 - t))
(line,) = ax.plot(t, isf_diffusive)
line.set_color(CAM_BLUE.dark)
line.set_linestyle("-.")
line.set_label("Diffusive limit")

ax.set_ylim(0, 1.1)
ax.set_xlim(0, 4)

legend = ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=8,
)
legend.get_frame().set_alpha(0)

fig.savefig("scripts/thesis/theoretical_isf.pdf")

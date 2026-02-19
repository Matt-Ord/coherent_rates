import numpy as np

from scripts.thesis.bandstructure_plot import get_fig_size
from scripts.thesis.util import CAM_BLUE, get_fancy_figure

x, y = get_fig_size()

wide = False
if wide:
    fig, ax = get_fancy_figure(fig_size=(2 * x, y))
    delta_k = 2
else:
    fig, ax = get_fancy_figure(fig_size=(x, y))
    delta_k = 1
t = np.linspace(0, 8 / delta_k, 1000)


isf_vals = np.exp(delta_k * (1 - t - np.exp(-t)))
(line,) = ax.plot(t, isf_vals)
line.set_color(CAM_BLUE.warm)
ax.set_xlabel(r"Time $t$ / $\lambda$")
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
    fontsize=9,
)
legend.get_frame().set_alpha(0)

fig.savefig("scripts/thesis/theoretical_isf.pdf")

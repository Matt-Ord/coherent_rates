import numpy as np
from scipy.optimize import curve_fit

from scripts.thesis.util import (
    CAM_BLUE,
    get_fancy_figure,
    setup_rc_params,
)

measured_polarization = [
    (6.8211939683377505, 0.5297344246530274),
    (7.864461130773987, 0.5980768744102642),
    (8.85600780168253, 0.4973247242477593),
    (11.232809545673101, 0.35854298698152964),
    (13.890926476914899, 0.2612036210218829),
    (17.283379487865453, 0.1683670333722571),
    (21.521739326915082, 0.1304287310154839),
    (27.000098243242284, 0.06689346340288371),
    (32.39539301916383, 0.045680618002088336),
]


energy_mev = np.asarray([x[0] for x in measured_polarization])
polarization_efficiency = np.asarray([x[1] for x in measured_polarization])

setup_rc_params()
fig, ax = get_fancy_figure()
ax.plot(
    energy_mev,
    polarization_efficiency,
    marker="x",
    color=CAM_BLUE.warm,
    linestyle="",
    label="Measured",
)
ax.set_xlabel(r"Outgoing Energy / $\mathrm{meV}$")
ax.set_ylabel(r"Polarization")
ax.set_xlim(0, 35)
ax.set_ylim(0, 1)


def lorentzian(
    x: np.ndarray,
    x0: float,
    gamma: float,
    a: float,
    d: float = 0,
) -> np.ndarray:
    return a * (gamma**2 / ((x - x0) ** 2 + gamma**2)) + d


def gaussian(
    x: np.ndarray,
    x_0: float,
    gamma: float,
    a: float,
) -> np.ndarray:
    return a * np.exp(-np.abs(x - x_0) / gamma)


def polarization_efficiency_model(
    x: np.ndarray,
    x0: float,
    gamma: float,
    a: float,
) -> np.ndarray:
    return np.where(x > 0, gaussian(x, x0, gamma, a), 0)


popt, pcov = curve_fit(
    polarization_efficiency_model,
    energy_mev,
    polarization_efficiency,
    p0=[8.0, 10.0, 0.5],
)
x_fit = np.linspace(0, 40, 1000)
y_fit = gaussian(x_fit, *popt)
ax.plot(
    x_fit,
    y_fit,
    linestyle="--",
    color=CAM_BLUE.dark,
    label="Fit",
)
legend = ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=9,
)
legend.get_frame().set_alpha(0)
print(popt)  # noqa: T201

fig.savefig("scripts/thesis/polarization_efficiency.pdf")

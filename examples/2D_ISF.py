from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.isf import get_boltzmann_isf, get_random_boltzmann_state
from coherent_rates.plot import (
    plot_alpha_deltak_comparison,
    plot_system_eigenstates_2d,
    plot_system_evolution_2d,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_2D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5, 5), (10, 10), 100, temperature=155)
    system = HYDROGEN_NICKEL_SYSTEM_2D

    plot_system_eigenstates_2d(system, config, 0)

    times = EvenlySpacedTimeBasis(100, 1, 0, 5e-13)
    state = get_random_boltzmann_state(system, config)
    plot_system_evolution_2d(system, config, state, times)

    n = (5, 5)
    isf = get_boltzmann_isf(system, config, times, n, n_repeats=10)
    fig, ax, line = plot_value_list_against_time(isf)
    fig.show()
    input()

    nk_points = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15)]
    plot_alpha_deltak_comparison(system, config, nk_points=nk_points)

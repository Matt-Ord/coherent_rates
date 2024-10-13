from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.isf import (
    get_boltzmann_isf,
    get_coherent_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), temperature=155, direction=(2,))
    system = SODIUM_COPPER_SYSTEM_1D.with_barrier_energy(0)
    times = EvenlySpacedTimeBasis(151, 1, -75, 5e-11)

    # Plot the ISF for a set of random coherent states
    n_repeats = 500
    coherent_isf = get_coherent_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_time(coherent_isf)
    line.set_label("abs")

    fig, ax, line = plot_value_list_against_time(coherent_isf, measure="real", ax=ax)
    line.set_label("real")

    fig, ax, line = plot_value_list_against_time(coherent_isf, measure="imag", ax=ax)
    line.set_label("imag")
    ax.legend()  # type: ignore unknown
    ax.set_title("Coherent state isf")  # type: ignore unknown
    fig.show()

    # Compare 500 samples to 50 samples
    isf_large = coherent_isf
    fig, ax, line = plot_value_list_against_time(isf_large)
    line.set_label(f"{n_repeats*2} runs")

    n_repeats = 50
    isf_small = get_coherent_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_time(isf_small, ax=ax)
    line.set_label(f"{n_repeats*2} runs")
    ax.legend()  # type: ignore unknown
    ax.set_title("Comparison of coherent ISF with a range of sample sizes")  # type: ignore unknown
    fig.show()

    # Plot the ISF for a set of random boltzmann states
    boltzmann_isf = get_boltzmann_isf(system, config, times, n_repeats=50)
    fig, ax, line = plot_value_list_against_time(boltzmann_isf)
    line.set_label("abs")

    fig, ax, line = plot_value_list_against_time(boltzmann_isf, measure="real", ax=ax)
    line.set_label("real")

    fig, ax, line = plot_value_list_against_time(boltzmann_isf, measure="imag", ax=ax)
    line.set_label("imag")
    ax.legend()  # type: ignore unknown
    ax.set_title("Boltzmann state isf")  # type: ignore unknown
    fig.show()

    # Compare the boltzmann and coherent states
    fig, ax, line = plot_value_list_against_time(coherent_isf, measure="real")
    line.set_label("coherent")
    fig, ax, line = plot_value_list_against_time(boltzmann_isf, ax=ax, measure="real")
    line.set_label("boltzmann")
    ax.legend()  # type: ignore unknown
    ax.set_title("isf comparison")  # type: ignore unknown
    fig.show()

    input()

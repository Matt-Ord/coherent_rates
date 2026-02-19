from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import get_default_isf_times
from coherent_rates.plot import (
    plot_weak_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        direction=(10,),
        truncation=50,
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    times = get_default_isf_times(system=system, config=config)
    times = EvenlySpacedTimeBasis(times.n, times.step, 0, times.delta_t * 5)

    plot_weak_boltzmann_isf(system, config, times)

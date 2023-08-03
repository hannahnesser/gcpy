"""
Provides default locations for several benchmark configuration files:

1. aod_species.yml          : AOD species definitions
2. benchmark_categories.yml : Benchmark category definitions
3. emission_inventories.yml : Emission inventories
4. emission_species.yml     : Emitted species (for plots & tables)
5. lumped_species.yml       : Lumped family species (e.g. Ox, Bry)

The default location for all of these files is in the
gcpy/benchmark/config folder.
"""
from os import path

# Absolute path to the config file folder (gcpy/benchmark/config)
CONFIG_DIR = path.abspath(
    path.join(
        path.dirname(__file__),
        "..",
        "config"
    )
)

# Configuration files in gcpy/benchmark/config
AOD_SPC_YAML = path.join(
    CONFIG_DIR,
    "aod_species.yml"
)
BENCHMARK_CAT_YAML = path.join(
    CONFIG_DIR,
    "benchmark_categories.yml"
)
EMISSION_INV_YAML = path.join(
    CONFIG_DIR,
    "emission_inventories.yml"
)
EMISSION_SPC_YAML = path.join(
    CONFIG_DIR,
    "emission_species.yml"
)
LUMPED_SPC_YAML = path.join(
    CONFIG_DIR,
    "lumped_species.yml"
)

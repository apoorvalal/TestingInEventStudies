# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyfixest as pf

# %%
from saturated import test_treatment_heterogeneity
from dgp import panel_dgp_stagg, generate_treatment_effect
from plotters import diag_plot


# %%
def generate_dgp(
    num_periods=30,
    cohort_specs=None,  # dict with keys for each cohort
    sigma_i=2,
    sigma_t=1,
    sigma_epsilon=1,
    num_units=20_000,
):
    # Default cohort specs if none provided
    if cohort_specs is None:
        cohort_specs = {
            "cohort1": {
                "effect_type": "concave",
                "start_time": 10,
                "max_effect": 1,
                "size": num_units // 3,
            },
            "cohort2": {
                "effect_type": "concave",
                "start_time": 15,
                "max_effect": 1,
                "size": num_units // 3,
            },
            "cohort3": {
                "effect_type": "concave",
                "start_time": 20,
                "max_effect": 1,
                "size": num_units // 3,
            },
        }

    # Extract lists for panel_dgp_stagg
    treatment_starts = [spec["start_time"] for spec in cohort_specs.values()]
    cohort_sizes = [spec["size"] for spec in cohort_specs.values()]

    # Generate treatment effects
    base_treatment_effects = [
        generate_treatment_effect(
            effect_type=spec["effect_type"],
            T=num_periods,
            T0=spec["start_time"],
            max_effect=spec["max_effect"],
        )
        for spec in cohort_specs.values()
    ]
    return base_treatment_effects

    # Run DGP
    dgp = panel_dgp_stagg(
        num_units=num_units,
        num_treated=cohort_sizes,
        num_periods=num_periods,
        treatment_start_cohorts=treatment_starts,
        base_treatment_effects=base_treatment_effects,
        sigma_unit=sigma_i,
        sigma_time=sigma_t,
        sigma_epsilon=sigma_epsilon,
    )

    return dgp


# %%
homog_specs = {
    f"cohort{i}": {
        "effect_type": "concave",
        "start_time": start,
        "max_effect": 1,
        "size": size,
    }
    for i, (start, size) in enumerate(zip([10, 15, 20], [2500, 5000, 2500]))
}

df_homog = generate_dgp(cohort_specs=homog_specs)


# %%
df_homog

# %%

# Extract parameters needed for diagnostic plot
treatment_starts = [spec["start_time"] for spec in homog_specs.values()]
base_effects = [
    generate_treatment_effect(
        effect_type=spec["effect_type"],
        T=30,
        T0=spec["start_time"],
        max_effect=spec["max_effect"],
    )
    for spec in homog_specs.values()
]

# Run diagnostic plot
diag_plot(df_homog, treatment_starts, base_effects)

# %%
test_treatment_heterogeneity(df_homog)

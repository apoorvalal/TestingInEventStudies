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
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
from dgp import panel_dgp_stagg
from plotters import checkplot, diag_plot
from saturated import test_treatment_heterogeneity

# %%
num_periods = 30
treatment_start_cohorts = [10, 15, 20]
num_treated_units = [25_00, 50_00, 25_00]

# effect functions
treat_effect_vector_1 = np.log(
    2 * np.arange(1, num_periods - treatment_start_cohorts[1] + 1)
)
treat_effect_vector_1[8:] = 0  # switch off effects after a week
base_treatment_effects = [
    np.r_[
        np.linspace(2, 0, num_periods - treatment_start_cohorts[0] - 10),
        np.repeat(0, 10),
    ],
    treat_effect_vector_1,
    np.sin(
        np.arange(1, num_periods - treatment_start_cohorts[2] + 1)
    ),  # Treatment effect function for cohort 2
]

sigma_i, sigma_t = 2, 1
sigma_epsilon = 1
dgp = panel_dgp_stagg(
    num_units=20_000,
    num_treated=num_treated_units,
    num_periods=num_periods,
    treatment_start_cohorts=treatment_start_cohorts,
    hetfx=False,
    base_treatment_effects=base_treatment_effects,
    sigma_unit=sigma_i,
    sigma_time=sigma_t,
    sigma_epsilon=sigma_epsilon,
)
Y0, Y1, W, df = dgp["Y0"], dgp["Y1"], dgp["W"], dgp["dataframe"]

# %%
checkplot(df)
plt.savefig("../figtab/respecification_verify.png")
# %%
diag_plot(df, treatment_start_cohorts, base_treatment_effects)
plt.savefig("../figtab/hetfx.png")

# %%
test_treatment_heterogeneity(df)

# %% [markdown]
# ## homogeneous DGP

# %%
num_periods = 30
treatment_start_cohorts = [10, 15, 20]
num_treated_units = [25_00, 50_00, 25_00]

base_treatment_effects = [
    np.log(np.arange(1, num_periods - t + 1)) for t in treatment_start_cohorts
]

# %%

sigma_i, sigma_t = 2, 1
sigma_epsilon = 1
dgp_homog = panel_dgp_stagg(
    num_units=20_000,
    num_treated=num_treated_units,
    num_periods=num_periods,
    treatment_start_cohorts=treatment_start_cohorts,
    hetfx=False,
    base_treatment_effects=base_treatment_effects,
    sigma_unit=sigma_i,
    sigma_time=sigma_t,
    sigma_epsilon=sigma_epsilon,
)
Y0_h, Y1_h, W_h, df_h = (
    dgp_homog["Y0"],
    dgp_homog["Y1"],
    dgp_homog["W"],
    dgp_homog["dataframe"],
)

# %%
diag_plot(df_h, treatment_start_cohorts, base_treatment_effects)
plt.savefig("../figtab/homfx.png")
# %%
test_treatment_heterogeneity(df_h)

# %%

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
import pandas as pd
import matplotlib.pyplot as plt

import pyfixest as pf

from importlib import resources

# %%
from saturated import saturated_event_study

# %% [markdown]
# ## example

# %%
df_multi_cohort = pd.read_csv(
    resources.files("pyfixest.did.data").joinpath("df_het.csv")
)
df_multi_cohort.head()

# %%
pf.panelview(
    df_multi_cohort,
    unit="unit",
    time="year",
    treat="treat",
    collapse_to_cohort=True,
    sort_by_timing=True,
    ylab="Cohort",
    xlab="Year",
    title="Treatment Assignment Cohorts",
    figsize=(0.5, 0.5),
)

# %% [markdown]
# ### event study

# %%
fit_evstud = pf.feols(
    "dep_var ~ i(rel_year, ref=-1.0) | state + year",
    df_multi_cohort,
    vcov={"CRV1": "state"},
)
indices = np.r_[-21:21:1]
labs = dict(zip(fit_evstud._coefnames, indices[indices != -1].astype(str).tolist()))
fit_evstud.iplot(
    plot_backend="matplotlib",
    coord_flip=False,
    labels=labs,
    rotate_xticks=90,
)

# %% [markdown]
# ### saturated

# %%
saturated_event_study(
    df_multi_cohort,
    outcome="dep_var",
    treatment="treat",
    unit_id="unit",
    time_id="year",
    ax = plt.gca(),
)

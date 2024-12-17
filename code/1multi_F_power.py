# %%
import contextlib
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)
# %%
from dgp import panel_dgp_stagg
from saturated import test_treatment_heterogeneity

# %%

#                                   ▄▄▄▄      ██
#                                  ██▀▀▀      ▀▀
#   ▄█████▄   ▄████▄   ██▄████▄  ███████    ████      ▄███▄██  ▄▄█████▄
#  ██▀    ▀  ██▀  ▀██  ██▀   ██    ██         ██     ██▀  ▀██  ██▄▄▄▄ ▀
#  ██        ██    ██  ██    ██    ██         ██     ██    ██   ▀▀▀▀██▄
#  ▀██▄▄▄▄█  ▀██▄▄██▀  ██    ██    ██      ▄▄▄██▄▄▄  ▀██▄▄███  █▄▄▄▄▄██
#    ▀▀▀▀▀     ▀▀▀▀    ▀▀    ▀▀    ▀▀      ▀▀▀▀▀▀▀▀   ▄▀▀▀ ██   ▀▀▀▀▀▀
#                                                     ▀████▀▀


num_periods = 30
treatment_start_cohorts = [10, 15, 20]
num_treated_units = [25_00, 50_00, 25_00]


configs = [
    {
        "name": "homogeneous",  # homogeneous effects
        "base_treatment_effects": lambda t: [
            np.log(np.arange(1, num_periods - t + 1)) for t in treatment_start_cohorts
        ],
    },
    {
        "name": "log_vs_linear_vs_sin",  # original heterogeneous case
        "base_treatment_effects": lambda _: [
            np.r_[
                np.linspace(2, 0, num_periods - treatment_start_cohorts[0] - 10),
                np.repeat(0, 10),
            ],
            np.log(2 * np.arange(1, num_periods - treatment_start_cohorts[1] + 1)),
            np.sin(np.arange(1, num_periods - treatment_start_cohorts[2] + 1)),
        ],
    },
    {
        "name": "small_differences",  # subtle heterogeneity
        "base_treatment_effects": lambda t: [
            np.log(np.arange(1, num_periods - t + 1)) * (1 + i * 0.1)
            for i, t in enumerate(treatment_start_cohorts)
        ],
    },
    {
        "name": "large_differences",  # effects depend on treatment timing
        "base_treatment_effects": lambda t: [
            np.log(np.arange(1, num_periods - t + 1)) * (t / 10)
            for t in treatment_start_cohorts
        ],
    },
    {
        "name": "selection_on_gains",  # subtle heterogeneity
        "base_treatment_effects": lambda t: [
            np.log(np.arange(1, num_periods - t + 1)) * (1 - i * 0.1)
            for i, t in enumerate(treatment_start_cohorts)
        ],
    },
]

additional_configs = [
    {
        "name": "novelty_effects",
        "base_treatment_effects": lambda t: [
            2 * np.exp(-0.3 * np.arange(num_periods - t)) + 0.5  # Sharp decay to 0.5
            for t in treatment_start_cohorts
        ],
    },
    {
        "name": "activity_bias",  # First cohort different from rest
        "base_treatment_effects": lambda t: [
            # First cohort has strong persistent effects
            (
                2.5 * np.ones(num_periods - treatment_start_cohorts[0])
                if i == 0
                # Other cohorts have standard log pattern
                else np.log(np.arange(1, num_periods - t + 1))
            )
            for i, t in enumerate(treatment_start_cohorts)
        ],
    },
]

configs.extend(additional_configs)


# %%
def plot_true_functions(
    treatment_start_cohorts,
    base_treatment_effects,
    title,
    ax,
):
    true_fns = {}
    for c, s in enumerate(treatment_start_cohorts):
        effect_vector_padded = np.pad(
            base_treatment_effects[c],
            (treatment_start_cohorts[-1], 0),
        )

        # Create x-axis values that skip -1
        x_values = np.arange(len(effect_vector_padded))
        x_values = np.where(
            x_values >= treatment_start_cohorts[-1],
            x_values - treatment_start_cohorts[-1],
            x_values - treatment_start_cohorts[-1] - 1,
        )
        true_fns[f"cohort_{s}"] = pd.Series(
            {x: y for x, y in zip(x_values, effect_vector_padded)}
        )

    true_event_study = pd.concat(true_fns).reset_index()
    true_event_study.columns = ["cohort", "rel_time", "true_effect"]
    true_event_study = true_event_study.groupby("rel_time")["true_effect"].mean()
    cmp = plt.get_cmap("viridis", len(true_fns))
    i = 0
    for k, v in true_fns.items():
        ax.plot(v, color=cmp(i), marker=".")
        i += 1
    ax.axvline(-1, color="black", linestyle="--")
    ax.axhline(0, color="black", linestyle=":")
    ax.set_title(title)


# %%
f, ax = plt.subplots(
    3, int(np.ceil(len(configs) / 3)), figsize=(10, 7), sharex=True, sharey=True
)
ax = ax.flatten()
for i, config in enumerate(configs):
    plot_true_functions(
        treatment_start_cohorts,
        config["base_treatment_effects"](treatment_start_cohorts),
        config["name"],
        ax[i],
    )
# delete last subplot
f.delaxes(ax[-2])
f.delaxes(ax[-1])
f.tight_layout()
f.savefig("../figtab/true_functions.png")
# %%

#  ██▄███▄    ▄████▄  ██      ██  ▄████▄    ██▄████
#  ██▀  ▀██  ██▀  ▀██ ▀█  ██  █▀ ██▄▄▄▄██   ██▀
#  ██    ██  ██    ██  ██▄██▄██  ██▀▀▀▀▀▀   ██
#  ███▄▄██▀  ▀██▄▄██▀  ▀██  ██▀  ▀██▄▄▄▄█   ██
#  ██ ▀▀▀      ▀▀▀▀     ▀▀  ▀▀     ▀▀▀▀▀    ▀▀
#  ██


# %%
@contextlib.contextmanager
def suppress_stdout():
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        yield stdout


def single_simulation(
    config, treatment_start_cohorts, num_periods, num_treated_units, seed=42
):
    with suppress_stdout(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Generate data
        dgp = panel_dgp_stagg(
            num_units=20_000,
            num_treated=num_treated_units,
            num_periods=num_periods,
            treatment_start_cohorts=treatment_start_cohorts,
            hetfx=False,
            base_treatment_effects=config["base_treatment_effects"](
                treatment_start_cohorts
            ),
            sigma_unit=2,
            sigma_time=1,
            sigma_epsilon=1,
            seed=seed,
        )
        # Run test
        return test_treatment_heterogeneity(dgp["dataframe"])


# %%
single_simulation(configs[0], treatment_start_cohorts, num_periods, num_treated_units)

# %%


def power_analysis(
    n_sims=1000,
    dgp_configs=configs,
    alpha=0.05,
    n_jobs=-1,
):
    results = []

    for config in dgp_configs:  # Parallel simulation with progress bar
        pvalues = Parallel(n_jobs=n_jobs)(
            delayed(single_simulation)(
                config, treatment_start_cohorts, num_periods, num_treated_units, seed=i
            )
            for i in tqdm(range(n_sims), desc=f"Running {config['name']}")
        )
        # Compute rejection rate
        rejection_rate = np.mean(np.array(pvalues) < alpha)
        results.append(
            {
                "dgp": config["name"],
                "rejection_rate": rejection_rate,
                "pvalues": pvalues,
            }
        )

    return pd.DataFrame(results)


# %% # Run power analysis
results = power_analysis(n_sims=1000, dgp_configs=configs, n_jobs=-1)
results
# %%
results.to_pickle("../tmp/rejection_rates_F.pkl")
# %%
results = pd.read_pickle("../tmp/rejection_rates_F.pkl")
results.loc[results.dgp == "timing_dependent", "dgp"] = "large_differences"
# %%
# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Plot power
results.plot(kind="bar", x="dgp", y="rejection_rate", ax=ax1)
ax1.set_title("Rejection Rate by DGP")
ax1.set_ylabel("rejection rate")
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
ax1.get_legend().remove()
ax1.axhline(0.05, color="r", linestyle="--", label="α=0.05")

# Plot p-value distributions
ax2.boxplot([r for r in results["pvalues"]], labels=results["dgp"])
ax2.set_title("P-value Distributions")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
# ax2.set_yscale("log")
ax2.axhline(0.05, color="r", linestyle="--")

plt.tight_layout()
plt.savefig("../figtab/rejection_rates_F.png")
# %%

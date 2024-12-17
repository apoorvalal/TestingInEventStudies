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
import pickle

import numpy as np
import pandas as pd
import pyfixest as pf

import matplotlib.pyplot as plt
from lets_plot import *
LetsPlot.setup_html()


# %% [markdown]
# ## regression estimators for effect dynamics

# %%
def diff_in_means(df, T, T0):
    cross_sec_df = (
        df.query(f"time >= {T0}").groupby("unit").agg({"Y": "mean", "W": "max"})
    )
    estimate = pf.feols("Y~W", cross_sec_df).tidy().loc["W", "Estimate"]
    return np.repeat(estimate, T - T0)


def cuped(df, T, T0):
    cross_sec_df = (
        df.query(f"time >= {T0}").groupby("unit").agg({"Y": "mean", "W": "max"})
    )
    cross_sec_df = cross_sec_df.merge(
        df.query(f"time < {T0}").groupby("unit").Y.mean().rename("ldv"),
        left_index=True,
        right_index=True,
    )
    estimate = pf.feols("Y~W+ldv", cross_sec_df).tidy().loc["W", "Estimate"]
    return np.repeat(estimate, T - T0)


def twfe(df, T, T0):
    m = pf.feols("Y~W | unit + time", df).tidy()
    estimate = m.loc["W", "Estimate"]
    return np.repeat(estimate, T - T0)


def panel_diff_in_means(df, T, T0):
    mean_outcomes = df.groupby(["W", "time"])["Y"].mean().unstack()
    diff_means = mean_outcomes.iloc[1, :] - mean_outcomes.iloc[0, :]
    return diff_means[diff_means.notna()].values


def event_study(df, T, T0):
    df["ever_treated"] = df.groupby("unit")["W"].transform("max")
    m = pf.feols(f"Y ~ i(time, ever_treated, ref = {T0-1}) | unit + time", df)
    return m.coef()[(T0 - 1) :].values


# %%
def sim_panel(
    base_effect,
    N=1_000_000,
    T=35,
    T0=15,
    sigma_list=[5, 2, 0.01, 2],
    hetfx=False,
    num_treated=None,
    rho=0.7,
    seed=42,
    debug=False,
):
    np.random.seed(seed)
    sigma_unit, sigma_time, sigma_tt, sigma_e = sigma_list
    # Generate data
    unit_ids = np.repeat(np.arange(N), T)
    time_ids = np.tile(np.arange(T), N)
    # Generate unit-specific intercepts and time trends
    unit_fe = np.random.normal(0, sigma_unit, N)
    time_fe = np.random.normal(0, sigma_time, T)
    unit_tt = np.random.normal(0, sigma_tt, N)
    # Generate treatment indicator
    if num_treated is None:
        W = np.random.binomial(1, 0.5, N)
    else:
        treated_units = np.random.choice(N, num_treated, replace=False)
        W = np.zeros(N)
        W[treated_units] = 1
    W = np.repeat(W, T)
    W = W * (time_ids >= T0)
    # Generate treatment effect
    if hetfx:
        unit_effects = np.random.uniform(0.5, 1.5, N)
    else:
        unit_effects = np.ones(N)
    treatment_effect = np.outer(unit_effects, base_effect)
    # Generate serially correlated residuals
    residuals = np.zeros((N, T))
    residuals[:, 0] = np.random.normal(0, sigma_e, N)
    epsilon = np.random.normal(0, 1, (N, T - 1))
    factor = sigma_e * np.sqrt(1 - rho**2)
    for t in range(1, T):
        residuals[:, t] = rho * residuals[:, t - 1] + factor * epsilon[:, t - 1]
    # Generate outcome
    Y = (
        np.repeat(unit_fe, T)
        + np.repeat(unit_tt, T) * time_ids
        + treatment_effect.flatten() * W
        + np.tile(time_fe, N)
        + residuals.flatten()
    )

    # Create DataFrame
    df = pd.DataFrame({"unit": unit_ids, "time": time_ids, "Y": Y, "W": W})
    if debug:
        return Y, W, treatment_effect, df
    return df


# %%
def generate_treatment_effect(effect_type, T, T0, max_effect=1):
    if effect_type == "constant":
        return np.concatenate([np.zeros(T0), np.full(T - T0, max_effect)])
    elif effect_type == "linear":
        return np.concatenate([np.zeros(T0), np.linspace(0, max_effect, T - T0)])
    elif effect_type == "concave":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * 0.5 * np.log(2 * np.arange(1, T - T0 + 1) / (T - T0) + 1),
            ]
        )
    elif effect_type == "positive_then_negative":
        half_point = (T - T0) // 2
        return np.concatenate(
            [
                np.zeros(T0),
                np.linspace(0, max_effect, half_point),
                np.linspace(max_effect, -max_effect, T - T0 - half_point),
            ]
        )
    elif effect_type == "exponential":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * (1 - np.exp(-np.linspace(0, 5, T - T0))),
            ]
        )
    elif effect_type == "sinusoidal":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * np.sin(np.linspace(0, 2 * np.pi, T - T0)),
            ]
        )
    elif effect_type == "random_walk":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * np.cumsum(np.random.randn(T - T0)),
            ]
        )
    else:
        raise ValueError("Unknown effect type")


# %%
# Define the simulation engine
def simulation_engine(
    effect_type, T, T0, max_effect, N, num_treated, sigma_list, hetfx, rho, seed
):
    # Generate true treatment effects
    effect_vector = generate_treatment_effect(effect_type, T, T0, max_effect)
    # Simulate data
    df = sim_panel(
        effect_vector,
        N=N,
        T=T,
        T0=T0,
        sigma_list=sigma_list,
        hetfx=hetfx,
        num_treated=num_treated,
        rho=rho,
        seed=seed,
    )
    # Apply estimators
    estimates = {}
    estimates["true_effect"] = effect_vector[T0:]
    estimators = [diff_in_means, twfe, event_study]
    # apply them and add to the estimates dictionary
    for estimator in estimators:
        estimates[estimator.__name__] = estimator(df, T, T0)
    return estimates


# %%
effect_types = [
    "constant",
    "linear",
    "concave",
    "positive_then_negative",
    "exponential",
    "sinusoidal",
    "random_walk",
]
T, T0 = 35, 15
max_effect = 1
ests = {}
for effect_type in effect_types:
    est = simulation_engine(
        effect_type, T, T0, max_effect, 50_000, 25_000, [5, 2, 0.01, 2], False, 0.7, 42
    )
    ests[effect_type] = pd.DataFrame.from_dict(est)
f, ax = plt.subplots(2, 4, figsize=(16, 8), sharey=False, sharex=True)
for k, v, i in zip(ests.keys(), ests.values(), range(7)):
    v[["true_effect"]].plot(
        ax=ax[i // 4, i % 4], marker=".", ms=6, alpha=1, legend=False
    )
    v.drop(["true_effect"], axis=1).plot(
        ax=ax[i // 4, i % 4], marker=".", alpha=0.7, legend=False
    )
    ax[i // 4, i % 4].set_title(k)
    ax[i // 4, i % 4].axhline(0, color="black", linestyle="--")
    ax[i // 4, i % 4].axhline(v.true_effect.mean(), color="red", linestyle="--")
ax[0, 0].legend()
# destroy empty axis
ax[1, 3].axis("off")
f.tight_layout()
# f.suptitle("Static and Dynamic Effects")
f.savefig("../figtab/static_dynamic_effects.png")


# %% [markdown]
# ## F test

# %%
def f_test_stability(df, T0, vcv={"CRV1": "unit"}, dgp_type="", return_plot=True):
    # Fit models
    df["rel_time"] = df["time"] - (T0) + 1
    df["rel_time"] = df["rel_time"].where(df["W"] == 1, 0)

    restricted = pf.feols("Y ~ i(W) | unit + time", df)
    unrestricted = pf.feols("Y ~ i(rel_time, ref=0) | unit + time", df, vcov=vcv)
    # Get the restricted estimate
    restricted_effect = restricted.coef().iloc[0]
    # Create R matrix - each row tests one event study coefficient
    # against restricted estimate
    n_evstudy_coefs = unrestricted.coef().shape[0]
    R = np.eye(n_evstudy_coefs)
    # q vector is the restricted estimate repeated
    q = np.repeat(restricted_effect, n_evstudy_coefs)
    # Conduct Wald test
    pv = unrestricted.wald_test(R=R, q=q, distribution="chi2")["pvalue"]
    if not return_plot:
        return pv
    plotout = pf.iplot(
        [restricted, unrestricted], coord_flip=False, figsize=(900, 400)
    ) + labs(
        title=f"{dgp_type}",
        subtitle=f"Stability Test p-value ={pv:.3f}",
        x="",
        y="",
    )
    return plotout


# %%
effect_types = [
    "constant",
    "linear",
    "concave",
    "positive_then_negative",
    "exponential",
    "sinusoidal",
    "random_walk",
]
N, num_treated, sigma_list, hetfx, rho, seed = (
    50_000,
    25_000,
    [5, 2, 0.01, 2],
    False,
    0.7,
    42,
)
T, T0 = 35, 15
max_effect = 0.1

for effect_type in effect_types:
    effect_vector = generate_treatment_effect(effect_type, T, T0, max_effect)
    # Simulate data
    df = sim_panel(
        effect_vector,
        N=N,
        T=T,
        T0=T0,
        sigma_list=sigma_list,
        hetfx=hetfx,
        num_treated=num_treated,
        rho=rho,
        seed=seed,
    )
    f_test_stability(df, T0, dgp_type=effect_type).show()

# %% [markdown]
# ## simulations

# %%
from joblib import Parallel, delayed


def run_single_simulation(
    effect_type, T, T0, max_effect, N, num_treated, sigma_list, hetfx, rho, seed
):
    # Generate treatment effect vector
    effect_vector = generate_treatment_effect(effect_type, T, T0, max_effect)

    # Simulate data
    df = sim_panel(
        effect_vector,
        N=N,
        T=T,
        T0=T0,
        sigma_list=sigma_list,
        hetfx=hetfx,
        num_treated=num_treated,
        rho=rho,
        seed=seed,
    )

    # Run stability test and return p-value
    return f_test_stability(df, T0, return_plot=False)


# %%
def compute_power(n_sims=1000, n_jobs=-1):
    effect_types = [
        "constant",
        "linear",
        "concave",
        "positive_then_negative",
        "exponential",
        "sinusoidal",
        "random_walk",
    ]

    base_params = {
        "N": 50_000,
        "num_treated": 25_000,
        "sigma_list": [5, 2, 0.01, 2],
        "hetfx": False,
        "rho": 0.7,
        "T": 35,
        "T0": 15,
        "max_effect": 0.1,
    }

    results = {}
    for effect_type in effect_types:
        # Run parallel simulations
        p_values = Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(
                effect_type=effect_type, seed=i, **base_params  # Use iteration as seed
            )
            for i in range(n_sims)
        )

        # Compute rejection rate at 5% level
        power = np.mean(np.array(p_values) < 0.05)
        results[effect_type] = {"power": power, "p_values": p_values}

    return results


# %%
# %%time
# Run simulations
results = compute_power(n_sims=1000, n_jobs=8)

# %%
with open("../tmp/results_dyn.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
with open("../tmp/results_dyn.pkl", "rb") as f:
    results = pickle.load(f)
results = pd.DataFrame(results).T.reset_index()
results.rename(columns={"power": "rejection_rate", "index": "dgp"}, inplace=True)
results

# %%
# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Plot power
results.plot(kind="bar", x="dgp", y="rejection_rate", ax=ax1)
ax1.xaxis.set_tick_params(rotation=45)
ax1.set_title("Rejection Rate by DGP")
ax1.set_ylabel("rejection rate")
ax1.set_xlabel("")
ax1.axhline(0.05, color="r", linestyle="--", label="α=0.05")

# Plot p-value distributions
ax2.boxplot([r for r in results["p_values"]], labels=results["dgp"])
ax2.set_title("P-value Distributions")
ax2.axhline(0.05, color="r", linestyle="--")
ax2.xaxis.set_tick_params(rotation=45)
plt.tight_layout()
plt.savefig("../figtab/rejection_rates_dyn.png")

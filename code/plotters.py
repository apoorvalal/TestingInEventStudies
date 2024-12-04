import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfixest as pf
from saturated import test_treatment_heterogeneity, saturated_event_study


def diag_plot(df, treatment_start_cohorts, base_treatment_effects):
    df2 = df.merge(
        df.assign(first_treated_period=df.time_id * df.W_it)
        .groupby("unit_id")["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on="unit_id",
    )
    df2["rel_time"] = df2.time_id - df2["first_treated_period"]
    df2["first_treated_period"] = (
        df2["first_treated_period"].replace(np.nan, 0).astype("int")
    )
    df2["rel_time"] = df2["rel_time"].replace(np.nan, np.inf)

    fit_evstud = pf.feols(
        "Y_it ~ i(rel_time, ref=-1.0) | unit_id + time_id",
        df2,
        vcov={"CRV1": "unit_id"},
    )
    res = fit_evstud.tidy()
    # truth
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
    f, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    cmp = plt.get_cmap("Set1")
    i = 0
    for k, v in true_fns.items():
        ax[0].plot(v, color=cmp(i), marker=".")
        i += 1
    ax[0].axvline(-1, color="black", linestyle="--")
    ax[0].axhline(0, color="black", linestyle=":")
    ax[0].set_title("True treatment effect functions")

    event_time = (
        res.index.str.extract(r"\[T\.(-?\d+\.\d+)\]").astype(float).values.flatten()
    )

    ax[1].plot(event_time, res["Estimate"], marker=".", label="2wfe", color=cmp(1))
    ax[1].fill_between(
        event_time,
        res["2.5%"],
        res["97.5%"],
        alpha=0.2,
        color=cmp(1),
    )
    ax[1].plot(true_event_study, color="black", label="true", marker=".")
    ax[1].axvline(-1, color="black", linestyle="--")
    ax[1].axhline(0, color="black", linestyle=":")
    ax[1].set_title("Pooled event study \n 2WFE")
    ax[1].legend()

    # saturated
    _ = saturated_event_study(
        df,
        outcome="Y_it",
        treatment="W_it",
        unit_id="unit_id",
        time_id="time_id",
        ax=ax[2],
    )
    ax[2].set_title("Saturated event study \n cohort X time interactions + 2WFE")

    f.tight_layout()

######################################################################
def checkplot(df):
    mm = test_treatment_heterogeneity(df, debug=True)
    mmres = mm.tidy().reset_index()
    mmres[["time", "cohort"]] = mmres.Coefficient.str.split(":", expand=True)
    mmres["time"] = mmres.time.str.extract(r"\[T\.(-?\d+\.\d+)\]").astype(float)
    mmres["cohort"] = mmres.cohort.str.extract(r"(\d+)")
    mmres.loc[~(mmres.cohort.isna()) & (mmres.time > 0)].index

    evstudy_coefs = {}
    evstudy_coefs["0"] = (
        mmres[mmres.cohort.isna()][["Estimate", "time"]].set_index("time").iloc[:, 0]
    )
    for cohort in mmres.cohort.unique()[1:]:
        evstudy_coefs[cohort] = (
            mmres.loc[mmres.cohort == cohort][["Estimate", "time"]]
            .set_index("time")
            .iloc[:, 0]
        )

    f, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    saturated_event_study(
        df,
        outcome="Y_it",
        treatment="W_it",
        time_id="time_id",
        unit_id="unit_id",
        ax=ax[0],
    )
    ax[1].plot(evstudy_coefs["0"], label="Cohort 0", marker=".")
    ax[1].plot(evstudy_coefs["15"] + evstudy_coefs["0"], label="Cohort 1", marker=".")
    ax[1].plot(evstudy_coefs["20"] + evstudy_coefs["0"], label="Cohort 2", marker=".")
    ax[1].axvline(-0.5, color="black", linestyle="--", alpha=0.5)
    ax[1].axhline(0, color="black", linestyle=":", alpha=0.5)

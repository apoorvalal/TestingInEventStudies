import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfixest as pf


def saturated_event_study(
    df: pd.DataFrame,
    outcome: str = "outcome",
    treatment: str = "treated",
    time_id: str = "time",
    unit_id: str = "unit",
    ax: plt.Axes = None,
):
    # create interactions
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["first_treated_period"] = (
        df["first_treated_period"].replace(np.nan, 0).astype("int")
    )
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    cohort_dummies = pd.get_dummies(
        df.first_treated_period, drop_first=True, prefix="cohort_dummy"
    )
    df_int = pd.concat([df, cohort_dummies], axis=1)
    # formula
    ff = f"""
                {outcome} ~
                {'+'.join([f"i(rel_time, {x}, ref = -1.0)" for x in df_int.filter(like = "cohort_dummy", axis = 1).columns])}
                | {unit_id} + {time_id}
                """
    m = pf.feols(ff, df_int, vcov={"CRV1": unit_id})
    if ax:
        # plot
        res = m.tidy()
        # create a dict with cohort specific effect curves
        res_dict = {}
        for c in cohort_dummies.columns:
            res_cohort = res.filter(like=c, axis=0)
            event_time = (
                res_cohort.index.str.extract(r"\[T\.(-?\d+\.\d+)\]")
                .astype(float)
                .values.flatten()
            )
            res_dict[c] = {"est": res_cohort, "time": event_time}

        i = 0
        cmp = plt.get_cmap("Set1")
        for k, v in res_dict.items():
            ax.plot(v["time"], v["est"]["Estimate"], marker=".", label=k, color=cmp(i))
            ax.fill_between(
                v["time"], v["est"]["2.5%"], v["est"]["97.5%"], alpha=0.2, color=cmp(i)
            )
            i += 1
        ax.axvline(-1, color="black", linestyle="--")
        ax.axhline(0, color="black", linestyle=":")
    return m


def test_treatment_heterogeneity(
    df: pd.DataFrame,
    outcome: str = "Y_it",
    treatment: str = "W_it",
    unit_id: str = "unit_id",
    time_id: str = "time_id",
    retmod: bool = False,
):
    # Get treatment timing info
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["first_treated_period"] = (
        df["first_treated_period"].replace(np.nan, 0).astype("int")
    )
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    # Create dummies but drop TWO cohorts - one serves as base for pooled effects
    cohort_dummies = pd.get_dummies(
        df.first_treated_period, drop_first=True, prefix="cohort_dummy"
    ).iloc[
        :, 1:
    ]  # drop an additional cohort - drops interactions for never treated and baseline

    df_int = pd.concat([df, cohort_dummies], axis=1)

    # Modified formula with base effects + cohort-specific deviations
    ff = f"""
    {outcome} ~
    i(rel_time, ref=-1.0) +
    {'+'.join([f"i(rel_time, {x}, ref = -1.0)" for x in df_int.filter(like = "cohort_dummy", axis = 1).columns])}
    | {unit_id} + {time_id}
    """

    model = pf.feols(ff, df_int, vcov={"CRV1": unit_id})
    P = model.coef().shape[0]

    if retmod:
        return model
    mmres = model.tidy().reset_index()
    mmres[["time", "cohort"]] = mmres.Coefficient.str.split(":", expand=True)
    mmres["time"] = mmres.time.str.extract(r"\[T\.(-?\d+\.\d+)\]").astype(float)
    mmres["cohort"] = mmres.cohort.str.extract(r"(\d+)")
    # indices of coefficients that are deviations from common event study coefs
    event_study_coefs = mmres.loc[~(mmres.cohort.isna()) & (mmres.time > 0)].index
    # Method 2 (K x P) - more efficient
    K = len(event_study_coefs)
    R2 = np.zeros((K, P))
    for i, idx in enumerate(event_study_coefs):
        R2[i, idx] = 1

    test_result = model.wald_test(R=R2, distribution="chi2")
    return test_result["pvalue"]


def test_dynamics(
    df,
    outcome="Y",
    treatment="W",
    time_id="time",
    unit_id="unit",
    vcv={"CRV1": "unit"},
):
    # Fit models
    df = df.merge(
        df.assign(first_treated_period=df[time_id] * df[treatment])
        .groupby(unit_id)["first_treated_period"]
        .apply(lambda x: x[x > 0].min()),
        on=unit_id,
    )
    df["rel_time"] = df[time_id] - df["first_treated_period"]
    df["rel_time"] = df["rel_time"].replace(np.nan, np.inf)
    restricted = pf.feols(f"{outcome} ~ i({treatment}) | {unit_id} + {time_id}", df)
    unrestricted = pf.feols(
        f"{outcome} ~ i(rel_time, ref=0) | {unit_id} + {time_id}", df, vcov=vcv
    )
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
    return pv

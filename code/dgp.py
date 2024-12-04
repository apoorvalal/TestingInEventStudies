import numpy as np
import pandas as pd


def panel_dgp_stagg(
    num_units=100,
    num_periods=30,
    num_treated=[50],
    treatment_start_cohorts=[15],
    sigma_unit=1,
    sigma_time=0.5,
    sigma_epsilon=0.2,
    hetfx=False,
    base_treatment_effects=[0.1 * np.log(np.arange(1, 30 - 15 + 1))],
    return_dataframe=True,
    ar_coef=0.8,
):
    # unit FEs
    unit_intercepts = np.random.normal(0, sigma_unit, num_units)
    ####################################################################
    # time FEs: Generate day-of-the-week pattern
    day_effects = np.array(
        [-0.1, 0.1, 0, 0, 0.1, 0.5, 0.5]
    )  # Stronger effects on weekends
    day_pattern = np.tile(day_effects, num_periods // 7 + 1)[:num_periods]
    # autoregressive structure in time FEs
    ar_coef_time = 0.2
    ar_noise_time = np.random.normal(0, sigma_time, num_periods)
    time_intercepts = np.zeros(num_periods)
    time_intercepts[0] = ar_noise_time[0]
    for t in range(1, num_periods):
        time_intercepts[t] = ar_coef_time * time_intercepts[t - 1] + ar_noise_time[t]
    # Combine day-of-the-week pattern and autoregressive structure
    time_intercepts = day_pattern + time_intercepts - np.mean(time_intercepts)
    ####################################################################
    # Generate autoregressive noise for each unit
    ar_noise = np.random.normal(0, sigma_epsilon, (num_units, num_periods))
    noise = np.zeros((num_units, num_periods))
    noise[:, 0] = ar_noise[:, 0]
    for t in range(1, num_periods):
        noise[:, t] = ar_coef * noise[:, t - 1] + ar_noise[:, t]
    # N X T matrix of potential outcomes under control
    Y0 = unit_intercepts[:, np.newaxis] + time_intercepts[np.newaxis, :] + noise
    ####################################################################
    # Generate heterogeneous multipliers for each unit
    if hetfx:
        heterogeneous_multipliers = np.random.uniform(0.5, 1.5, num_units)
    else:
        heterogeneous_multipliers = np.ones(num_units)
    # random assignment
    treated_units = np.array([], dtype=int)
    treatment_status = np.zeros((num_units, num_periods), dtype=bool)
    ####################################################################
    # Create a 2D array to store the heterogeneous treatment effects
    treatment_effect = np.zeros((num_units, num_periods))
    # iterate over treatment cohorts
    for cohort_idx, (treatment_start, num_treated_cohort) in enumerate(
        zip(treatment_start_cohorts, num_treated)
    ):
        base_treatment_effect = base_treatment_effects[cohort_idx]
        cohort_treatment_effect = np.zeros((num_units, num_periods - treatment_start))

        for i in range(num_units):
            cohort_treatment_effect[i, :] = (
                heterogeneous_multipliers[i] * base_treatment_effect
            )
        cohort_treated_units = np.random.choice(
            np.setdiff1d(np.arange(num_units), treated_units),
            num_treated_cohort,
            replace=False,
        )
        treated_units = np.concatenate((treated_units, cohort_treated_units))
        treatment_status[cohort_treated_units, treatment_start:] = True
        treatment_effect[
            cohort_treated_units, treatment_start:
        ] += cohort_treatment_effect[cohort_treated_units, :]

    # Apply the heterogeneous treatment effect to the treated units
    Y1 = Y0.copy()
    Y1[treatment_status] += treatment_effect[treatment_status]
    ####################################################################
    result = {
        "Y1": Y1,
        "Y0": Y0,
        "W": treatment_status,
        "unit_intercepts": unit_intercepts,
        "time_intercepts": time_intercepts,
    }

    if return_dataframe:
        # Create a DataFrame
        unit_ids = np.repeat(np.arange(num_units), num_periods)
        time_ids = np.tile(np.arange(num_periods), num_units)
        W_it = treatment_status.flatten().astype(int)
        Y_it = np.where(W_it, Y1.flatten(), Y0.flatten())
        unit_intercepts_flat = np.repeat(unit_intercepts, num_periods)
        time_intercepts_flat = np.tile(time_intercepts, num_units)
        df = pd.DataFrame(
            {
                "unit_id": unit_ids,
                "time_id": time_ids,
                "W_it": W_it,
                "Y_it": Y_it,
                "unit_intercept": unit_intercepts_flat,
                "time_intercept": time_intercepts_flat,
            }
        )
        result["dataframe"] = df
    return result


def generate_treatment_effect(effect_type, T, T0, max_effect=1):
    if effect_type == "constant":
        return np.full(T - T0, max_effect)
    elif effect_type == "linear":
        return np.linspace(0, max_effect, T - T0)
    elif effect_type == "concave":
        return max_effect * np.log(2 * np.arange(1, T - T0 + 1) / (T - T0) + 1)
    elif effect_type == "positive_then_negative":
        half_point = (T - T0) // 2
        return np.concatenate(
            [
                np.linspace(0, max_effect, half_point),
                np.linspace(max_effect, -max_effect, T - T0 - half_point),
            ]
        )
    elif effect_type == "exponential":
        return max_effect * (1 - np.exp(-np.linspace(0, 5, T - T0)))
    elif effect_type == "sinusoidal":
        return max_effect * np.sin(np.linspace(0, 2 * np.pi, T - T0))
    elif effect_type == "random_walk":
        return max_effect * np.cumsum(np.random.randn(T - T0))
    else:
        raise ValueError("Unknown effect type")

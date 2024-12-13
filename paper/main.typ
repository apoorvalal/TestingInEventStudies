#import "jmlr.typ": jmlr, theorem, proof
#let authors = (
  (name: "Apoorva Lal", affl: "one", email: ""),
)
#let affls = (
  one: (
    department: "Netflix",
  ),
)


#show: jmlr.with(
  title: [When can we get away with using the two-way fixed effects regression?],
  authors: (authors, affls),
  abstract: "
  The widespread use of the two-way fixed effects regression motivated by folk wisdom that it uncovers the average treatment effect on the treated (ATT) has come under scrutiny recently due to recent results in applied econometrics showing that it fails to uncover meaningful averages of heterogeneous treatment effects in the presence of effect heterogeneity over time and across adoption cohorts. In this paper, we propose simple tests that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT.
  ",
  keywords: ("difference in differences", "panel data", "heterogeneous treatment effects"),
  bibliography: bibliography("main.bib"),
  // appendix: include "appendix.typ",
)

// #set math.equation(numbering: "(1)")

= Introduction

Consider a balanced panel-data setting with $i = 1, ..., N$ individuals observed over $t = 1, ..., T$ time periods. For each unit $i$, a binary treatment $w_(i t) := 1(t >= g_(i))$ is assigned at some adoption time $g_(i) in cal(G)$ where $cal(G) := {T} union infinity$ is the set of treatment adoption times and $g_i = infinity$ indicates a never-treated unit. We observe a scalar outcome $y_(i t) = w_(i t) y^(1)_(i t) + (1-w_(i t)) y^(0)_(i t)$, where $y^(1)_(i t)$ and $y^(0)_(i t)$ are potential outcomes under treatment and control, respectively.

The following two-way fixed effects regression

$
  y_(i t) = tau w_(i t) + alpha_i + lambda_t + epsilon_(i t)
$

is a workhorse regression in applied economics and adjacent fields for the estimation of causal effects in such settings. The estimand that researchers typically seek to estimate in panel data settings is the Average Treatment effect on the Treated (ATT) ($EE[y^(1)_(i t) - y^(0)_(i t) | w_(i t) = 1]$), and researchers often interpret the coefficient on the treatment indicator, $tau$, as an estimate of the ATT.
The above regression's dynamic ('event study') counterpart

$
y_(i t) = sum_(s != -1)^(T) gamma_s Delta_(i t)^s + alpha_i + lambda_t + epsilon_(i t)
$

where $Delta_(i t)^s$ is an indicator for the $s$-th period relative to the adoption time (which in turn is the first-difference of the treatment indicator), is also widely used to estimate the dynamic ATT.

When $g_i in {T_0, infinity}$, the above regressions are unbiased estimates of the ATT under the assumption of parallel trends. However, when $g_i in {T_0, ..., T-1}$, the above regressions exhibit the 'negative weighting'/'contamination bias' problem (@Goodman-Bacon2021-ys, @De_Chaisemartin2020-za, @Goldsmith-Pinkham2024-ef) the regression coefficient on the treatment indicator, $tau$, is a weighted average of the ATT over time and across cohorts, where the weights are functions of the treatment timing distribution and the dynamic treatment effect heterogeneity and can be negative for some cohorts. This implies that the two-way fixed effects regression can fail to uncover meaningful averages of heterogeneous treatment effects over time and across adoption cohorts.

This has prompted a cambrian explosion of new estimators that aim to uncover the ATT in the presence of heterogeneous treatment effects over time and across adoption cohorts ( @De_Chaisemartin2021-ln, @Roth2022-sz, @Arkhangelsky2023-rf for reviews). Such heterogeneity-robust estimators typically involve estimating the ATT separately for each cohort using a more precise comparison between the treated cohort and a never-treated group, and then averaging these estimates to obtain an overall estimate of the ATT. While their consistency properties for the ATT are well understood, they are often computationally expensive and have higher variance than the two-way fixed effects regression. This motivates the primary focus of this paper: to develop simple tests that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT. Heuristically, if the dynamic treatment effects are homogeneous over cohorts, then the two-way fixed effects regression is likely to yield unbiased estimates of the ATT that are considerably more precise than alternative estimators that typically discard more data in order to shut down the negative weighting problem.

To motivate the procedure, consider @homfx and @hetfx. In @homfx, there are three adoption cohorts (plus a never-treated cohort - bottom panel), and all cohorts exhibit the same temporal heterogeneity pattern (the effect function is $log(t)$ - top panel), and so the 2WFE event study (blue line in panel 2) is consistent for the true dynamic ATT (black line in panel 2). We can also consistently estimate the cohort-level ATTs with an appropriately saturated regression  (@Abraham2020-wu, @Wooldridge2021-op) as shown in the third panel. In @hetfx, in contrast, we have the same three adoption cohorts, but the three cohorts exhibit radically different temporal heterogeneity: the first exhibits a linear decay down to zero, the second exhibits a log increase followed by zero, and the third exhibits sinusoidal effects. In this case, the 2WFE event study (blue line in panel 2) is not consistent for the true dynamic ATT (black line in panel 2); in fact, the estimated event study suggests a violation of the parallel trend assumption despite the treatments being randomized and thus parallel trends being true in the DGP. We can still estimate the cohort-level ATTs correctly with a saturated regression. The key insight is that testing for differences between a 'pooled' event study (the blue line in the second panel) and cohort X time interactions (that yield the cohort-level estimates in the hird panel) can help us distinguish between the two scenarios. This can be formulated as a joint F-test on the coefficients of the cohort X time interactions in a saturated regression.

#figure(
  image("../figtab/homfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions with homogeneous treatment effects across three cohorts
  ],
) <homfx>


#figure(
  image("../figtab/hetfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions in a DGP with heterogeneous treatment effects across three cohorts
  ],
) <hetfx>


= Methodology

We propose using a joint F-test on the following specification

$
  y_(i t) = alpha_i + lambda_t + beta W_(i t) + sum_(s != -1)^(T) gamma_s Delta_(i t)^s + sum_(s != -1)^(T) delta_s W_(i t) Delta_(i t)^s + epsilon_(i t)
$


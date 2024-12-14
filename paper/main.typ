#import "jmlr.typ": jmlr, theorem, proof
#let authors = (
  (name: "Apoorva Lal", affl: "one", email: ""),
)
#let affls = (
  one: (
    department: "Netflix",
  ),
)
#set math.equation(numbering: "(1)")
#set text(font: "Iosevka", size: 11pt)


#show: jmlr.with(
  title: [When can we get away with using the two-way fixed effects regression?],
  authors: (authors, affls),
  abstract: "
  The widespread use of the two-way fixed effects regression motivated by folk wisdom that it uncovers the average treatment effect on the treated (ATT) has come under scrutiny recently due to recent results in applied econometrics showing that it fails to uncover meaningful averages of heterogeneous treatment effects in the presence of effect heterogeneity over time and across adoption cohorts. In this paper, we propose simple tests that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT. These tests are implemented as methods in the pyfixest python library.
  ",
  keywords: ("difference in differences", "panel data", "heterogeneous treatment effects"),
  bibliography: bibliography("main.bib"),
  appendix: include "appendix.typ",
)



= Introduction

Consider a balanced panel-data setting with $i = 1, ..., N$ individuals observed over $t = 1, ..., T$ time periods. For each unit $i$, a binary treatment $w_(i t) := 1(t >= g_(i))$ is assigned at some adoption time $g_(i) in cal(G)$ where $cal(G) := {T} union infinity$ is the set of treatment adoption times and $g_i = infinity$ indicates a never-treated unit. We observe a scalar outcome $y_(i t) = w_(i t) y^(1)_(i t) + (1-w_(i t)) y^(0)_(i t)$, where $y^(1)_(i t)$ and $y^(0)_(i t)$ are potential outcomes under treatment and control, respectively.

The following two-way fixed effects regression

$
  y_(i t) = tau w_(i t) + alpha_i + lambda_t + epsilon_(i t)
$ <statictwfe>

is a workhorse regression in applied economics and adjacent fields for the estimation of causal effects in such settings. The estimand that researchers typically seek to estimate in panel data settings is the Average Treatment effect on the Treated (ATT) ($EE[y^(1)_(i t) - y^(0)_(i t) | w_(i t) = 1]$), and researchers often interpret the coefficient on the treatment indicator, $hat(tau)$, as an estimate of the ATT.
The above regression's dynamic ('event study') counterpart

$
y_(i t) = sum_(s != -1)^(T) gamma_s Delta_(i t)^s + alpha_i + lambda_t + epsilon_(i t)
$ <eventstudy>

where $Delta_(i t)^s$ is an indicator for the $s$-th period relative to the adoption time (which in turn is the first-difference of the treatment indicator), is also widely used to estimate the dynamic ATT.

When $g_i in {T_0, infinity}$, the above regressions are unbiased estimates of the ATT under the assumption of parallel trends. However, when $g_i in {T_0, ..., T-1}$, the above regressions exhibit the 'negative weighting'/'contamination bias' problem (@Goodman-Bacon2021-ys, @De_Chaisemartin2020-za, @Goldsmith-Pinkham2024-ef) the regression coefficient on the treatment indicator, $hat(tau)$, is a weighted average of the ATT over time and across cohorts, where the weights are functions of the treatment timing distribution and the dynamic treatment effect heterogeneity and can be negative for some cohorts. This implies that the two-way fixed effects regression can fail to uncover meaningful averages of heterogeneous treatment effects over time and across adoption cohorts. The same is true for the event study coefficient vector $bold(gamma)$.

This has prompted a cambrian explosion of new estimators that aim to uncover the ATT in the presence of heterogeneous treatment effects over time and across adoption cohorts ( @De_Chaisemartin2021-ln, @Roth2022-sz, @Arkhangelsky2023-rf for reviews). Such heterogeneity-robust estimators typically involve estimating the ATT separately for each cohort using a more precise comparison between the treated cohort and a never-treated group, and then averaging these estimates to obtain an overall estimate of the ATT. While their consistency properties for the ATT are well understood, they are often computationally expensive and have higher variance than the two-way fixed effects regression. Additionally, a large re-analysis of published work in political science by @chiu2023and finds that they rarely overturn the conclusions of the two-way fixed effects regression, and are typically have considerably larger variance.
This motivates the primary focus of this paper: to develop simple tests that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT. Heuristically, if the dynamic treatment effects are homogeneous over cohorts, then the two-way fixed effects regression is likely to yield unbiased estimates of the ATT that are considerably more precise than alternative estimators that typically discard more data in order to shut down the negative weighting problem.

To motivate the procedure, consider @homfx and @hetfx. In @homfx, there are three adoption cohorts (plus a never-treated cohort - bottom panel), and all cohorts exhibit the same temporal heterogeneity pattern (the effect function is $log(t)$ - top panel), and so the 2WFE event study (blue line in panel 2) is consistent for the true dynamic ATT (black line in panel 2). We can also consistently estimate the cohort-level ATTs with an appropriately saturated regression  (@Abraham2020-wu, @Wooldridge2021-op) as shown in the third panel. In @hetfx, in contrast, we have the same three adoption cohorts, but the three cohorts exhibit radically different temporal heterogeneity: the first exhibits a linear decay down to zero, the second exhibits a log increase followed by zero, and the third exhibits sinusoidal effects. In this case, the 2WFE event study (blue line in panel 2) is not consistent for the true dynamic ATT (black line in panel 2); in fact, the estimated event study suggests a violation of the parallel trend assumption despite the treatments being randomized and thus parallel trends being true in the DGP. We can still estimate the cohort-level ATTs correctly with a saturated regression. The key insight is that testing for differences between a 'pooled' event study (the blue line in the second panel) and cohort X time interactions (that yield the cohort-level estimates in the hird panel) can help us distinguish between the two scenarios. This can be formulated as a joint F-test on the coefficients of the cohort X time interactions in a saturated regression. We provide a formal statement of this test in the next section, and show through simulation studies that this approach can detect cohort-level temporal heterogeneity in a variety of DGPs.

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

Tests considered in the following section take the form of traditional joint tests of multiple linear restrictions, where the null hypothesis is that $bold(R) bold(beta) = bold(q)$ where $bold(R)$ is a $m times k$ matrix of linear restrictions, $bold(beta)$ is a $k times 1$ vector of coefficients, and $q$ is a $m times 1$ vector of constants. The test statistic is then

$
F = frac(
(bold(R) hat(beta) - bold(q))'
[bold(R) hat(bb(V))) bold(R)']^(-1)
(bold(R) hat(beta) - bold(q)),
m) ~ F(m, n-k) " under the null hypothesis"
$

where $hat(bb(V))$ is the cluster-robust variance-covariance matrix of the coefficient estimates.
#footnote[This can be implemented using either a $chi^2$ or $F$ test; the distinction between the two is due to different degrees of freedom that disappear for realistic sample sizes]
We consider two tests: one for testing for event study dynamics, and one for testing for heterogeneity in event study dynamics.

== Testing for event study dynamics <test_dyn>

As a warmup, consider a simple comparison between @statictwfe and @eventstudy. The latter decomposes the ATT across time-periods. For the purposes of testing for event study dynamics, we only care about comparing the equality of the dynamic treatment effects after the treatment is assigned (${gamma_t}_(t=0)^T$) against the common ATT estimate $tau$. We can test the following null hypothesis
$
H_0: {gamma_t}_(t=0)^T = hat(tau) " for all k" > 0
$

by specifying $bold(R) = bold(I)_K$ as a $T_1 times T_1$ identity matrix and $bold(q) = (hat(tau),  ..., hat(tau))'$ as a $T_1$-vector of the restricted estimate ($hat(tau)$ from @statictwfe).
#footnote[
  this can equivalently be formulated by testing for the equality of adjacent elements of $bold(gamma)$, e.g. $gamma_1 = gamma_2$ by specifying $bold(R)$ that contains rows like $[1, -1, 0, ..., 0]$ and $q = [0, ..., 0]$.
]

== Testing for across-cohort heterogeneity in dynamic treatment effects <test_het>

Next, we extend the approach outlined above to construct a test for across-cohort heterogeneity in dynamic treatment effects. A conventional method to estimate the cohort-level ATTs is to estimate the dynamic treatment effects separately for each cohort and then average these estimates to obtain an overall estimate of the ATT (@Abraham2020-wu, @Wooldridge2021-op, @lal2024large), which involves specifying the following regression

$
y_(i t) = alpha_i + lambda_t + sum_(g_i in cal(C)\\ infinity) sum_(s != -1)^(T) bb(1)(g_i = c) tau^(s c) Delta_(i t)^s + epsilon_(i t)
$ <satevent>

This is a saturated event study that constructs cohort $times$ time interactions for each adoption cohort (with $g_i = infinity$ never treated cohort) omitted and therefore recovers the cohort-level event studies. These coefficients are reported in the third panel in @homfx and @hetfx, and correctly uncover the true cohort-level ATTs in the presence of arbitrary heterogeneous treatment effects across cohorts (top panel). The downside of this approach, however, are twofold. First, these regressions can get unwieldy with many cohorts, and the number of parameters grows linearly with the number of cohorts. Second, the cohort level ATTs are self-contained and therefore constructing a test for equality across multiple cohorts is not straightforward. Instead, one may re-specify the saturated event-study regression @satevent as follows:

$
  y_(i t) = alpha_i + lambda_t + sum_(s != -1)^(T) gamma_s Delta_(i t)^s + sum_(c in cal(C)) sum_(s != -1)^(T) delta_s Delta_(i t)^(c s) + epsilon_(i t)
$ <jointreg>

@jointreg returns numerically identical estimates of the cohort-level dynamic ATT as @satevent (illustrated for the @hetfx dgp in @respec), but it allows us to test for differences in dynamic treatment effects over cohorts more easily. This is because unlike @satevent, @jointreg contains a common event study coefficient vector, and cohort-level deviations, which in turn can be jointly tested against the null of zero.


We show in the next section that this test is consistent for the null hypothesis of homogeneous dynamic treatment effects over cohorts, and that it has power against a variety of alternatives. As a concrete example, the joint $p-$value for the cohort $times$ time interactions in @homfx is $0.11$, while the joint p-value for the cohort $times$ time interactions in @hetfx is $0.000$. Thus, we can reject the null hypothesis of homogeneous dynamic treatment effects in @hetfx but not in @homfx, which is consistent with the underlying DGP. In the next section, we show through simulation studies that this test has good power to detect across-cohort heterogeneity in dynamic treatment effects in a variety of DGPs.

= Simulation Studies

== Testing for event study dynamics

To begin, we perform simulation studies based on to study the properties of the testing procedure described in @test_dyn. We consider the simple setting with a single adoption cohort where the treatment effects follow one of the following seven DGPs visualised in @static_dyn.

#figure(
  image("../figtab/static_dynamic_effects.png", width: 105%),
  caption: [
    true treatment effect functions and estimates from difference in means, static, and dynamic two-way fixed effects regressions
  ],
) <static_dyn>

The first DGP has constant effects over time, while the others have varying degrees of temporal heterogeneity. We simulate 500 replications of the data for each DGP, and compute the rejection rate of the joint test for dynamic treatment effects outlined in the previous section. We report the rejection rate and p-value distribution in @rejrates_dyn. We find that the rejection rate for the constant DGP (null) is under the nominal level of $alpha = 0.05$, while the rejection rates for the other DGPs considerably higher. The rejection rate for concave effects is considerably lower, although this is likely due to the fact that the the treatment effects do actually tail off in later time periods and the static effect captures this well.

#figure(
  image("../figtab/rejection_rates_dyn.png", width: 105%),
  caption: [
    Rejection rates over 500 replications for the joint test of dynamic treatment effects using an F-test in DGPs from @static_dyn
  ],
) <rejrates_dyn>


== Testing for across-cohort heterogeneity in dynamic treatment effects

Next, we we perform simulation studies based on to study the properties of the testing procedure described in @test_het. Here, we consider four different DGPs with homogeneous and heterogeneous treatment effect functions across cohorts as illustrated in @truefns. In addition to the two DGPs described in the previous section, we also consider DGPs with 'mild' heterogeneity that applies a scaler multiplier to the concave (log) effect function in @homfx.

#figure(
  image("../figtab/true_functions.png", width: 80%),
  caption: [
    true cohort level effect functions: the first DGP has homogeneous effects across cohorts, while others have heterogeneous effects of varying complexity
  ],
) <truefns>

For each DGP, we simulate 500 replications of the data, and compute the rejection rate of the joint test for cohort-level coefficients outlined in the previous section. We report the rejection rate and p-value distribution in @rejrates. We find that the rejection rate for the homogeneous DGP (null) is under the nominal level of $alpha = 0.05$, while the rejection rates for heterogeneous DGPs are close to 1. This suggests that the test has good power to detect across-cohort heterogeneity in dynamic treatment effects.

#figure(
  image("../figtab/rejection_rates_F.png", width: 100%),
  caption: [
    Rejection rates over 500 replications for the joint test of cohort-level coefficients using an F-test in DGPs from @truefns
  ],
) <rejrates>



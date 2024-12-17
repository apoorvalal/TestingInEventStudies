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
  The use of the two-way fixed effects regression in empirical social science was historically motivated by folk wisdom that it uncovers the average treatment effect on the treated (ATT). This has come under scrutiny recently due to recent results in applied econometrics showing that it fails to uncover meaningful averages of heterogeneous treatment effects in the presence of effect heterogeneity over time and across adoption cohorts, and several heterogeneity-robust alternatives have been proposed. However, these estimators often have higher variance and are therefore under-powered for many applications, which poses a bias-variance tradeoff that is challenging for researchers to navigate. In this paper, we propose simple tests of linear restrictions that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT. These tests are implemented as methods in the pyfixest python library.
  ",
  keywords: ("difference in differences", "panel data", "heterogeneous treatment effects"),
  bibliography: bibliography("main.bib"),
  appendix: include "appendix.typ",
)



= Introduction

Consider a balanced panel-data setting with $i = 1, ..., N$ individuals observed over $t = 1, ..., T$ time periods. For each unit $i$, a binary treatment $w_(i t) := 1(t >= g_(i))$ is assigned at some adoption time $g_(i) in cal(G)$ where $cal(G) := {T} union infinity$ is the set of treatment adoption times and $g_i = infinity$ indicates a never-treated unit. We observe a scalar outcome $y_(i t) = w_(i t) y^(1)_(i t) + (1-w_(i t)) y^(0)_(i t)$, where $y^(1)_(i t)$ and $y^(0)_(i t)$ are potential outcomes under treatment and control, respectively.
#footnote[Defining potential outcomes as $y^(w)_(i t)$ is a strong but common assumption; it requires no carryover - that the outcome for unit $i$ at time $t$ is only influenced by $i$'s current-period treatment and not treatment history. Alternative estimators such as Marginal Structural Models (MSMs) and dynamic panel models permit estimation in the presence of carryover under different strong assumptions but are considerably more computationally challenging, and as such are used infrequently.]

The following two-way fixed effects regression

$
  y_(i t) = tau w_(i t) + alpha_i + lambda_t + epsilon_(i t)
$ <statictwfe>

is a workhorse regression in applied economics and adjacent fields for the estimation of causal effects in such settings. The estimand that researchers typically seek to estimate in panel data settings is the Average Treatment effect on the Treated (ATT) ($EE[y^(1)_(i t) - y^(0)_(i t) | w_(i t) = 1]$), and researchers often interpret the coefficient on the treatment indicator, $hat(tau)$, as an estimate of the ATT.
The above regression's dynamic ('event study') counterpart

$
y_(i t) = sum_(s != -1)^(T) gamma_s Delta_(i t)^s + alpha_i + lambda_t + epsilon_(i t)
$ <eventstudy>

where $Delta_(i t)^s$ is an indicator for the $s$-th period relative to the adoption time for treated units (which in turn is the first-difference of the treatment indicator), is also widely used to estimate the dynamic ATT (@angrist2009mostly ch 5) and diagnose the validity of the parallel trends (which tends to have low power, @rambachan2023more).

When $g_i in {T_0, infinity}$, the above regressions are unbiased estimates of the ATT under the assumption of parallel trends (@lechner2011estimation). However, when $g_i in {T_0, ..., T-1}$, the above regressions exhibit the 'negative weighting'/'contamination bias' problem (@Goodman-Bacon2021-ys, @De_Chaisemartin2020-za, @Goldsmith-Pinkham2024-ef) the regression coefficient on the treatment indicator, $hat(tau)$, is a weighted average of the ATT over time and across cohorts, where the weights are functions of the treatment timing distribution and the dynamic treatment effect heterogeneity and can be negative for some cohorts. This implies that the two-way fixed effects regression can fail to uncover meaningful averages of heterogeneous treatment effects over time and across adoption cohorts. The same is true for the event study coefficient vector $bold(gamma)$.

This has prompted a explosion of research in applied econometrics on new estimators that aim to uncover the ATT in the presence of heterogeneous treatment effects over time and across adoption cohorts (@De_Chaisemartin2021-ln, @Roth2022-sz, @Arkhangelsky2023-rf for reviews). Such heterogeneity-robust estimators typically involve estimating the ATT separately for each cohort using tailored comparisons between each treated cohort and either a never-treated or not-yet-treated group, and then averaging (optionally weighted by inverse-propensity weights, e.g. @Callaway2021-gv) these estimates to obtain an overall estimate of the ATT. While their consistency properties for the ATT are well understood and they avoid the negative weighting problem by construction, they are often computationally expensive and have higher variance than the two-way fixed effects regression.

This poses a practical bias-variance tradeoff for researchers: while the two-way fixed effects regression is computationally simple and has low variance, it may yield biased estimates of the ATT in the presence of heterogeneous treatment effects over time and across adoption cohorts. In contrast, heterogeneity-robust estimators are computationally expensive and have higher variance, but they are consistent for the ATT in the presence of heterogeneous treatment effects over time and across adoption cohorts.  As a practical matter, a large re-analysis of published work in political science by @chiu2023and finds that they rarely overturn the conclusions of the two-way fixed effects regression, and are typically have considerably larger variance. Similarly, @weiss2024much finds that most new heterogeneity-robust estimators are underpowered for realistic effect sizes in the state-level US setting where difference-in-differences approaches commonly used.

This motivates the primary focus of this paper: to develop simple tests that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of the ATT. Heuristically, if the dynamic treatment effects are homogeneous over cohorts, then the two-way fixed effects regression is likely to yield unbiased estimates of the ATT that are considerably more precise than alternative estimators that typically discard more data in order to shut down the negative weighting problem.

To build intuition for this approach, consider @homfx and @hetfx. In @homfx, there are three adoption cohorts (plus a never-treated cohort - bottom panel), and all cohorts exhibit the same temporal heterogeneity pattern (the effect function is $log(t)$ - top panel), and so the 2WFE event study (blue line in panel 2) is consistent for the true dynamic ATT (black line in panel 2). We can also consistently estimate the cohort-level ATTs with an appropriately saturated regression  (@Abraham2020-wu, @Wooldridge2021-op) as shown in the third panel. In @hetfx, in contrast, we have the same three adoption cohorts, but the three cohorts exhibit radically different temporal heterogeneity: the first exhibits a linear decay down to zero, the second exhibits a log increase followed by zero, and the third exhibits sinusoidal effects. In this case, the 2WFE event study (blue line in panel 2) is not consistent for the true dynamic ATT (black line in panel 2); in fact, the estimated event study suggests a violation of the parallel trend assumption despite the treatments being randomized and thus parallel trends being true in the DGP, which is a pernicious side-effect of the negative weights problem. We can still estimate the cohort-level ATTs correctly with a saturated regression. The key insight is that testing for differences between a 'pooled' event study (the blue line in the second panel) and cohort X time interactions (that yield the cohort-level estimates in the third panel) can help us distinguish between the two scenarios. This can be formulated as a joint F-test on the coefficients of the cohort X time interactions in a saturated regression. We provide a formal statement of this test in the next section, and show through simulation studies that this approach can detect cohort-level temporal heterogeneity in a variety of DGPs.

#figure(
  image("../figtab/homfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions with homogeneous treatment effects across three cohorts. Joint test p-value = 0.11
  ],
) <homfx>


#figure(
  image("../figtab/hetfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions in a DGP with heterogeneous treatment effects across three cohorts. Joint test p-value = 0.000
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
We consider two tests: one for testing for event study dynamics, and one for testing for heterogeneity in event study dynamics. These tests are both classical Wald tests for linear restrictions and are asymptotically equivalent to the Likelihood Ratio test and Lagrange Multiplier test. The test is optimal (most powerful) in the class of invariant tests for local alternatives when errors are normally distributed (@lehmann2005testing lemma 8.5.2).
#footnote[This can be implemented using either a $chi^2$ or $F$ test; the distinction between the two is due to different degrees of freedom that disappear for realistic sample sizes]

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
y_(i t) = alpha_i + lambda_t +
  underbrace(
    sum_(g_i in cal(C)\\ infinity) sum_(s != -1)^(T) bb(1)(g_i = c) tau^(s c) Delta_(i t)^s,
    "Cohort-Time Interactions")
  + epsilon_(i t)
$ <satevent>

This is a saturated event study that constructs cohort $times$ time interactions for each adoption cohort (with $g_i = infinity$ never treated cohort) omitted and therefore recovers the cohort-level event studies. These coefficients are reported in the third panel in @homfx and @hetfx, and correctly uncover the true cohort-level ATTs in the presence of arbitrary heterogeneous treatment effects across cohorts (top panel). The downside of this approach, however, are twofold. First, these regressions can get unwieldy with many cohorts, and the number of parameters grows linearly with the number of cohorts. Second, the cohort level ATTs are self-contained and therefore constructing a test for equality across multiple cohorts is not straightforward. Instead, one may re-specify the saturated event-study regression @satevent as follows:

$
  y_(i t) = alpha_i + lambda_t +
    underbrace(sum_(s != -1)^(T) gamma_s Delta_(i t)^s, "(a) Common event study coefficients")
    +
    underbrace(sum_(c in cal(C)) sum_(s != -1)^(T) delta_s Delta_(i t)^(c s), "(b) Cohort-specific deviations")
    + epsilon_(i t)
$ <jointreg>

@jointreg returns numerically identical estimates of the cohort-level dynamic ATT as @satevent, but it allows us to test for differences in dynamic treatment effects over cohorts more easily. This is because @jointreg contains a common event study coefficient vector (a), and cohort-level deviations (b). The (b) terms can be jointly tested against the null of zero, which serves as a direct test of cohort-level treatment effect heterogeneity relative to a traditional event study. This approach is similar to omnibus tests of effect heterogeneity in cross-sectional RCTs proposed by @Ding2019-nr, testing the joint null of $gamma = 0$ in the interacted regression $y ~ tau W + X beta + W X gamma + epsilon$ serves as a test for explained effect heterogeneity. We illustrate an application of this test in @respec, where the top panel reports the saturated event study @satevent, the middle panel reports the coefficients from re-specified model @jointreg, and the bottom panel reports the sum of the common event study and cohort-specific deviations, which reproduces the saturated event study estimates exactly.

#figure(
    grid(
        columns: 2,
        gutter: 1mm,
        [ #image("../figtab/respecification_verify_hom.png", width: 120%) ],
        [ #image("../figtab/respecification_verify_het.png", width: 120%) ],
    ),
    caption: [
    For each DGP (homogeneous - @homfx - on the left and heterogeneous - @hetfx - on the right), the top panel illustrates the traditional event study estimates from eqn @satevent, which are unbiased for the true effects. The middle panel plots the re-specified model, which plots an overall event study (first cohort : blue) and subsequent cohort deviations (second and third cohorts - which are null in this DGP). The final panel plots the sum of the blue and cohort-specific coefficients, which reproduces the event study coefficient from the first panel exactly.
  ]
) <respec>



We show in the next section that this test is consistent for the null hypothesis of homogeneous dynamic treatment effects over cohorts, and that it has power against a variety of alternatives. As a concrete example, the joint $p-$value for the cohort $times$ time interactions in @homfx is $0.11$, while the joint p-value for the cohort $times$ time interactions in @hetfx is $0.000$. Thus, we can reject the null hypothesis of homogeneous dynamic treatment effects in @hetfx but not in @homfx, which is consistent with the underlying DGP. In the next section, we show through simulation studies that this test has good power to detect across-cohort heterogeneity in dynamic treatment effects in a variety of DGPs.

= Simulation Studies

== Testing for event study dynamics

To begin, we perform simulation studies based on to study the properties of the testing procedure described in @test_dyn. We consider the simple setting with a single adoption cohort where the treatment effects follow one of the following seven DGPs visualised in @static_dyn.

#figure(
  image("../figtab/static_dynamic_effects.png", width: 105%),
  caption: [
    true treatment effect functions and estimates from difference in means, static, and dynamic two-way fixed effects regressions. The treatment effect is truly stationary in the first DGP and varies over time in the others.
  ],
) <static_dyn>

The first DGP has constant effects over time, while the others have varying degrees of temporal heterogeneity. We simulate 1000 replications of the data for each DGP, and compute the rejection rate of the joint test for dynamic treatment effects outlined in the previous section. We report the rejection rate and p-value distribution in @rejrates_dyn. We find that the rejection rate for the constant DGP (null) is under the nominal level of $alpha = 0.05$, while the rejection rates for the other DGPs considerably higher. The rejection rate for concave effects is considerably lower, although this is likely due to the fact that the treatment effects do actually tail off in later time periods and the static effect captures this well.

#figure(
  image("../figtab/rejection_rates_dyn.png", width: 105%),
  caption: [
    Rejection rates over 1000 replications for the joint test of dynamic treatment effects using an F-test in DGPs from @static_dyn
  ],
) <rejrates_dyn>


== Testing for across-cohort heterogeneity in dynamic treatment effects

Next, we perform simulation studies based on to study the properties of the testing procedure described in @test_het. Here, we consider seven different DGPs with homogeneous and heterogeneous treatment effect functions across cohorts as illustrated in @truefns. In addition to the two DGPs described in the previous section, we consider DGPs with heterogeneity that applies a scaler multiplier to the concave (log) effect function in @homfx with 'small' and 'large' differences; a DGP with 'selection on gains' where the cohort with the largest treatment effect adopts first; a DGP with 'novelty effects' where the treatment effect is large for the first few periods and then diminishes; and finally a DGP with 'activity bias' where the treatment effect is immediate and large for the earliest adopting cohort and much more gradual for the others. Among all these DGPs, the homogenous and novelty effects DGPs have homogeneous treatment effects across cohorts, while all others have heterogeneous treatment effects across cohorts.

For each DGP, we simulate 1000 replications of the data, and compute the rejection rate of the joint test for cohort-level coefficients outlined in the previous section. We report the rejection rate and p-value distribution in @rejrates. We find that the rejection rate for the homogeneous DGP (null) is under the nominal level of $alpha = 0.05$, while the rejection rates for heterogeneous DGPs are close to 1. This suggests that the test has good power to detect across-cohort heterogeneity in dynamic treatment effects.

#figure(
  image("../figtab/true_functions.png", width: 80%),
  caption: [
    true cohort level effect functions for homogeneous and heterogeneous treatment effects across three cohorts. Earliest-treated cohort is in purple, middle cohort in green, and latest cohort in yellow.
    'Homogenous' and 'novelty effects' DGPs have homogeneous treatment effects across cohorts, while all others have heterogeneous treatment effects across cohorts.
  ],
) <truefns>



#figure(
  image("../figtab/rejection_rates_F.png", width: 100%),
  caption: [
    Rejection rates over 1000 replications for the joint test of cohort-level coefficients using an F-test in DGPs from @truefns
  ],
) <rejrates>


=  Conclusion

The two-way fixed effects regression remains a workhorse tool in applied economics despite recent critiques highlighting its potential shortcomings under treatment effect heterogeneity. This paper provides simple diagnostic tests that help researchers determine when TWFE is likely to yield reliable estimates versus when more complex estimators are needed. Our simulation evidence shows these tests have good power to detect problematic patterns of effect heterogeneity while maintaining correct size under the null of homogeneous effects.

The tests we propose are computationally simple and implemented in the pyfixest library and readily implementable in standard statistical software. Since heterogeneity-robust estimators often come with higher variance and computational complexity, the ability to test when they are truly needed helps researchers make principled choices about their estimation strategy. While these tests cannot guarantee TWFE will recover meaningful treatment effects, they provide a practical tool for detecting scenarios where the recent critiques of TWFE are most relevant.

#import "jmlr.typ": jmlr, theorem, proof
#let authors = (
  (name: "Apoorva Lal", affl: "one", email: ""),
)
#let affls = (
  one: (
    department: "MLIR, Netflix",
  ),
)


#show: jmlr.with(
  title: [When do we need to deviate from the two-way fixed effects regression?],
  authors: (authors, affls),
  abstract: "
  The widespread use of the two-way fixed effects regression has come under scrutiny recently due to results from Chaismartin and D'Haultfoeuille (2020) and Goodman-Bacon (2021) suggesting that it fails to uncover meaningful averages of heterogeneous treatment effects under the presence of effect heterogeneity over time and across adoption cohorts. In this paper, we propose a simple test that can be used to test for differences in dynamic treatment effects over cohorts, which allows us to test for when the two-way fixed effects regression is likely to yield biased estimates of average treatment effects on the treated (ATT)s.
  ",
  keywords: ("difference in differences", "panel data", "heterogeneous treatment effects"),
  bibliography: bibliography("main.bib"),
  // appendix: include "appendix.typ",
)

// #set math.equation(numbering: "(1)")

= Introduction

The following two-way fixed effects regression

$
// attach(y, br:"it") = alpha_i + lambda_t + beta W_i t + epsilon_it
y_(i j t) = alpha_i + lambda_t + beta W_(i t) + epsilon_(i t)
$

and its dynamic ('event study') counterpart

$
y_(i t) = alpha_i + lambda_t + beta W_(i t) + sum_(s != -1)^(T) gamma_s Delta_(i t)^s + epsilon_(i t)
$

are extremely popular in applied econometrics.

= Methodology

We propose using a joint F-test on the following specification

$
y_(i t) = alpha_i + lambda_t + beta W_(i t) + sum_(s != -1)^(T) gamma_s Delta_(i t)^s + sum_(s != -1)^(T) delta_s W_(i t) Delta_(i t)^s + epsilon_(i t)
$

= Simulation Studies

#figure(
  image("../figtab/homfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions with homogeneous treatment effects across three cohorts
  ],
) <hetfx>


#figure(
  image("../figtab/hetfx.png", width: 100%),
  caption: [
    true and estimated effects from pooled and saturated event study regressions in a DGP with heterogeneous treatment effects across three cohorts
  ],
) <hetfx>



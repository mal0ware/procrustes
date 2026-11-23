# Sources, formulas and comparison

Every formula in `src/procrustes/` is listed here with the place it comes from. The formulas were written into this file from the sources before the code was written.

Notation: `Z` is the standard Normal CDF, `Z^-1` its inverse, `γ ≈ 0.5772156649` the Euler-Mascheroni constant, `e` Euler's number. Sharpe ratios are per observation (not annualised) unless a line says otherwise. Kurtosis `γ4` is the raw fourth standardised moment, so a Normal distribution has `γ4 = 3`.

## 1. Sources read

| Key | Citation | Copy read |
|---|---|---|
| BLdP12 | Bailey and López de Prado, "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2), 2012 | Authors' preprint, `davidhbailey.com/dhbpapers/sharpe-frontier.pdf`, 46 pages |
| BLdP14 | Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5), 2014 | SSRN 2460551 preprint dated 31 July 2014, `davidhbailey.com/dhbpapers/deflated-sharpe.pdf`, 22 pages |
| BBLZ14 | Bailey, Borwein, López de Prado and Zhu, "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of the AMS 61(5), May 2014, pp. 458-471 | Published version, `ams.org/notices/201405/rnoti-p458.pdf` |
| BBLZ17 | Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest Overfitting", Journal of Computational Finance 20(4), 2017 | Authors' preprint, `davidhbailey.com/dhbpapers/backtest-prob.pdf`, 34 pages |
| HLZ16 | Harvey, Liu and Zhu, "... and the Cross-Section of Expected Returns", Review of Financial Studies 29(1), 2016, pp. 5-68 | Published version from Campbell Harvey's Duke page, 64 pages |
| AFML | López de Prado, Advances in Financial Machine Learning, Wiley, 2018, chapters 7 and 12 | Not read. The book is not freely available. See section 7 for what was used instead. |

BLdP12 was not on the original reading list. BLdP14 defines the deflated Sharpe ratio as a probabilistic Sharpe ratio with a particular benchmark and refers to BLdP12 for the definition, so it had to be read too.

Page and equation numbers below refer to the copies in the table. The journal versions may number pages differently.

## 2. Probabilistic Sharpe ratio (BLdP12)

Standard error of the estimated Sharpe ratio. BLdP12 eq. (8) gives the Mertens (2002) limiting distribution, and section 2.5 gives the estimate with Bessel's correction:

```
σ(SR^) = sqrt( (1 - γ3·SR^ + (γ4 - 1)/4 · SR^²) / (n - 1) )          BLdP12 section 2.5, from eq. (8)
```

Probabilistic Sharpe ratio, the probability that the true Sharpe ratio exceeds a benchmark `SR*`:

```
PSR(SR*) = Z[ (SR^ - SR*) · sqrt(n - 1) / sqrt(1 - γ3·SR^ + (γ4 - 1)/4 · SR^²) ]      BLdP12 eq. (11)
```

Minimum track record length, the number of observations needed for `PSR(SR*)` to reach confidence `1 - α` when `SR^ > SR*`:

```
MinTRL = 1 + [1 - γ3·SR^ + (γ4 - 1)/4 · SR^²] · ( Z^-1[1 - α] / (SR^ - SR*) )²        BLdP12 eq. (13)
```

`n` is the number of return observations. `SR^`, `γ3` and `γ4` are computed at the sampling frequency of the returns. The Python listing in BLdP12 appendix A.3 agrees with eqs. (11) and (13) term by term.

## 3. Expected maximum Sharpe ratio and the deflated Sharpe ratio (BLdP14)

Expected maximum of `N` independent standard Normal draws, for large `N`:

```
E[max_N] ≈ (1 - γ) · Z^-1[1 - 1/N] + γ · Z^-1[1 - 1/(N·e)]          BLdP14 eq. (5); BBLZ14 eq. (4), Proposition 1
```

Expected maximum Sharpe ratio across `N` independent trials whose estimated Sharpe ratios have mean `E[{SR_n}]` and variance `V[{SR_n}]`:

```
E[max{SR_n}] ≈ E[{SR_n}] + sqrt(V[{SR_n}]) · E[max_N]                BLdP14 eq. (1), eq. (6)
```

Deflated Sharpe ratio:

```
DSR = PSR(SR0) = Z[ (SR^ - SR0) · sqrt(T - 1) / sqrt(1 - γ3·SR^ + (γ4 - 1)/4 · SR^²) ]     BLdP14 eq. (2)
SR0 = sqrt(V[{SR_n}]) · E[max_N]
```

`SR^`, `T`, `γ3` and `γ4` belong to the selected strategy. `SR0` is the expected maximum under the null that every true Sharpe ratio is zero, so the mean term of eq. (1) is dropped.

Trials that are not independent, BLdP14 appendix A.3. With `M` trials whose pairwise correlations are `ρ_ij`:

```
ρ^ = ( Σ_i Σ_j ρ_ij - M ) / ( M·(M - 1) )                            BLdP14 eq. (8)
N^ = ρ^ + (1 - ρ^) · M                                               BLdP14 eq. (9)
```

`N^` replaces `N` in eq. (1). Eq. (9) is a linear interpolation between `N = M` at `ρ = 0` and `N = 1` at `ρ = 1`. The paper presents it as one option and warns that an average correlation estimated from more trials than observations "may itself be overfit".

### Where the code departs from BLdP14

1. Eq. (5) is an approximation for large `N` and equals minus infinity at `N = 1`. The code returns 0 at `N = 1`, which is the exact expected maximum of one standard Normal draw. This is what makes the deflated Sharpe ratio equal `PSR(0)` for a single trial.
2. Eq. (9) can give a non-integer `N^` between 1 and 2, where eq. (5) is badly wrong (it is negative below about 1.5). For `1 < N < 2` the code interpolates linearly between 0 at `N = 1` and the eq. (5) value at `N = 2`. This is my choice and is not in the paper. The exact value at `N = 2` is `1/sqrt(π) ≈ 0.564`; eq. (5) gives 0.520.
3. `deflated_sharpe_ratio_from_returns` estimates `V[{SR_n}]` as the sample variance (`ddof=1`) of the trials' Sharpe ratios. The paper does not say which estimator to use.

## 4. Minimum backtest length (BBLZ14)

Lo's asymptotic distribution of the annualised Sharpe ratio estimated over `y` years with `q` returns per year is BBLZ14 eq. (3). With a true Sharpe ratio of zero the annualised estimate has variance `1/y`, so the expected maximum annualised Sharpe ratio among `N` skill-less trials is:

```
E[max_N] ≈ y^(-1/2) · ( (1 - γ)·Z^-1[1 - 1/N] + γ·Z^-1[1 - 1/(N·e)] )         BBLZ14 eq. (5)
```

Solving for `y` gives the minimum backtest length in years:

```
MinBTL ≈ ( ((1 - γ)·Z^-1[1 - 1/N] + γ·Z^-1[1 - 1/(N·e)]) / E[max_N] )²  <  2·ln(N) / E[max_N]²     BBLZ14 eq. (6), Theorem 2
```

Here `E[max_N]` is the annualised in-sample Sharpe ratio one wants skill-less selection to be unable to reach in expectation. The paper calls MinBTL "a necessary, nonsufficient condition to avoid overfitting".

## 5. Probability of backtest overfitting (BBLZ17)

Definition 2.2, eq. (2.2): PBO is the probability that the configuration ranked best in sample ranks below the median out of sample.

Combinatorially symmetric cross-validation, Algorithm 2.3:

1. Form the `T × N` matrix `M` of returns, one column per trial.
2. Split the rows into an even number `S` of disjoint blocks of equal size, in time order.
3. Form every combination of `S/2` blocks. There are `C(S, S/2)` of them (eq. 2.3).
4. For each combination `c`: the chosen blocks, in their original order, are the in-sample set `J`; the remaining blocks are the out-of-sample set. Compute the performance statistic of every column on each set. Let `n*` be the column that is best in sample and `r̄` its out-of-sample rank among the `N` columns, with rank 1 the worst and rank `N` the best. Then

```
ω̄_c = r̄ / (N + 1)                    step 4(f)
λ_c = ln( ω̄_c / (1 - ω̄_c) )           step 4(g)
```

5. The relative frequency of the logits across all combinations is `f(λ)` (eq. 2.4), and

```
PBO = φ = ∫_{-∞}^{0} f(λ) dλ           section 3.1
```

Performance degradation, section 3.2: regress the out-of-sample statistic of the selected column on its in-sample statistic across combinations, `R̄_{n*} = α + β·R_{n*} + ε`. The probability of loss is the share of combinations in which the selected column's out-of-sample statistic is negative.

### Where the code departs from BBLZ17

1. The paper's integral does not say what to do with a logit of exactly zero, which happens when `N` is odd and the selected column lands exactly on the median. The code counts such a combination as half. Under pure noise the expected value of the estimate is then exactly 0.5 for any `N`.
2. Ties in the out-of-sample statistic get the average rank. The paper does not discuss ties.
3. If `T` is not a multiple of `S` the code drops the earliest `T mod S` rows. The paper requires equal blocks and does not say which rows to drop.
4. Stochastic dominance (section 3.3) is not implemented.
5. The paper says `S = 16` gives "12,780" combinations (pages 11 and 22 of the preprint). `C(16, 8)` is 12,870. The paper's other counts, 924 for `S = 12` and 2,704,156 for `S = 24`, are correct, so this looks like transposed digits.

## 6. Multiple-testing adjustments (HLZ16)

HLZ16 section 4.4 (the PDF I read numbers the subsections 3.4.1 to 3.4.3 in their headings and 4.4.1 to 4.4.3 in its cross-references). `M` tests, p-values `p_i`, ordered `p_(1) ≤ ... ≤ p_(M)`.

```
Bonferroni:   p_i^adj   = min[ M·p_i, 1 ]                                            section 4.4.1
Holm:         p_(i)^adj = min[ max_{j ≤ i} { (M - j + 1)·p_(j) }, 1 ]                section 4.4.2
BHY:          reject H_(1)..H_(k), k the largest index with p_(b) ≤ b·α_d / (M·c(M))  section 4.4.3
              c(M) = Σ_{j=1}^{M} 1/j                                                 HLZ16 eq. (1)
```

Holm rejects `H_(1)..H_(k-1)` where `k` is the smallest index with `p_(b) > α_w / (M + 1 - b)`. Bonferroni and Holm control the family-wise error rate. BHY controls the false discovery rate under any dependence when `c(M)` is the harmonic sum, and `c(M) = 1` gives Benjamini-Hochberg, valid under independence or positive dependence.

HLZ16 applies these to t-statistics by converting each to a two-sided p-value. Table 4 is consistent with the Normal distribution for that conversion (`t = 1.99` gives 4.66%).

### Where the code departs from HLZ16

1. HLZ16 writes the BHY adjusted p-value recursively as `p_(M)^adj = p_(M)` and `p_(i)^adj = min[ p_(i+1)^adj, (M·c(M)/i)·p_(i) ]`. The first line is inconsistent with the paper's own rejection rule, which rejects `H_(M)` only when `p_(M) ≤ α_d / c(M)`. The code starts the recursion at `min[ c(M)·p_(M), 1 ]`, which is the Benjamini-Yekutieli adjusted p-value and agrees with the rejection rule for every index.
2. The code reports a t-statistic hurdle. For Bonferroni it is the t-ratio whose p-value is `α/M`. For Holm and BHY it is the smallest absolute t-statistic among the rejected tests, which matches how Table 4 reports cutoffs ("for Holm is 0.60%, and for BHY is 0.85%" are the largest rejected p-values).

## 7. Purged k-fold, embargo and combinatorial purged cross-validation (AFML, secondary sources)

I could not read the book. This section is implemented from public descriptions of chapters 7 and 12:

- The documentation of `skfolio.model_selection.CombinatorialPurgedCV` (skfolio.org), which defines purging as "removing from the training set all observations whose labels overlapped in time with those labels included in the testing set" and embargoing as "removing from the training set all observations that immediately follow an observation in the testing set", and cites AFML.
- The EverybodyWiki article "Combinatorial Purged Cross-Validation" (a copy of a former Wikipedia article), which states the split count `C(N, k)`, the path count `φ[N, k] = (k/N)·C(N, k)`, and the worked case `N = 6, k = 2` with 15 splits and 5 paths.
- A Towards AI article, "The Combinatorial Purged Cross-Validation method", which walks through the same `N = 6, k = 2` case and the assignment of test groups to paths.

No code was copied from any of them or from any other implementation.

What is implemented:

```
Each sample i has a label interval [t0_i, t1_i]. Samples are ordered by t0.

Purging:   drop training sample j if [t0_j, t1_j] intersects [t0_i, t1_i] for any test sample i.
Embargo:   after each contiguous test block, also drop the next h training samples that start
           after the block's last label ends. h is a count, or a fraction of the sample size.

Purged k-fold:        k contiguous test folds, one per split.
Combinatorial purged: N contiguous groups, every combination of k groups is a test set.
                      splits = C(N, k)
                      paths  = φ[N, k] = (k/N)·C(N, k) = C(N - 1, k - 1)
```

Each group is in the test set of `C(N - 1, k - 1)` splits. Path `p` takes, for each group, the forecasts from the `p`-th split (in lexicographic order of combinations) that has the group in its test set. Every path then covers every sample once with an out-of-sample forecast.

Points I could not check against the book:

1. Intervals are treated as closed, so a training label that ends at the instant a test label starts counts as overlapping. My recollection is that the book states three overlap conditions with weak inequalities, which is the same thing, but its reference code keeps a training label that ends exactly at the test start. Closed intervals are the conservative reading.
2. The embargo is applied only after a test block, as both secondary sources describe. The embargo size is in samples. My recollection is that the book suggests about 1% of the sample; no default is taken from it here (the default embargo is zero).
3. The rule for which split feeds which path is the one shown in the secondary sources' `N = 6, k = 2` table. Any rule that uses each (split, group) pair once gives the same set of out-of-sample forecasts, arranged into paths differently.

## 8. Comparison with the sources

`scripts/reproduce.py` writes every worked number I found in the sources next to the value this library computes, in `results/paper_examples.md`. The tests in `tests/` assert the same values.

Thirty quantities are compared. Twenty-seven agree to the precision the sources print. Three do not:

| Source | Quantity | Paper | Here | Likely reason |
|---|---|---|---|---|
| BLdP14 Exhibit 3.1 | Error of the expected-maximum formula at N = 1,000 | "as small as 0.006" | 0.0137 | The paper measured the error against a simulation of 10,000 draws per cell. I compared with the exact integral of `x·N·φ(x)·Φ(x)^(N-1)`. At N = 1,000 the maximum has a standard deviation of about 0.35, so the paper's simulated mean carries a standard error near 0.0035 and a reading of 0.006 is within about two standard errors of 0.0137. The paper's other claim, an error under 0.05 for small N, holds: 0.036 at N = 10 and 0.023 at N = 100. |
| BBLZ17 eq. (2.3) | Number of CSCV splits at S = 16 | 12,780 | 12,870 | Transposed digits in the paper. `C(16, 8) = 12,870`. |
| HLZ16 Table 4 | Holm discoveries among the ten example tests | 4 | 6 when computed from the t-statistics, 4 when computed from the printed p-values | Test 6 has `t = 2.64`. The table prints its p-value as 0.84%, which is above Holm's fifth threshold of 0.8333%, so Holm stops there. The two-sided Normal p-value of 2.64 is 0.829%, which is below the threshold, so Holm continues and also rejects tests 6 and 2. The printed p-values were presumably computed from unrounded t-statistics. Bonferroni (3) and BHY (6) agree either way. |

The approximation error in the first row matters in practice only slightly. It makes `SR0` a little too high, so the deflated Sharpe ratio is slightly conservative. In the simulation below the mean best Sharpe ratio among 1,000 skill-less strategies was 1.451 and the mean `SR0` was 1.457.

Formulas whose statement in the source is ambiguous or inconsistent, and what the code does, are listed under "Where the code departs" in sections 3, 5 and 6.

Not compared, because the sources give no number to compare with: the effective number of trials (BLdP14 eq. 9), the performance degradation regression, and everything in section 7. BBLZ17's own PBO examples run on a simulated seasonal strategy whose random series is not published, so their 55% cannot be reproduced.

## 9. Simulations run here

All from `scripts/reproduce.py` with fixed seeds. Returns are independent Normal draws.

### Sampling distribution of the PBO estimate (`results/pbo_sampling.csv`)

50 strategies, 1,000 observations, 10 blocks (252 splits), 300 datasets per row. Strategy 0 has the per-observation Sharpe ratio shown; the others have none.

| Sharpe ratio of strategy 0 | Mean PBO | Standard deviation | 2.5% to 97.5% of estimates |
|---|---|---|---|
| 0 | 0.493 | 0.181 | 0.13 to 0.84 |
| 0.1 | 0.275 | 0.178 | 0.02 to 0.68 |
| 0.2 | 0.013 | 0.029 | 0.00 to 0.11 |
| 0.3 | 0.0001 | 0.0007 | 0.00 to 0.00 |

BBLZ17 section 4 says that when the backtest carries no information the logits are roughly standard Normal, which implies a PBO near 0.5. The mean here is 0.493 with a standard error of 0.010, which agrees. A single estimate is another matter: its standard deviation is 0.18, because the 252 splits reuse the same data and are far from independent. BBLZ17 section 4 computes the precision of PBO as if the splits were independent draws ("σ[f(λ)] < 0.0045" for S = 16). That is the precision of the count given the dataset. It is much smaller than the variation from one dataset to another.

These numbers set the test tolerances in `tests/test_pbo.py`:

- Pure noise: the mean of 200 estimates must be within 0.05 of 0.5. The standard error of that mean is `0.18/sqrt(200) = 0.013`, so the tolerance is about four standard errors.
- Real drift: with a per-observation Sharpe ratio of 0.3 every one of 50 estimates must be below 0.02. In the table above the largest of 300 was 0.012 and 299 were zero. With the test's seed the largest of the 50 is 0.004 and the mean of the 200 noise estimates is 0.511.

### The 1,000-strategy experiment (`results/headline.csv`, `results/monte_carlo.csv`)

1,000 strategies, 1,260 daily observations, daily volatility 1%, 252 days per year. One seed for the headline tables, then 200 fresh datasets for each of four cases in which strategy 0 has a true annualised Sharpe ratio of 0, 1, 2 or 3 and the other 999 have none. In each dataset the strategy with the highest full-sample Sharpe ratio is selected and every tool is applied to it. CSCV uses 16 blocks. CPCV uses 6 groups, 2 test groups and a 5-day embargo, and its "model" is the rule "trade the strategy with the best Sharpe ratio on the training set".

| True Sharpe of strategy 0 | Mean Sharpe of the winner | Winner is strategy 0 | Single test p < 0.05 | PSR(0) > 0.95 | DSR > 0.95 | Bonferroni p < 0.05 | BHY p < 0.05 | PBO mean (sd) | PBO < 0.05 | Mean CPCV path Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.45 | 0% | 100% | 100% | 0% | 2% | 0% | 0.50 (0.10) | 0% | 0.02 |
| 1 | 1.52 | 20% | 100% | 100% | 0% | 8% | 2% | 0.47 (0.12) | 0% | 0.16 |
| 2 | 2.01 | 89% | 100% | 100% | 28% | 70% | 49% | 0.27 (0.14) | 4% | 1.26 |
| 3 | 2.99 | 100% | 100% | 100% | 96% | 100% | 100% | 0.04 (0.06) | 71% | 2.93 |

Percentages are shares of the 200 datasets. With 200 datasets a share near 50% has a standard error of about 3.5 points. Holm gave the same shares as Bonferroni in every row.

## 10. Variants tried

The standard for this repository asks for a count of everything that was tried, so that the results cannot have been tuned.

- The headline experiment was run with one configuration (1,000 strategies, five years of daily returns, seed 20261123, control strategy with a true Sharpe ratio of 2). The configuration was fixed before the first run and never changed. No other seed was tried.
- The Monte Carlo levels (0, 1, 2, 3) and the decision thresholds (0.95 for PSR and DSR, 0.05 for p-values and for PBO) were fixed before the first run.
- The PBO sampling configuration (50 strategies, 1,000 observations, 10 blocks) was the first one tried. It was run once informally with a different seed before the committed run; the two agreed (mean 0.517 and 0.493 under noise).
- Two tests were changed after failing on first run. A check of the expected-maximum formula against simulation had tolerances taken from the paper's Exhibit 3.1 and failed at N = 100; it was replaced with a comparison against the exact integral, which is how the first mismatch in section 8 was found. An input-validation test for `cscv` used an input that is in fact valid.
- Figures were restyled once (a clipped title and a legend that covered the bars). No data changed.
- `scripts/reproduce.py` was run in full three times. The tables were identical each time.

# Sources, formulas and comparison

Every formula in `src/procrustes/` is listed here with the place it comes from. The formulas went into this file in the first commit after the skeleton (6fe1b9e), before the library code (dc1f226).

Notation: `Z` is the standard Normal CDF, `Z^-1` its inverse, `γ ≈ 0.5772156649` the Euler-Mascheroni constant, `e` Euler's number. Sharpe ratios are per observation (not annualised) unless a line says otherwise. Kurtosis `γ4` is the raw fourth standardised moment, so a Normal distribution has `γ4 = 3`.

## 1. Sources

| Key | Citation | Copy used |
|---|---|---|
| BLdP12 | Bailey and López de Prado, "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2), 2012 | Authors' preprint, `davidhbailey.com/dhbpapers/sharpe-frontier.pdf`, 46 pages |
| BLdP14 | Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5), 2014 | SSRN 2460551 preprint dated 31 July 2014, `davidhbailey.com/dhbpapers/deflated-sharpe.pdf`, 22 pages |
| BBLZ14 | Bailey, Borwein, López de Prado and Zhu, "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of the AMS 61(5), May 2014, pp. 458-471 | Published version, `ams.org/notices/201405/rnoti-p458.pdf` |
| BBLZ17 | Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest Overfitting", Journal of Computational Finance 20(4), 2017 | Authors' preprint, `davidhbailey.com/dhbpapers/backtest-prob.pdf`, 34 pages |
| HLZ16 | Harvey, Liu and Zhu, "... and the Cross-Section of Expected Returns", Review of Financial Studies 29(1), 2016, pp. 5-68 | Published version from Campbell Harvey's Duke page, 64 pages |
| AFML | López de Prado, Advances in Financial Machine Learning, Wiley, 2018, chapters 7 and 12 | Not read. The cross-validation code is my implementation of the published idea. See section 7. |

BLdP14 defines the deflated Sharpe ratio as a probabilistic Sharpe ratio with a particular benchmark and refers to BLdP12 for the definition, which is why BLdP12 is here.

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

## 7. Purged k-fold, embargo and combinatorial purged cross-validation (AFML)

I have not read the book. The code is my implementation of the published idea: drop from the training set every sample whose label period overlaps a test label (purging), and also drop the samples just after each test block (embargo); in the combinatorial version, test every combination of k out of N contiguous groups and rebuild full-length out-of-sample paths. No code was copied from any implementation.

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

Choices I made that the book may make differently:

1. Intervals are closed, so a training label that ends at the instant a test label starts counts as overlapping. This is the conservative reading.
2. The embargo is applied only after a test block. Its size is in samples. The default is zero.
3. The rule for which split feeds which path is the lexicographic one above. Any rule that uses each (split, group) pair once gives the same set of out-of-sample forecasts, arranged into paths differently.

## 8. Comparison with the sources

`scripts/reproduce.py` writes every worked number I found in the sources next to the value this library computes, in `results/paper_examples.md`. The tests in `tests/` assert the same values.

Twenty-eight quantities are compared. Twenty-five agree to the precision the sources print. Three do not:

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

1,000 strategies, 1,260 daily observations, daily volatility 1%, 252 days per year. One seed for the headline tables, then 200 fresh datasets for each of four cases in which strategy 0 has a true annualised Sharpe ratio of 0, 1, 2 or 3 and the other 999 have none. In each dataset the strategy with the highest full-sample Sharpe ratio is selected and every tool is applied to it. CSCV uses 16 blocks. Combinatorial CV uses 6 groups, 2 test groups and a 5-day embargo, and its "model" is the rule "trade the strategy with the best Sharpe ratio on the training set". Each label is one day's return, so no labels overlap and purging removes nothing in this experiment; section 10 has one where it can.

| True Sharpe of strategy 0 | Mean Sharpe of the winner | Winner is strategy 0 | Single test p < 0.05 | PSR(0) > 0.95 | DSR > 0.95 | Bonferroni p < 0.05 | BHY p < 0.05 | PBO mean (sd) | PBO < 0.05 | Mean combinatorial CV path Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.45 | 0% | 100% | 100% | 0% | 2% | 0% | 0.50 (0.10) | 0% | 0.02 |
| 1 | 1.52 | 20% | 100% | 100% | 0% | 8% | 2% | 0.47 (0.12) | 0% | 0.16 |
| 2 | 2.01 | 89% | 100% | 100% | 28% | 70% | 49% | 0.27 (0.14) | 4% | 1.26 |
| 3 | 2.99 | 100% | 100% | 100% | 96% | 100% | 100% | 0.04 (0.06) | 71% | 2.93 |

Percentages are shares of the 200 datasets. With 200 datasets a share near 50% has a standard error of about 3.5 points. Holm gave the same shares as Bonferroni in every row.

DSR > 0.95 passed in 0 of 200 datasets under the null, so it is not a 5%-level test. To pass, the winner must beat the expected maximum of the skill-less trials by 1.645 of its own standard errors. The best of many skill-less trials rarely lands that far above its expected value, because the maximum of many draws varies much less than a single draw does.

## 10. Real-data experiments

Data: the daily market excess return (`Mkt-RF`, in percent with two decimals) from the "Fama/French 3 Factors [Daily]" file of Kenneth French's data library, 1926-07-01 to 2025-12-31, 26,151 days. The file used was built from the 202609 CRSP release. `results/data_fingerprint.csv` records a hash of the series.

### A correlated family of trend rules (`results/family*.csv`, `results/family.md`)

Rules: for each pair (f, s) with f in {1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100}, s in {10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300} and f < s, compare the f-day and s-day trailing means of the log excess-return index at the close of day t. Hold +1 if the fast mean is higher, -1 if lower, and earn `position × Mkt-RF` on day t+1. 118 rules, no costs.

Windows: 20 non-overlapping windows of 1,260 return days, counted back from 2025-12-31, each with 300 days of history before it for the signals. The first starts 1929-09-10.

Null: multiply every daily return in the window and its history by an independent random sign (+1 or -1 with equal probability), rebuild the index and re-run all 118 rules. Let `ε_t` be the sign and `x_t` the return. A rule's return on day t+1 is `p_t · ε_{t+1} · x_{t+1}`, where `p_t` depends only on `ε_1 x_1, ..., ε_t x_t`. Since `ε_{t+1}` is independent of those and of `x_{t+1}`, the expectation is zero for every rule. 500 datasets per window, 10,000 in all.

Power: in each null dataset one rule, chosen at random, gets a constant daily drift of `S/sqrt(252)` of its own standard deviation, for S = 1, 2, 3. A constant shift changes that rule's Sharpe ratio by exactly `S/sqrt(252)` and leaves its skewness, kurtosis and correlations unchanged, so the DSR can be recomputed without re-running the rules (a test checks this against adding the drift to the returns).

In every dataset the best rule's DSR is computed twice, with N = 118 and with `N^` from BLdP14 eq. (9) using that dataset's average pairwise correlation. The third threshold is the 95% quantile of the null DSR (with N = 118) from the first 250 datasets of each window, applied to the remaining 250.

| Planted true Sharpe (annualised) | Planted rule is the best | DSR > 0.95 with N | DSR > 0.95 with N^ | Threshold from the null |
|---|---|---|---|---|
| none | | 0.7% | 1.2% | 5.3% |
| 1 | 79% | 10.8% | 15.0% | 32.1% |
| 2 | 100% | 76.7% | 81.2% | 91.3% |
| 3 | 100% | 98.8% | 99.2% | 99.8% |

Across windows the average correlation of the rules under the null was 0.48 to 0.54 and `N^` was 55 to 62. "Implied N" in `results/family.md` is the N for which `sqrt(V)·E[max_N]` equals the mean best Sharpe ratio under the null in that window. It was 41 to 63. So eq. (9) gives roughly the right count for this family. The DSR still falls short of a 5% rate, and the reason is how its threshold is built (section 9). With N = 118 the threshold sits above the mean null maximum in every window; with `N^` it sits close to it.

On the real data the best rule's DSR passed 0.95 in five windows with N (1942-46, 1951-55, 1960-65, 1965-71, 1971-76) and six with `N^` (adding 1955-60 at 0.951). In each the best rule was the price against its 10-day mean. No window after 1985 passed. Over all 20 windows together the 10-day rule had an annualised Sharpe ratio of 0.76 and a DSR of 1.0000. `results/family.md` lists the lag-1 autocorrelation of the market return in each window next to these results.

### Leakage through overlapping labels (`results/leakage.csv`, `results/leakage.md`)

Sample: daily data from 1990 to 2025-12-31. A sample is a day with 250 days of history and 20 days of label after it, 8,798 samples. Label: the sum of the next 20 days' `Mkt-RF`. Features: trailing sums over 5, 20, 60 and 250 days and the trailing 20-day standard deviation, standardised on each training set. Model: k-nearest-neighbour regression with k = 25 (scikit-learn). Splits: shuffled 5-fold; combinatorial CV with 6 groups and 2 test groups and no purging; the same with label spans `[i+1, i+20]` and a 20-day embargo. Score: the correlation between forecast and label, on each of the 5 rebuilt paths and on the test days within 20 days of a training group. Run on the real series and on 20 sign-randomised copies.

| Correlation of forecast and label | Real | Sign-randomised, mean (standard error) |
|---|---|---|
| Shuffled 5-fold | 0.411 | 0.458 (0.006) |
| Combinatorial, no purging, whole paths | 0.049 | -0.008 (0.013) |
| Combinatorial, purged and embargoed, whole paths | 0.050 | -0.007 (0.013) |
| No purging minus purged, whole paths | | -0.0009 (0.0011) |
| No purging minus purged, test days near a training group | | 0.015 (0.010) |

Purging and the embargo removed 77 of about 5,865 training samples per split. With contiguous groups the leak is confined to the edges of each test group, and on these data I could not separate it from noise. Shuffling puts every test day's neighbours in the training set, and the leak is then large.

## 11. Variants tried

- The simulated headline experiment uses one configuration (1,000 strategies, five years of daily returns, seed 20261123, control strategy with a true Sharpe ratio of 2). I report the one seed I ran.
- The Monte Carlo levels are 0, 1, 2 and 3, and the thresholds are 0.95 for PSR and DSR and 0.05 for p-values and PBO.
- The PBO sampling configuration (50 strategies, 1,000 observations, 10 blocks) was run once informally with a different seed before the committed run; the two agreed (mean 0.517 and 0.493 under noise).
- Two library tests were changed after failing on first run. A check of the expected-maximum formula against simulation had tolerances taken from the paper's Exhibit 3.1 and failed at N = 100; it was replaced with a comparison against the exact integral, which is how the first mismatch in section 8 was found. An input-validation test for `cscv` used an input that is in fact valid.
- The two worked numbers for combinatorial CV (15 splits and 5 paths at N = 6, k = 2) were dropped from the comparison table in section 8, since they came from descriptions of a book I have not read. The tests still check them as plain combinatorics.
- Rule family: one grid, one window length, 500 datasets per window and planted levels 1, 2 and 3. Before the full run I timed one window with 20 datasets. I ran the full experiment twice; the second run added the lag-1 autocorrelation column, which I added after seeing which rule won in the early windows (post hoc). The other numbers did not change.
- Leakage: before the committed run I tried the design on 3 and then 20 sign-randomised copies with another seed. The first look showed no whole-path difference from purging, and I then added the edge-day score and the shuffled k-fold comparison (both post hoc). In that trial the edge-day difference was 0.035 with a standard error of 0.012; in the committed run it is 0.015 (0.010). I report the committed run.
- A test of the sign-randomised null first used a series with daily autocorrelation 0.3, on which the rules' edge was too weak to clear the test's own bar; it now uses a persistent drift.
- Figures were restyled after the first run (clipped titles, a legend that covered the bars). No data changed.
- The final code was run in full twice and both runs wrote identical tables.

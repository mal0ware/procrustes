# Procrustes

**Tools that tell you when a backtest was stretched to fit.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

You tried 200 versions of a strategy and the best one has a Sharpe ratio of 2. How much of that is skill, and how much is the fact that you tried 200?

## What this will contain

- Probabilistic and deflated Sharpe ratio, with the expected maximum Sharpe under N independent trials
- Minimum backtest length for a given number of trials
- Probability of backtest overfitting by combinatorially symmetric cross-validation
- Purged k-fold cross-validation with an embargo
- Combinatorial purged cross-validation, with the backtest paths rebuilt
- Multiple-testing haircuts: Bonferroni, Holm, and Benjamini-Hochberg-Yekutieli

## The result it is built to show

A notebook that generates 1,000 strategies with no edge, picks the best, and shows the raw Sharpe ratio saying "great" while every tool here says "noise".

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5), 2014
- Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest Overfitting", Journal of Computational Finance 20(4), 2017
- Harvey, Liu and Zhu, "... and the Cross-Section of Expected Returns", Review of Financial Studies 29(1), 2016
- López de Prado, Advances in Financial Machine Learning, Wiley, 2018 (purging, embargo, combinatorial purged CV)

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

None. Everything runs on simulated returns or on a returns matrix you supply.

## How it is tested

The maths is tested before any result is trusted.

- The deflated Sharpe ratio equals the probabilistic one when there is a single trial
- PBO sits near 0.5 on pure noise and near 0 when one strategy has real drift
- Under random label spans, no training interval ever overlaps a test interval
- The number of combinatorial paths equals (k/N) times C(N, k)

## Layout

```
src/procrustes/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Probabilistic and deflated Sharpe ratio, with the expected maximum Sharpe under N independent trials
- [ ] Minimum backtest length for a given number of trials
- [ ] Probability of backtest overfitting by combinatorially symmetric cross-validation
- [ ] Purged k-fold cross-validation with an embargo
- [ ] Combinatorial purged cross-validation, with the backtest paths rebuilt
- [ ] Multiple-testing haircuts: Bonferroni, Holm, and Benjamini-Hochberg-Yekutieli
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- These tools assume you counted your trials honestly. They cannot find trials you forgot.
- The deflated Sharpe ratio needs an estimate of how correlated the trials were. That estimate is a judgement.
- Why another one: the best-known library for this is no longer open source, and the other I found is AGPL. This one is MIT and small.
- This is research code. It is not investment advice and it does not trade.

## The name

Procrustes made every guest fit his bed, by stretching them or cutting them down. An overfitted backtest does the same thing to data.

## Licence

MIT. See `LICENSE`.

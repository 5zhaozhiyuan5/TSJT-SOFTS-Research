# PMME Stage-II: Target-only Low-Rank Residual Adapter

## Scope

This version replaces the previous target-only temporal convolution and
Backcast-Forecast residual network. Stage-I SOFTS architecture, source-city
training, and Stage-I checkpoint format remain unchanged.

Stage-II uses only the target city's three-day support split. It does not use
source-city residual memory, graph propagation, adjacency matrices, or held-out
test labels.

## Adapter design

For each target node/window, the adapter builds a 40-dimensional feature:

- 12 segment means from the 288-step RevIN-normalized history;
- 12 segment standard deviations;
- 12 segment endpoint slopes;
- 4 summary statistics from the frozen Stage-I 36-step forecast.

The 36-dimensional target residual is defined in the same per-window RevIN
space as the correction:

\[
R = Y_{\mathrm{RevIN}} - \hat Y_{\mathrm{base,RevIN}}.
\]

The support residuals are centered by their mean trajectory. A mixed basis is
then constructed:

- 3 fixed low-frequency DCT-II directions;
- 3 data-driven detail directions from the SVD of the residual after removing
  its projection onto the smooth basis.

The default rank is 6. The adapter predicts the six basis coefficients from
the 40-dimensional feature using ridge regression with an unpenalized
intercept. The reconstructed 36-step correction is added to the frozen
Stage-I forecast.

\[
\hat R = \mu_R + \hat C B^\top,
\qquad
\hat C = [\operatorname{Norm}(Z),1]W.
\]

## Runtime

The adapter is fitted once in closed form. There is no Stage-II epoch loop and
no Stage-II backpropagation. The frozen SOFTS backbone is evaluated once per
support batch during adapter fitting; test-time inference evaluates SOFTS once
per test batch and applies a small feature transform and matrix multiplication.

The legacy finetune_epochs argument remains as an enable/disable flag for
launcher compatibility: values greater than zero enable one closed-form fit;
zero skips Stage-II. The provided launchers use finetune_epochs=1.

## Normalization protocol

For target-city support and held-out test datasets, speed mean and standard
deviation are fitted only on the first 288 * target_days time steps. The
held-out test split reuses these support-prefix statistics. Source-city
normalization continues to use each source city's own available training
series.

## Logged diagnostics

The one-shot fit writes support-only diagnostics to log.txt:

- support row count, feature dimension, total/smooth/detail rank;
- ridge coefficient and residual-basis captured-energy ratios;
- residual loss, forecast loss, correction penalty;
- support MSE, MAE, RMSE, MAPE and fit time.

Support metrics are diagnostics only; they are not held-out test performance.
The held-out target test split is constructed and evaluated only after fitting.

## Initial defaults

- history length: 288;
- forecast length: 36;
- feature bins: 12;
- residual rank: 6;
- smooth basis rank: 3;
- ridge alpha: 10.0.

These are starting settings for controlled experiments, not claims of
optimality. The low-rank assumption must be checked through the captured
residual energy and by comparing held-out metrics against Stage-I-only SOFTS.

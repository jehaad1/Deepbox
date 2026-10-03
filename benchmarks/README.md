# Deepbox Benchmark Suite

Website: https://deepbox.dev. Docs: https://deepbox.dev/docs.

These benchmarks time Deepbox (TypeScript) and the Python libraries it replaces: Pandas, NumPy, SciPy, scikit-learn, PyTorch and Matplotlib. Each Deepbox case has a matching Python case with the same name and size label, and `compare.ts` pairs them by that name.

## Quick Start

```bash
# Run all Deepbox benchmarks
npm run bench:deepbox

# Run all Python benchmarks
npm run bench:python

# Run everything and generate the comparison
npm run bench:all
```

## Individual Benchmarks

### Deepbox (TypeScript)

```bash
npm run bench:dataframe    # 01: DataFrame operations    (vs Pandas)
npm run bench:datasets     # 02: Dataset loading         (vs scikit-learn)
npm run bench:linalg       # 03: Linear algebra          (vs NumPy/SciPy)
npm run bench:metrics      # 04: ML metrics              (vs scikit-learn)
npm run bench:ml           # 05: ML training & inference (vs scikit-learn)
npm run bench:ndarray      # 06: NDArray / tensor ops    (vs NumPy)
npm run bench:nn           # 07: Neural networks         (vs PyTorch)
npm run bench:optim        # 08: Optimizers & schedulers (vs PyTorch)
npm run bench:plot         # 09: Plotting / SVG render   (vs Matplotlib)
npm run bench:preprocess   # 10: Preprocessing           (vs scikit-learn)
npm run bench:random       # 11: Random generation       (vs NumPy)
npm run bench:stats        # 12: Statistical analysis    (vs SciPy)
npm run bench:core         # 13: Core runtime utilities  (Deepbox-only, local)
npm run bench:tensor       # 14: Extra tensor / ndarray ops (Deepbox-only, writes deepbox-tensor.json, not part of bench:compare)
```

### Python

```bash
pip3 install -r benchmarks/requirements.txt

python3 benchmarks/python/01_dataframe.py
python3 benchmarks/python/02_datasets.py
python3 benchmarks/python/03_linalg.py
python3 benchmarks/python/04_metrics.py
python3 benchmarks/python/05_ml.py
python3 benchmarks/python/06_ndarray.py
python3 benchmarks/python/07_nn.py
python3 benchmarks/python/08_optim.py
python3 benchmarks/python/09_plot.py
python3 benchmarks/python/10_preprocess.py
python3 benchmarks/python/11_random.py
python3 benchmarks/python/12_stats.py
```

### Compare Results

```bash
npm run bench:compare      # writes RESULTS.md and updates the Performance section of the root README.md
```

See [RESULTS.md](./RESULTS.md) for the full comparison, with the winner for each operation.

`bench:compare` also rewrites the Performance section of the root `README.md`, so check `git diff README.md` after running it.

## What's Benchmarked

| # | Benchmark | Deepbox Module | Python Library | Key Operations |
| --- | --- | --- | --- | --- |
| 01 | DataFrame | `deepbox/dataframe` | Pandas | create, select, filter, sort, groupBy (sum/mean/count/min/max), head, tail, iloc, join, concat, fillna, dropna, describe, corr, drop |
| 02 | Datasets | `deepbox/datasets` | scikit-learn | loadIris, loadBreastCancer, loadDiabetes, loadDigits, loadWine, loadLinnerud, makeBlobs, makeCircles, makeMoons, makeClassification, makeRegression, makeGaussianQuantiles, Friedman/Swiss-roll style generators; local-only coverage also records DataLoader iteration, dataset splits/transforms, samplers, and CSV parsing |
| 03 | Linalg | `deepbox/linalg` | NumPy/SciPy | det, trace, norm, cond, matrixRank, slogdet, inv, pinv, svd, qr, lu, cholesky, eigvalsh, solve, solveTriangular, lstsq, matmul |
| 04 | Metrics | `deepbox/metrics` | scikit-learn | accuracy, precision, recall, f1, fbeta, confusionMatrix, hamming, jaccard, cohen kappa, matthews, balanced accuracy, logLoss, rocAuc, averagePrecision, mse, rmse, mae, r2, adjustedR2, explainedVariance, maxError, medianAbsError, mape, silhouette, calinski-harabasz, davies-bouldin, adjusted rand/mutual info, homogeneity, completeness, v-measure, fowlkes-mallows |
| 05 | ML | `deepbox/ml` | scikit-learn | LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet, LogisticRegression, LDA, GaussianNB, BernoulliNB, MultinomialNB, KNN/radius/centroid classifiers, LinearSVC/SVR, SGD (C+R), DecisionTree (C+R), RandomForest (C+R), ExtraTrees, AdaBoost, Bagging, Voting, Stacking, KMeans, MiniBatchKMeans, DBSCAN, Birch, MeanShift, OPTICS, GaussianMixture, SpectralClustering, PCA, TSNE, GaussianRandomProjection |
| 06 | NDArray | `deepbox/ndarray` | NumPy | creation (zeros, ones, full, empty, arange, linspace, eye, randn), arithmetic (add, sub, mul, div, neg, pow), math (sqrt, exp, log, abs, sin, cos, clip, sign), reductions (sum, mean, max, min, variance, std, prod, median, cumsum, cumprod), sort, argsort, reshape, flatten, transpose, squeeze, unsqueeze, concatenate, stack, slice, comparison, logical, activations (relu, sigmoid, tanh, softmax), matmul |
| 07 | NN | `deepbox/nn` | PyTorch | layer creation (Linear, Sequential, Conv1d/2d, RNN, LSTM, GRU, BatchNorm1d, LayerNorm), forward pass, activations (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU, Swish, Mish, Softmax), forward+backward, losses (mse, mae, rmse, huber, crossEntropy, bce), training loops (Adam, SGD), inference (noGrad), module ops |
| 08 | Optim | `deepbox/optim` | PyTorch | optimizer creation (SGD, Adam, AdamW, Adagrad, AdaDelta, Nadam, RMSprop), step, training loops per optimizer, LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, OneCycleLR, ReduceLROnPlateau, WarmupLR), stateDict, zeroGrad |
| 09 | Plot | `deepbox/plot` | Matplotlib | scatter, line plot, bar, stacked/grouped bar, barh, hist, boxplot, violinplot, pie, heatmap, imshow, contour, contourf, plotConfusionMatrix, plotRocCurve, plotPrecisionRecallCurve, plotLearningCurve, plotValidationCurve, kdeplot, stem, quiver, polar, surface, dendrogram, SVG/PNG/PDF rendering; local-only coverage also records strip, radar, and waterfall plots |
| 10 | Preprocess | `deepbox/preprocess` | scikit-learn | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer, KBinsDiscretizer, SplineTransformer, SimpleImputer, KNNImputer, MissingIndicator, Binarizer, PolynomialFeatures, VarianceThreshold, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, CountVectorizer, TfidfVectorizer, HashingVectorizer, LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer, MultiLabelBinarizer, trainTestSplit, KFold, StratifiedKFold, GroupKFold, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit, RepeatedKFold, LeaveOneOut |
| 11 | Random | `deepbox/random` | NumPy | rand, randn, randint, uniform, normal, binomial, poisson, exponential, gamma, beta, choice (replace/no-replace), shuffle, permutation, Generator array generation, multinomial, dirichlet, categorical, lognormal, weibull, triangular, rayleigh |
| 12 | Stats | `deepbox/stats` | SciPy | mean, median, mode, std, variance, skewness, kurtosis, quantile, percentile, geometricMean, harmonicMean, trimMean, moment, pearsonr, spearmanr, kendalltau, corrcoef, cov, ttest_1samp, ttest_ind, ttest_rel, f_oneway, chisquare, shapiro, mannwhitneyu, kruskal, friedmanchisquare, anderson, kstest, levene, bartlett, normaltest, wilcoxon, zscore, sem, bootstrap, gaussian_kde, multiple-comparison corrections, chi2_contingency, fisher_exact, fligner, ks_2samp, median_test |
| 13 | Core | `deepbox/core` | n/a | local-only runtime utilities: JSON serialization, validation helpers, axis normalization, logger overhead |
| 14 | Tensor | `deepbox/ndarray` | n/a | optional Deepbox-only harness (`npm run bench:tensor`). Writes `deepbox-tensor.json` with extended creation, arithmetic, reduction, sort and shape timings. Excluded from the Python comparison tables |

## Methodology

- Warm-up: 5 iterations per case, excluded from timing.
- Sampling: adaptive batching with 20 to 60 timed samples per case, targeting about 250 ms total runtime.
- Timing: `process.hrtime.bigint()` in TypeScript and `time.perf_counter_ns()` in Python.
- Primary compare metric: `median_ms`.
- Stability metrics: `p95_ms`, `relative_std_pct`, sample count, batch size and total invocations.
- Result validation: duplicate ids and duplicate `operation + size` pairs are rejected before the JSON is written.
- Scope: `match` for head-to-head cases, `local` for Deepbox-only cases that are excluded from winner counts.
- Fairness: same data sizes, equivalent algorithms where possible, and identical case names and size labels on both sides. A case without a size uses `n/a`.

## Key Context

- NumPy uses C and Fortran BLAS backends (OpenBLAS or MKL).
- Deepbox runs pure TypeScript on V8 with `TypedArray` backing. It does not call a native BLAS.
- PyTorch uses the C++ ATen backend. All measurements are CPU-only.
- Matplotlib uses the C-based Agg backend for rendering.
- scikit-learn uses Cython and C extensions for core algorithms.
- Results depend on the hardware. Compare only on the same machine.

### Performance Profile

The result depends on the kind of operation:

- Faster than the Python package: DataFrame creation, small-table filters, `iloc`/`head`/`tail`, random generation, SVG rendering, core utilities and metrics.
- About equal: small linear algebra (`det`, `trace`), dataset generators and ML inference.
- Slower: large linear algebra (`matmul` at 200x200 is 45x to 52x slower than NumPy in the last recorded run), `groupBy` aggregations and sorting.

This is expected for a pure-TypeScript engine without native BLAS. The WebGPU and WASM backends accelerate only a subset of operations, so linear algebra is mostly unaccelerated. Treat the results per module, not as one overall figure.

## Structure

```text
benchmarks/
├── README.md              this file
├── RESULTS.md             generated comparison
├── compare.ts             generates RESULTS.md
├── tsconfig.json
├── requirements.txt
├── utils.ts               shared TS benchmark utilities
├── utils.py               shared Python benchmark utilities
├── deepbox/               Deepbox benchmarks
│   ├── 01-dataframe.ts
│   ├── 02-datasets.ts
│   ├── 03-linalg.ts
│   ├── 04-metrics.ts
│   ├── 05-ml.ts
│   ├── 06-ndarray.ts
│   ├── 07-nn.ts
│   ├── 08-optim.ts
│   ├── 09-plot.ts
│   ├── 10-preprocess.ts
│   ├── 11-random.ts
│   ├── 12-stats.ts
│   ├── 13-core.ts
│   └── 14-tensor.ts       optional Deepbox-only (see `bench:tensor`)
├── python/                Python benchmarks
│   ├── 01_dataframe.py
│   ├── 02_datasets.py
│   ├── 03_linalg.py
│   ├── 04_metrics.py
│   ├── 05_ml.py
│   ├── 06_ndarray.py
│   ├── 07_nn.py
│   ├── 08_optim.py
│   ├── 09_plot.py
│   ├── 10_preprocess.py
│   ├── 11_random.py
│   └── 12_stats.py
└── results/               JSON output (gitignored)
```

Head-to-head comparison tables exclude local-only cases such as `deepbox/core` and any Deepbox feature that has no close Python equivalent.

Prefer the npm `bench:*` scripts above for reproducible runs. `npm run bench:ci` runs the core, ndarray and random suites only, as a quick check that the harness still works.

## License

MIT. See LICENSE in the project root.

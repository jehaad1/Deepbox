# Deepbox Benchmark Suite

> **Website:** https://deepbox.dev · **Docs:** https://deepbox.dev/docs

Performance benchmarks comparing **Deepbox** (TypeScript) against Python's major data science libraries.

## Quick Start

```bash
# Run all Deepbox benchmarks
npm run bench:deepbox

# Run all Python benchmarks
npm run bench:python

# Run everything + generate comparison
npm run bench:all
```

## Individual Benchmarks

### Deepbox (TypeScript)

```bash
npm run bench:dataframe    # 01 — DataFrame operations    (vs Pandas)
npm run bench:datasets     # 02 — Dataset loading         (vs scikit-learn)
npm run bench:linalg       # 03 — Linear algebra          (vs NumPy/SciPy)
npm run bench:metrics      # 04 — ML metrics              (vs scikit-learn)
npm run bench:ml           # 05 — ML training & inference (vs scikit-learn)
npm run bench:ndarray      # 06 — NDArray / tensor ops    (vs NumPy)
npm run bench:nn           # 07 — Neural networks         (vs PyTorch)
npm run bench:optim        # 08 — Optimizers & schedulers (vs PyTorch)
npm run bench:plot         # 09 — Plotting / SVG render   (vs Matplotlib)
npm run bench:preprocess   # 10 — Preprocessing           (vs scikit-learn)
npm run bench:random       # 11 — Random generation       (vs NumPy)
npm run bench:stats        # 12 — Statistical analysis    (vs SciPy)
npm run bench:core         # 13 — Core runtime utilities  (Deepbox-only, local)
npm run bench:tensor       # 14 — Extra tensor / ndarray ops (Deepbox-only → deepbox-tensor.json; not in bench:compare)
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
npm run bench:compare      # generates RESULTS.md + updates root README.md
```

See **[RESULTS.md](./RESULTS.md)** for the full comparison with winner indicators per operation.

> **Note:** `bench:compare` also auto-updates the **Performance** section in the root `README.md`.

## What's Benchmarked

| # | Benchmark | Deepbox Module | Python Library | Key Operations |
| --- | --- | --- | --- | --- |
| 01 | **DataFrame** | `deepbox/dataframe` | Pandas | create, select, filter, sort, groupBy (sum/mean/count/min/max), head, tail, iloc, join, concat, fillna, dropna, describe, corr, drop |
| 02 | **Datasets** | `deepbox/datasets` | scikit-learn | loadIris, loadBreastCancer, loadDiabetes, loadDigits, loadWine, loadLinnerud, makeBlobs, makeCircles, makeMoons, makeClassification, makeRegression, makeGaussianQuantiles, Friedman/Swiss-roll style generators; local-only coverage also records DataLoader iteration, dataset splits/transforms, samplers, and CSV parsing |
| 03 | **Linalg** | `deepbox/linalg` | NumPy/SciPy | det, trace, norm, cond, matrixRank, slogdet, inv, pinv, svd, qr, lu, cholesky, eigvalsh, solve, solveTriangular, lstsq, matmul |
| 04 | **Metrics** | `deepbox/metrics` | scikit-learn | accuracy, precision, recall, f1, fbeta, confusionMatrix, hamming, jaccard, cohen kappa, matthews, balanced accuracy, logLoss, rocAuc, averagePrecision, mse, rmse, mae, r2, adjustedR2, explainedVariance, maxError, medianAbsError, mape, silhouette, calinski-harabasz, davies-bouldin, adjusted rand/mutual info, homogeneity, completeness, v-measure, fowlkes-mallows |
| 05 | **ML** | `deepbox/ml` | scikit-learn | LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet, LogisticRegression, LDA, GaussianNB, BernoulliNB, MultinomialNB, KNN/radius/centroid classifiers, LinearSVC/SVR, SGD (C+R), DecisionTree (C+R), RandomForest (C+R), ExtraTrees, AdaBoost, Bagging, Voting, Stacking, KMeans, MiniBatchKMeans, DBSCAN, Birch, MeanShift, OPTICS, GaussianMixture, SpectralClustering, PCA, TSNE, GaussianRandomProjection |
| 06 | **NDArray** | `deepbox/ndarray` | NumPy | creation (zeros, ones, full, empty, arange, linspace, eye, randn), arithmetic (add, sub, mul, div, neg, pow), math (sqrt, exp, log, abs, sin, cos, clip, sign), reductions (sum, mean, max, min, variance, std, prod, median, cumsum, cumprod), sort, argsort, reshape, flatten, transpose, squeeze, unsqueeze, concatenate, stack, slice, comparison, logical, activations (relu, sigmoid, tanh, softmax), matmul |
| 07 | **NN** | `deepbox/nn` | PyTorch | layer creation (Linear, Sequential, Conv1d/2d, RNN, LSTM, GRU, BatchNorm1d, LayerNorm), forward pass, activations (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU, Swish, Mish, Softmax), forward+backward, losses (mse, mae, rmse, huber, crossEntropy, bce), training loops (Adam, SGD), inference (noGrad), module ops |
| 08 | **Optim** | `deepbox/optim` | PyTorch | optimizer creation (SGD, Adam, AdamW, Adagrad, AdaDelta, Nadam, RMSprop), step, training loops per optimizer, LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, OneCycleLR, ReduceLROnPlateau, WarmupLR), stateDict, zeroGrad |
| 09 | **Plot** | `deepbox/plot` | Matplotlib | scatter, line plot, bar, stacked/grouped bar, barh, hist, boxplot, violinplot, pie, heatmap, imshow, contour, contourf, plotConfusionMatrix, plotRocCurve, plotPrecisionRecallCurve, plotLearningCurve, plotValidationCurve, kdeplot, stem, quiver, polar, surface, dendrogram, SVG/PNG/PDF rendering; local-only coverage also records strip, radar, and waterfall plots |
| 10 | **Preprocess** | `deepbox/preprocess` | scikit-learn | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer, KBinsDiscretizer, SplineTransformer, SimpleImputer, KNNImputer, MissingIndicator, Binarizer, PolynomialFeatures, VarianceThreshold, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression, CountVectorizer, TfidfVectorizer, HashingVectorizer, LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer, MultiLabelBinarizer, trainTestSplit, KFold, StratifiedKFold, GroupKFold, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit, RepeatedKFold, LeaveOneOut |
| 11 | **Random** | `deepbox/random` | NumPy | rand, randn, randint, uniform, normal, binomial, poisson, exponential, gamma, beta, choice (replace/no-replace), shuffle, permutation, Generator array generation, multinomial, dirichlet, categorical, lognormal, weibull, triangular, rayleigh |
| 12 | **Stats** | `deepbox/stats` | SciPy | mean, median, mode, std, variance, skewness, kurtosis, quantile, percentile, geometricMean, harmonicMean, trimMean, moment, pearsonr, spearmanr, kendalltau, corrcoef, cov, ttest_1samp, ttest_ind, ttest_rel, f_oneway, chisquare, shapiro, mannwhitneyu, kruskal, friedmanchisquare, anderson, kstest, levene, bartlett, normaltest, wilcoxon, zscore, sem, bootstrap, gaussian_kde, multiple-comparison corrections, chi2_contingency, fisher_exact, fligner, ks_2samp, median_test |
| 13 | **Core** | `deepbox/core` | — | local-only runtime utilities: JSON serialization, validation helpers, axis normalization, logger overhead |
| 14 | **Tensor** | `deepbox/ndarray` | — | optional Deepbox-only harness (`npm run bench:tensor`); writes `deepbox-tensor.json` for extended creation/arithmetic/reduction/sort/shape timing — excluded from Python comparison tables |

## Methodology

- **Warm-up**: 5 iterations per case, excluded from timing
- **Sampling**: adaptive batching with 20-60 timed samples per case, targeting about 250 ms total runtime
- **Timing**: `process.hrtime.bigint()` (TS) / `time.perf_counter_ns()` (Python)
- **Primary compare metric**: `median_ms`
- **Stability metrics**: `p95_ms`, `relative_std_pct`, sample count, batch size, total invocations
- **Result validation**: duplicate ids and duplicate `operation + size` pairs are rejected before JSON is written
- **Scope**: `match` for head-to-head cases, `local` for Deepbox-only coverage excluded from winner counts
- **Fairness**: same data sizes, equivalent algorithms where possible, and identical benchmark labels across platforms

## Key Context

- **NumPy** uses C/Fortran BLAS backends (OpenBLAS or MKL)
- **Deepbox** runs pure TypeScript on V8 with TypedArray backing — no native BLAS dispatch
- **PyTorch** uses C++ ATen backend; all measurements are CPU-only
- **Matplotlib** uses C-based Agg backend for rendering
- **scikit-learn** uses Cython/C extensions for core algorithms
- Results are hardware-dependent — always compare on the same machine

### Performance Positioning

Deepbox's performance profile varies significantly by operation type:

- **Fast**: DataFrame creation, small-table filters, iloc/head/tail, random generation, plotting (SVG render), core utilities, metrics evaluation
- **Competitive**: Small-scale linear algebra (det, trace), dataset generators, ML inference
- **Slower**: Large-scale linear algebra (matmul up to 227.5x slower than NumPy due to pure-TS impl), groupBy aggregations, sort operations

This is expected for a pure-TypeScript engine without backend dispatch or native BLAS.
Linear algebra speed will improve once WASM SIMD / WebGPU backends ship runnable modules and ndarray dispatch gates on backend availability. The blanket "fast" positioning should be understood as module-specific — Deepbox is fast for dataset I/O, visualization, metrics, and ML
orchestration, but linear algebra primitives are currently unaccelerated.

## Structure

```text
benchmarks/
├── README.md              ← this file
├── RESULTS.md             ← auto-generated comparison
├── compare.ts             ← generates RESULTS.md
├── tsconfig.json
├── requirements.txt
├── utils.ts               ← shared TS benchmark utilities
├── utils.py               ← shared Python benchmark utilities
├── deepbox/               ← all Deepbox benchmarks
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
│   └── 14-tensor.ts       ← optional Deepbox-only (see `bench:tensor`)
├── python/                ← all Python benchmarks
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
└── results/               ← JSON output (gitignored)
```

Head-to-head comparison tables intentionally exclude local-only cases such as `deepbox/core` and any Deepbox feature without a clean Python standard-library analogue.

The nested path `benchmarks/benchmarks/results/` holds JSON from an older Bun-oriented harness; canonical checked-in comparison inputs live under `benchmarks/results/`. Prefer the npm `bench:*` scripts above for reproducible runs.

## License

MIT — See LICENSE in the project root.

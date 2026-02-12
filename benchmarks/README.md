# Deepbox Benchmark Suite

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

| #   | Benchmark      | Deepbox Module       | Python Library | Key Operations                                                                                                                                                                                                                                                                                                                                                                                                     |
| --- | -------------- | -------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01  | **DataFrame**  | `deepbox/dataframe`  | Pandas         | create, select, filter, sort, groupBy (sum/mean/count/min/max), head, tail, iloc, join, concat, fillna, dropna, describe, corr, drop                                                                                                                                                                                                                                                                               |
| 02  | **Datasets**   | `deepbox/datasets`   | scikit-learn   | loadIris, loadBreastCancer, loadDiabetes, loadDigits, loadLinnerud, makeBlobs, makeCircles, makeMoons, makeClassification, makeRegression, makeGaussianQuantiles                                                                                                                                                                                                                                                   |
| 03  | **Linalg**     | `deepbox/linalg`     | NumPy/SciPy    | det, trace, norm, cond, matrixRank, slogdet, inv, pinv, svd, qr, lu, cholesky, eigvalsh, solve, solveTriangular, lstsq, matmul                                                                                                                                                                                                                                                                                     |
| 04  | **Metrics**    | `deepbox/metrics`    | scikit-learn   | accuracy, precision, recall, f1, fbeta, confusionMatrix, hamming, jaccard, cohen kappa, matthews, balanced accuracy, logLoss, rocAuc, averagePrecision, mse, rmse, mae, r2, adjustedR2, explainedVariance, maxError, medianAbsError, mape, silhouette, calinski-harabasz, davies-bouldin, adjusted rand/mutual info, homogeneity, completeness, v-measure, fowlkes-mallows                                         |
| 05  | **ML**         | `deepbox/ml`         | scikit-learn   | LinearRegression, Ridge, Lasso, LogisticRegression, GaussianNB, KNN (C+R), LinearSVC/SVR, DecisionTree (C+R), RandomForest (C+R), GradientBoosting (C+R), KMeans, DBSCAN, PCA, TSNE                                                                                                                                                                                                                                |
| 06  | **NDArray**    | `deepbox/ndarray`    | NumPy          | creation (zeros, ones, full, empty, arange, linspace, eye, randn), arithmetic (add, sub, mul, div, neg, pow), math (sqrt, exp, log, abs, sin, cos, clip, sign), reductions (sum, mean, max, min, variance, std, prod, median, cumsum, cumprod), sort, argsort, reshape, flatten, transpose, squeeze, unsqueeze, concatenate, stack, slice, comparison, logical, activations (relu, sigmoid, tanh, softmax), matmul |
| 07  | **NN**         | `deepbox/nn`         | PyTorch        | layer creation (Linear, Sequential, Conv1d/2d, RNN, LSTM, GRU, BatchNorm1d, LayerNorm), forward pass, activations (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU, Swish, Mish, Softmax), forward+backward, losses (mse, mae, rmse, huber, crossEntropy, bce), training loops (Adam, SGD), inference (noGrad), module ops                                                                                               |
| 08  | **Optim**      | `deepbox/optim`      | PyTorch        | optimizer creation (SGD, Adam, AdamW, Adagrad, AdaDelta, Nadam, RMSprop), step, training loops per optimizer, LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, OneCycleLR, ReduceLROnPlateau, WarmupLR), stateDict, zeroGrad                                                                                                                                                        |
| 09  | **Plot**       | `deepbox/plot`       | Matplotlib     | scatter, line plot, bar, barh, hist, boxplot, violinplot, pie, heatmap, imshow, contour, contourf, plotConfusionMatrix, plotRocCurve, plotPrecisionRecallCurve, plotLearningCurve, plotValidationCurve, SVG rendering                                                                                                                                                                                              |
| 10  | **Preprocess** | `deepbox/preprocess` | scikit-learn   | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer, trainTestSplit, KFold, StratifiedKFold, LeaveOneOut                                                                                                                                                                                      |
| 11  | **Random**     | `deepbox/random`     | NumPy          | rand, randn, randint, uniform, normal, binomial, poisson, exponential, gamma, beta, choice (replace/no-replace), shuffle, permutation                                                                                                                                                                                                                                                                              |
| 12  | **Stats**      | `deepbox/stats`      | SciPy          | mean, median, mode, std, variance, skewness, kurtosis, quantile, percentile, geometricMean, harmonicMean, trimMean, moment, pearsonr, spearmanr, kendalltau, corrcoef, cov, ttest_1samp, ttest_ind, ttest_rel, f_oneway, chisquare, shapiro, mannwhitneyu, kruskal, friedmanchisquare, anderson, kstest, levene, bartlett, normaltest, wilcoxon                                                                    |

## Methodology

- **Warm-up**: 5 iterations (excluded from timing)
- **Iterations**: 20 timed runs per operation (configurable)
- **Timing**: `performance.now()` (TS) / `time.perf_counter()` (Python)
- **Metrics**: mean, std dev, min, max, ops/sec
- **Fairness**: same data sizes, same algorithms, same iteration counts

## Key Context

- **NumPy** uses C/Fortran BLAS backends (OpenBLAS or MKL)
- **Deepbox** runs pure TypeScript on V8 with TypedArray backing
- **PyTorch** uses C++ ATen backend; all measurements are CPU-only
- **Matplotlib** uses C-based Agg backend for rendering
- **scikit-learn** uses Cython/C extensions for core algorithms
- Results are hardware-dependent — always compare on the same machine

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
│   └── 12-stats.ts
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

## License

MIT — See LICENSE in the project root.

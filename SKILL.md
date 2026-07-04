---
name: deepbox-agent-guide
description: Use when writing Deepbox code, examples, or full projects so imports, module selection, types, errors, docs pages, and framework-specific patterns stay accurate.
---

# Deepbox Skill Guide

This file is for AI agents, code generators, and automation that need to build applications and projects with Deepbox.

Use it when generating Deepbox code, selecting modules, following the website docs, or turning a task into a runnable Deepbox example or project.

## Project Facts

- Package: `deepbox`
- Current release line in this repository: `v1.0.0`
- Runtime requirement: Node.js `>= 24.13.0`
- Package shape: ESM + CommonJS + `.d.ts`
- Runtime dependencies: `0`
- Root import behavior: `deepbox` exports namespaces, not direct named APIs

## Install

```bash
npm install deepbox
```

## Import Rules

### Preferred

Use named imports from subpath exports:

```ts
import { tensor, parameter } from "deepbox/ndarray";
import { LinearRegression } from "deepbox/ml";
import { DataFrame } from "deepbox/dataframe";
```

### Also Valid

Use the root package as a namespace container:

```ts
import * as db from "deepbox";

const x = db.ndarray.tensor([1, 2, 3]);
const model = new db.ml.LogisticRegression();
```

### Avoid

Do not generate this:

```ts
import { tensor, LinearRegression } from "deepbox";
```

The root entry does not export those names directly.

## Source of Truth

Assume the agent may only have the package repo and the public website.

When you need the real current Deepbox feature set for writing code, use this precedence order:

1. the live website at `https://deepbox.dev`
2. `https://deepbox.dev/docs`
3. `https://deepbox.dev/examples`
4. `https://deepbox.dev/projects`
5. local examples and projects if this repo is present
6. local source barrels such as `src/<module>/index.ts` only as a verification step

Important:

- The website is the primary discovery surface for agents writing Deepbox code.
- Examples and projects matter as much as API pages because they show intended usage.
- If local source is available, use it to verify exports before making strong claims about named APIs.

## Website-First Discovery Workflow

When an agent needs full module coverage instead of a short summary:

1. Open `https://deepbox.dev`.
2. Open `https://deepbox.dev/docs`.
3. Enumerate the docs pages from the docs navigation and inspect the specific pages relevant to the task.
4. Open `https://deepbox.dev/examples` for runnable usage patterns.
5. Open `https://deepbox.dev/projects` for larger end-to-end integrations.
6. If local source is available, cross-check the exported surface in `src/<module>/index.ts`.
7. If the task uses the root package, remember that `deepbox` exports namespaces only.

If the live site is unavailable, fall back to local examples and projects first, then local source barrels.

## Module Chooser

| Module | Use it for | Common exports |
| --- | --- | --- |
| `deepbox/core` | shared types, errors, config, validation, serialization, backends | `DType`, `Shape`, `InvalidParameterError`, `save()` |
| `deepbox/ndarray` | tensors, autograd, sparse arrays, general numerical ops | `Tensor`, `tensor()`, `parameter()`, `einsum()` |
| `deepbox/linalg` | decompositions, inverse problems, matrix functions | `svd()`, `solve()`, `matrix_power()` |
| `deepbox/dataframe` | tabular data workflows | `DataFrame`, `Series`, `to_datetime()` |
| `deepbox/stats` | descriptive stats, tests, distributions, KDE, power analysis | `mean()`, `ttest_ind()`, `norm`, `GaussianKDE` |
| `deepbox/metrics` | evaluation metrics | `accuracy()`, `mse()`, `silhouetteScore()` |
| `deepbox/preprocess` | feature prep and data splitting | `StandardScaler`, `OneHotEncoder`, `KFold` |
| `deepbox/ml` | classical ML estimators and pipelines | `RandomForestClassifier`, `Pipeline`, `GridSearchCV` |
| `deepbox/nn` | neural-network modules and losses | `Module`, `Sequential`, `Linear`, `crossEntropyLoss()` |
| `deepbox/optim` | optimizers and schedulers | `Adam`, `SGD`, `CosineAnnealingLR` |
| `deepbox/random` | seeded randomness and distributions | `setSeed()`, `Generator`, `multivariate_normal()` |
| `deepbox/datasets` | built-in datasets, samplers, remote fetchers | `loadIris()`, `DataLoader`, `makeBlobs()` |
| `deepbox/plot` | figures and plots | `figure()`, `plot()`, `heatmap()`, `saveFig()` |

## Website Page Map

Use this table when you need exact places to fetch current features quickly from the website while writing code.

| Area | Website pages | Local source to verify exports |
| --- | --- | --- |
| Home | `https://deepbox.dev/` | `src/index.ts` if available |
| Docs index | `https://deepbox.dev/docs` | module barrels in `src/` if available |
| Examples index | `https://deepbox.dev/examples` | `docs/examples/*` if available |
| Projects index | `https://deepbox.dev/projects` | `docs/projects/*` if available |
| Core docs | `/docs/core-types`, `/docs/core-config`, `/docs/core-errors`, `/docs/core-utils` | `src/core/index.ts` |
| NDArray docs | `/docs/ndarray-tensor`, `/docs/ndarray-ops`, `/docs/ndarray-activations`, `/docs/ndarray-shape`, `/docs/ndarray-autograd`, `/docs/ndarray-sparse` | `src/ndarray/index.ts` |
| Linalg docs | `/docs/linalg-decompositions`, `/docs/linalg-solvers`, `/docs/linalg-properties` | `src/linalg/index.ts` |
| DataFrame docs | `/docs/dataframe-overview`, `/docs/dataframe-series`, `/docs/dataframe-io-styling` | `src/dataframe/index.ts` |
| Stats docs | `/docs/stats-descriptive`, `/docs/stats-distributions`, `/docs/stats-tests` | `src/stats/index.ts` |
| Metrics docs | `/docs/metrics-classification`, `/docs/metrics-regression`, `/docs/metrics-clustering` | `src/metrics/index.ts` |
| Preprocess docs | `/docs/preprocess-scalers`, `/docs/preprocess-encoders`, `/docs/preprocess-features`, `/docs/preprocess-splitting` | `src/preprocess/index.ts` |
| ML docs | `/docs/ml-linear`, `/docs/ml-tree`, `/docs/ml-ensemble`, `/docs/ml-svm`, `/docs/ml-neighbors`, `/docs/ml-naive-bayes`, `/docs/ml-clustering`, `/docs/ml-manifold`, `/docs/ml-decomposition`, `/docs/ml-model-selection`, `/docs/ml-advanced` | `src/ml/index.ts` |
| NN docs | `/docs/nn-module`, `/docs/nn-layers`, `/docs/nn-recurrent`, `/docs/nn-attention`, `/docs/nn-normalization`, `/docs/nn-activations`, `/docs/nn-losses` | `src/nn/index.ts` |
| Optim docs | `/docs/optim-optimizers`, `/docs/optim-schedulers` | `src/optim/index.ts` |
| Random docs | `/docs/random-generation`, `/docs/random-distributions` | `src/random/index.ts` |
| Datasets docs | `/docs/datasets-builtin`, `/docs/datasets-synthetic`, `/docs/datasets-dataloader` | `src/datasets/index.ts` |
| Plot docs | `/docs/plot-basic`, `/docs/plot-statistical`, `/docs/plot-ml` | `src/plot/index.ts` |

Full URLs use the `https://deepbox.dev` prefix.

## Page Enumeration Rule

If a task asks for "all features", "all docs pages", or "everything available":

1. Start from `https://deepbox.dev/docs`.
2. List the page slugs from the docs navigation.
3. Visit the relevant module pages for detail.
4. If local source is available, cross-check the final answer against the matching `src/<module>/index.ts`.

The current known docs page list is:

- `core-types`
- `core-config`
- `core-errors`
- `core-utils`
- `ndarray-tensor`
- `ndarray-ops`
- `ndarray-activations`
- `ndarray-shape`
- `ndarray-autograd`
- `ndarray-sparse`
- `linalg-decompositions`
- `linalg-solvers`
- `linalg-properties`
- `dataframe-overview`
- `dataframe-series`
- `dataframe-io-styling`
- `stats-descriptive`
- `stats-distributions`
- `stats-tests`
- `metrics-classification`
- `metrics-regression`
- `metrics-clustering`
- `preprocess-scalers`
- `preprocess-encoders`
- `preprocess-features`
- `preprocess-splitting`
- `ml-linear`
- `ml-tree`
- `ml-ensemble`
- `ml-svm`
- `ml-neighbors`
- `ml-naive-bayes`
- `ml-clustering`
- `ml-manifold`
- `ml-decomposition`
- `ml-model-selection`
- `ml-advanced`
- `nn-module`
- `nn-layers`
- `nn-recurrent`
- `nn-attention`
- `nn-normalization`
- `nn-activations`
- `nn-losses`
- `optim-optimizers`
- `optim-schedulers`
- `random-generation`
- `random-distributions`
- `datasets-builtin`
- `datasets-synthetic`
- `datasets-dataloader`
- `plot-basic`
- `plot-statistical`
- `plot-ml`

## Core Mental Model

### Tensors

- `deepbox/ndarray` is mostly functional.
- `Tensor` is the non-gradient tensor type.
- `GradTensor` powers reverse-mode autodiff.
- `parameter()` creates trainable tensors.
- Use `noGrad()` when a computation must not build a graph.

### Classical ML

- Estimators generally follow `fit()`, `predict()`, `transform()`, and `fitTransform()` style patterns.
- Many estimators also support `getParams()` and `setParams()`.
- Pipeline-style composition lives in `deepbox/ml`, not `deepbox/preprocess`.

### Neural Networks

- `deepbox/nn` follows a `Module`/`forward()` mental model similar to PyTorch.
- Parameters come from modules and are passed to optimizers in `deepbox/optim`.
- Training helpers such as `Trainer`, `EarlyStopping`, and `ModelCheckpoint` live in `deepbox/nn`.

### Plotting

- `deepbox/plot` is figure-oriented.
- `figure()`, `subplot()`, and `gca()` manage state.
- `show()` renders to SVG by default.
- `saveFig()` writes SVG, PNG, or PDF.

### DataFrames

- `DataFrame` and `Series` are object-oriented APIs.
- CSV and JSON helpers largely live as `DataFrame` methods.
- Excel and Parquet helpers are exported from the module barrel.

## Core Types

These are the types agents should know when generating or reviewing code:

- `Shape`: readonly tensor dimensions such as `[]`, `[3]`, or `[2, 4]`
- `Axis`: `number | "index" | "rows" | "columns"`
- `Device`: `"cpu" | "webgpu" | "wasm"`
- `DType`: `"float16" | "bfloat16" | "float32" | "float64" | "int32" | "int64" | "uint8" | "bool" | "complex64" | "complex128" | "string"`
- `ScalarDType`: numeric scalar dtypes excluding `int64`, complex dtypes, and `string`
- `ElementOf<D>`: maps a `DType` to its JS element type
- `TypedArray`: native numeric storage types supported by Deepbox
- `ExtendedTypedArray`: Deepbox half-precision and complex array wrappers
- `TensorStorage`: `TypedArray | ExtendedTypedArray | string[]`
- `TensorLike<S, D>`: structural tensor interface
- `AnyTensor`: `Tensor | GradTensor`

## Error Hierarchy

Prefer these errors when authoring Deepbox-adjacent code or explaining failure modes:

- `DeepboxError`: base class
- `InvalidParameterError`: bad function or constructor arguments
- `ShapeError`: incompatible or invalid shapes
- `BroadcastError`: broadcasting incompatibility
- `DTypeError`: unsupported or mismatched dtype
- `IndexError`: invalid indexing
- `NotFittedError`: estimator used before `fit()`
- `ConvergenceError`: iterative algorithm failed to converge
- `DeviceError`: unavailable or unsupported backend/device
- `MemoryError`: allocation or memory-bound failure
- `DataValidationError`: invalid input data
- `NotImplementedError`: declared surface not implemented

## Public API Inventory

This section is intentionally more exact than a marketing summary. It is still not a replacement for the barrel files; treat it as a fast map of what exists and where to fetch the exhaustive list.

### `deepbox/core`

- Types and constants: `Axis`, `Device`, `DType`, `ElementOf`, `ExtendedTypedArray`, `ScalarDType`, `Shape`, `TensorLike`, `TensorStorage`, `TypedArray`, `DEVICES`, `DTYPES`, `isDevice()`, `isDType()`
- Config: `getConfig()`, `getDevice()`, `getDtype()`, `getSeed()`, `resetConfig()`, `setConfig()`, `setDevice()`, `setDtype()`, `setSeed()`
- Errors: `DeepboxError`, `BroadcastError`, `ConvergenceError`, `DataValidationError`, `DeviceError`, `DTypeError`, `IndexError`, `InvalidParameterError`, `MemoryError`, `NotFittedError`, `NotImplementedError`, `ShapeError`
- Backend: `CpuBackend`, `WasmBackend`, `WebGpuBackend`, `registerBackend()`, `unregisterBackend()`, `getBackend()`, `getKernelBackend()`, `getHostAccelerator()`, `isBackendAvailable()`, `isKernelBackend()`, `isHostAcceleratorBackend()`, `listBackends()`, `WGSL_SHADERS`, `WAT_MODULES`, `WASM_BINARIES`
- Validation and utilities: `check_array()`, `check_X_y()`, `check_is_fitted()`, `validateShape()`, `validateDtype()`, `shapeToSize()`, dtype helpers, axis normalization helpers, typed-array access helpers
- Serialization, logging, warnings, parallelism: `save()`, `load()`, `toJSON()`, `fromJSON()`, `Logger`, warning helpers, `WorkerPool`, `createWorkerPool()`, `availableCores()`

### `deepbox/ndarray`

- Tensor types and autograd: `Tensor`, `GradTensor`, `AnyTensor`, `parameter()`, `noGrad()`
- Creation and tensor helpers: `tensor()`, `Complex`, `Complex64Array`, `Complex128Array`, `Float16Array`, `BFloat16Array`
- Linalg-style helpers: `dot()`, `corrcoef()`, `cov()`, `tensordot()`
- Activation functions: `relu()`, `sigmoid()`, `softmax()`, `logSoftmax()`, `gelu()`, `mish()`, `swish()`, `elu()`, `leakyRelu()`, `softplus()`
- Core operation families exported from the barrel: arithmetic, comparison, logical, reduction, shape manipulation, sorting, set operations, FFT, signal, numerical, stacking, indexing, NaN-aware reduction, window functions
- High-value advanced exports: `einsum()`, `broadcast_to()`, `booleanIndex()`, `fancyIndex()`, `meshgrid()`, `index_select()`, `insert()`, `delete_()`, `searchsorted()`, `histogram()`, `bincount()`, `fft*`, `ifft*`, `rfft()`, `irfft()`, `convolve()`, `correlate()`, `interp()`, `trapz()`, `gradient()`, `pad()`, `rot90()`, `where()`
- Sparse: `CSRMatrix` support is part of the ndarray surface; inspect `src/ndarray/sparse/` and the barrel for exact types

### `deepbox/linalg`

- Decompositions: `svd()`, `qr()`, `lu()`, `cholesky()`, `eig()`, `eigh()`, `eigvals()`, `eigvalsh()`, `schur()`, `polar()`, `hessenberg()`
- Inverses and properties: `inv()`, `pinv()`, `det()`, `trace()`, `matrixRank()`, `slogdet()`, `norm()`, `cond()`
- Matrix functions and constructors: `expm()`, `logm()`, `sqrtm()`, `matrix_power()`, `kron()`, `block_diag()`, `companion()`, `circulant()`, `hadamard()`, `hankel()`, `hilbert()`, `toeplitz()`, `vandermonde()`
- Solvers: `solve()`, `lstsq()`, `solveTriangular()`, `solve_banded()`, `lyapunov()`, `sylvester()`, `sparseSolve()`, `sparseCholeskySolve()`, `denseToCSR()`

### `deepbox/dataframe`

- Core structures: `DataFrame`, `DataFrameGroupBy`, `Series`
- Accessors and types: `StringAccessor`, `DateTimeAccessor`, `PlotAccessor`, `StyleAccessor`, `MultiIndex`, `Categorical`
- Date/time helpers: `to_datetime()`, `date_range()`, `timedelta()`
- Exported IO helpers: `readParquet()`, `writeParquet()`, `readXlsx()`, `writeXlsx()`
- Important note: CSV and JSON workflows are primarily implemented as `DataFrame` methods such as `fromCsvString()`, `toCsvString()`, `toCsv()`, `fromJsonString()`, `toJsonString()`, and `toJson()`

### `deepbox/stats`

- Confidence intervals: `meanConfidenceInterval()`, `meanConfidenceIntervalZ()`, `meanDiffConfidenceInterval()`, `proportionConfidenceInterval()`
- Correlation: `corrcoef()`, `cov()`, `pearsonr()`, `spearmanr()`, `kendalltau()`, `partialcorr()`, `pointbiserialr()`
- Descriptive statistics: `mean()`, `median()`, `mode()`, `variance()`, `std()`, `sem()`, `iqr()`, `zscore()`, `bootstrap()`, `cohenD()`, `moment()`, `skewness()`, `kurtosis()`
- Distributions: `beta`, `binom`, `cauchy`, `chi2`, `expon`, `f`, `gamma`, `geom`, `hypergeom`, `laplace`, `lognorm`, `nbinom`, `norm`, `pareto`, `poisson`, `t`, `uniform`, `weibull`
- KDE and power analysis: `GaussianKDE`, `gaussian_kde()`, `tTestPower()`
- Multiple testing: `bonferroni()`, `holm()`, `sidak()`, `benjaminiHochberg()`
- Tests: `ttest_*`, `f_oneway()`, `f_twoway()`, `chisquare()`, `chi2_contingency()`, `fisher_exact()`, `mannwhitneyu()`, `wilcoxon()`, `kruskal()`, `friedmanchisquare()`, `shapiro()`, `normaltest()`, `kstest()`, `ks_2samp()`, `anderson()`, `levene()`, `bartlett()`, `fligner()`, `lilliefors()`, `runs_test()`, `median_test()`

### `deepbox/metrics`

- Classification: `accuracy()`, `precision()`, `recall()`, `f1Score()`, `fbetaScore()`, `rocAucScore()`, `rocCurve()`, `precisionRecallCurve()`, `confusionMatrix()`, `classificationReport()`, `logLoss()`, `hammingLoss()`, `jaccardScore()`, `matthewsCorrcoef()`, `cohenKappaScore()`, `balancedAccuracyScore()`, `averagePrecisionScore()`
- Regression: `mse()`, `rmse()`, `mae()`, `mape()`, `r2Score()`, `adjustedR2Score()`, `explainedVarianceScore()`, `maxError()`, `medianAbsoluteError()`
- Clustering: `silhouetteScore()`, `silhouetteSamples()`, `daviesBouldinScore()`, `calinskiHarabaszScore()`, `adjustedRandScore()`, `adjustedMutualInfoScore()`, `normalizedMutualInfoScore()`, `homogeneityScore()`, `completenessScore()`, `vMeasureScore()`, `fowlkesMallowsScore()`
- Extra and pairwise: `brierScoreLoss()`, `coverageError()`, `d2TweedieScore()`, `detCurve()`, `hingeLoss()`, `labelRankingLoss()`, `meanGammaDeviance()`, `meanPinballLoss()`, `meanPoissonDeviance()`, `meanSquaredLogError()`, `multilabelConfusionMatrix()`, `smape()`, `topKAccuracyScore()`, `zeroOneLoss()`, `ndcgScore()`, `pairwiseCosine()`, `pairwiseEuclidean()`, `pairwiseManhattan()`, `reciprocalRank()`

### `deepbox/preprocess`

- Discretization: `KBinsDiscretizer`
- Encoders: `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`, `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`
- Feature selection: `f_classif()`, `f_regression()`, `RFE`, `RFECV`, `SelectFromModel`, `SelectKBest`, `VarianceThreshold`
- Imputation: `SimpleImputer`, `KNNImputer`, `MissingIndicator`
- Mutual information: `mutual_info_classif()`, `mutual_info_regression()`
- Feature transformers: `Binarizer`, `FunctionTransformer`, `PolynomialFeatures`, `SplineTransformer`
- Scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`
- Splitting: `trainTestSplit()`, `KFold`, `StratifiedKFold`, `GroupKFold`, `LeaveOneOut`, `LeavePOut`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplit`, `TimeSeriesSplit`
- Text: `CountVectorizer`, `HashingVectorizer`, `TfidfVectorizer`

### `deepbox/ml`

- Base and output utilities: estimator types, `assertEstimator()`, `getEstimatorTags()`, `get_output()`, `set_output()`, `reset_output()`
- Anomaly detection: `IsolationForest`, `LocalOutlierFactor`
- Calibration: `CalibratedClassifierCV`, `calibrationCurve()`
- Clustering: `AffinityPropagation`, `AgglomerativeClustering`, `Birch`, `DBSCAN`, `GaussianMixture`, `KMeans`, `MeanShift`, `MiniBatchKMeans`, `OPTICS`, `SpectralClustering`
- Decomposition: `PCA`, `FastICA`, `NMF`, `TruncatedSVD`, `LatentDirichletAllocation`
- Discriminant analysis: `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`
- Ensemble: `AdaBoost*`, `Bagging*`, `GradientBoosting*`, `Stacking*`, `Voting*`
- Gaussian processes: `GaussianProcessClassifier`, `GaussianProcessRegressor`
- Inspection: `permutationImportance()`
- Linear models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`, `BayesianRidge`, `HuberRegressor`, `IsotonicRegression`, `KernelRidge`, `QuantileRegressor`, `RANSACRegressor`
- Manifold learning: `TSNE`, `Isomap`, `MDS`, `SpectralEmbedding`
- MLP: `MLPClassifier`, `MLPRegressor`
- Model selection: `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score()`, `cross_validate()`
- Multiclass: `OneVsOneClassifier`, `OneVsRestClassifier`
- Naive Bayes: `GaussianNB`, `BernoulliNB`, `CategoricalNB`, `ComplementNB`, `MultinomialNB`
- Neighbors: `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestNeighbors`, `NearestCentroid`, `KDTree`, `BallTree`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`
- Pipeline and feature composition: `Pipeline`, `FeatureUnion`, `ColumnTransformer`, `makePipeline()`
- Random projection: `GaussianRandomProjection`
- Semi-supervised: `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`
- SVM: `LinearSVC`, `LinearSVR`, `SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM`
- Trees: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `export_text()`

### `deepbox/nn`

- Gradient clipping: `clip_grad_norm_()`, `clip_grad_value_()`
- Containers: `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`
- Initialization: `constant_()`, `kaiming_*()`, `normal_()`, `ones_()`, `orthogonal_()`, `sparse_()`, `uniform_()`, `xavier_*()`, `zeros_()`
- Activation layers: `ELU`, `GELU`, `GLU`, `Hardsigmoid`, `Hardswish`, `LeakyReLU`, `LogSoftmax`, `Mish`, `PReLU`, `ReLU`, `SELU`, `Sigmoid`, `SiLU`, `Softmax`, `Softplus`, `Softsign`, `Swish`, `Tanh`
- Attention and transformer: `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`, `FullTransformer`, `PositionalEncoding`
- Convolution and pooling: `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, adaptive pooling exports
- Regularization, embedding, normalization, padding, recurrent, spectral norm, upsampling, utility layers
- Losses: `binaryCrossEntropyLoss()`, `binaryCrossEntropyWithLogitsLoss()`, `cosineEmbeddingLoss()`, `crossEntropyLoss()`, `ctcLoss()`, `gaussianNLLLoss()`, `huberLoss()`, `klDivLoss()`, `maeLoss()`, `marginRankingLoss()`, `mseLoss()`, `nllLoss()`, `poissonNLLLoss()`, `rmseLoss()`, `smoothL1Loss()`, `tripletMarginLoss()`
- Base module and training: `Module`, `Trainer`, `EarlyStopping`, `GradientAccumulator`, `ModelCheckpoint`

### `deepbox/optim`

- Base optimizer: `Optimizer`
- Optimizers: `SGD`, `Adam`, `AdamW`, `Nadam`, `Adamax`, `RAdam`, `RMSprop`, `Adagrad`, `AdaDelta`, `LBFGS`, `LARS`, `LAMB`, `SparseAdam`, `ASGD`, `Rprop`
- Schedulers: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `CyclicLR`, `LambdaLR`, `LinearLR`, `OneCycleLR`, `PolynomialLR`, `ReduceLROnPlateau`, `SequentialLR`, `WarmupLR`

### `deepbox/random`

- The module exports `Generator` plus a large set of inline random functions from `src/random/index.ts`
- Source barrel inspection is required here for the exhaustive current function list because many exports are defined directly in the file body rather than re-exported from smaller barrels
- High-value capabilities include seed management, scalar and tensor generation, discrete and continuous distributions, multivariate sampling, permutation/sampling helpers, and dtype/device-aware random tensors

### `deepbox/datasets`

- Data loading core: `DataLoader`
- Synthetic generators: `makeBiclusters()`, `makeBlobs()`, `makeCheckerboard()`, `makeCircles()`, `makeClassification()`, `makeFriedman1()`, `makeFriedman2()`, `makeFriedman3()`, `makeGaussianQuantiles()`, `makeLowRankMatrix()`, `makeMoons()`, `makeRegression()`, `makeSCurve()`, `makeSPDMatrix()`, `makeSparseUncorrelated()`, `makeSwissRoll()`
- Image helpers: `fetchMNIST()`, `fetchCIFAR10()`
- Kaggle helpers: `readKaggleCredentials()`, `searchKaggleDatasets()`, `fetchKaggleDatasetInfo()`, `listKaggleFiles()`, `fetchKaggleDataset()`
- Built-in loaders: `loadIris()`, `loadDigits()`, `loadBreastCancer()`, `loadDiabetes()`, `loadLinnerud()`, plus the larger family of domain-specific datasets exported in the barrel
- Remote CSV: `fetchCSVDataset()`, `parseCSV()`
- Samplers and transforms: `SequentialSampler`, `SubsetRandomSampler`, `WeightedRandomSampler`, `filterDataset()`, `mapDataset()`, `randomSplit()`, `Subset`
- Text helpers: `fetch20Newsgroups()`, `fetchIMDB()`

### `deepbox/plot`

- Core figure API: `Figure`, `Axes`, `figure()`, `gca()`, `subplot()`, `show()`, `saveFig()`
- Animation, interactivity, palette, and renderer-related exports are part of the public surface
- Base plotting: `plot()`, `scatter()`, `bar()`, `barh()`, `hist()`, `boxplot()`, `violinplot()`, `pie()`, `heatmap()`, `imshow()`, `contour()`, `contourf()`, `legend()`
- Plot variants and helpers: `axhline()`, `axvline()`, `stackedBar()`, `groupedBar()`
- ML and analysis visuals: `plotConfusionMatrix()`, `plotRocCurve()`, `plotPrecisionRecallCurve()`, `plotLearningCurve()`, `plotValidationCurve()`, `plotDecisionBoundary()`, `plotResiduals()`, `plotFeatureImportance()`, `plotElbowCurve()`, `plotSilhouette()`, `plotCalibrationCurve()`, `plotDendrogram()`
- Statistical and specialized visuals: `kdeplot()`, `pairplot()`, `jointplot()`, `stem()`, `strip()`, `radar()`, `waterfall()`, `quiver()`, `polar()`

## Examples and Projects Lookup

When you need real usage instead of API summaries, prefer the website examples and projects first:

- `https://deepbox.dev/examples`
- `https://deepbox.dev/projects`

If this repo is present locally:

- Examples live in `docs/examples/00-quick-start` through `docs/examples/49-advanced-linear-algebra`
- Projects live in `docs/projects/01-financial-risk-analysis` through `docs/projects/09-experimentation-platform`
- Each example or project typically contains an `index.ts` entry and a `README.md`
- If you need runnable command names, inspect `package.json`

## Common Recipes

### Tensor and Autograd

```ts
import { parameter } from "deepbox/ndarray";

const x = parameter([1, 2, 3]);
const y = x.mul(x).sum();
y.backward();

console.log(x.grad?.toString());
```

### Classical ML

```ts
import { tensor } from "deepbox/ndarray";
import { StandardScaler } from "deepbox/preprocess";
import { LogisticRegression } from "deepbox/ml";

const X = tensor([
  [0, 1],
  [1, 0],
  [1, 1],
  [0, 0],
]);
const y = tensor([1, 1, 0, 0]);

const scaler = new StandardScaler();
const XScaled = scaler.fitTransform(X);

const model = new LogisticRegression();
model.fit(XScaled, y);
```

### Neural Network Training

```ts
import { Linear, ReLU, Sequential, mseLoss } from "deepbox/nn";
import { Adam } from "deepbox/optim";

const model = new Sequential(
  new Linear(4, 16),
  new ReLU(),
  new Linear(16, 1)
);

const optimizer = new Adam(model.parameters(), { lr: 1e-3 });
```

### Plotting

```ts
import { tensor } from "deepbox/ndarray";
import { figure, plot, saveFig } from "deepbox/plot";

figure({ width: 800, height: 500 });
plot(tensor([1, 2, 3]), tensor([2, 4, 8]), { label: "growth" });
await saveFig("growth.svg");
```

## Agent Guardrails

- Prefer subpath imports unless you intentionally want namespace imports.
- Assume CPU execution unless backend registration is explicitly shown. With `WebGpuBackend` registered, `webgpu` tensors execute element-wise arithmetic, activations, matmul (`dot`), and full reductions on the GPU; other ops throw `DeviceError` with a transfer hint — read results back with `await t.cpu()`. With `WasmBackend` registered, `wasm` tensors accelerate contiguous float32 arithmetic via SIMD and silently fall back to identical CPU results otherwise.
- Kernel devices (`webgpu`) support `float32` and `float16` (true on-device half via WGSL `shader-f16`), plus `bfloat16`; other dtypes need `astype('float32')` before moving tensors.
- `Module.to(device)` returns a `Promise` (it actually transfers parameters) and can throw `DeviceError` if the backend is unavailable.
- Use `setSeed()` or `Generator` in reproducible examples that depend on randomness.
- Mention remote/network behavior when using dataset fetch helpers.
- Do not claim an API exists at the root package unless it is actually exported there.
- When documenting `dataframe` IO, distinguish between `DataFrame` methods and module-barrel helpers.
- When building wrappers or utilities around Deepbox, prefer Deepbox custom errors over generic `Error`.
- Prefer website docs and examples over maintainer-oriented repo files.

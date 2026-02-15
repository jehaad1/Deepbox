# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-14

78 source files changed (1,418 insertions, 674 deletions). Bug fixes, type safety improvements, multiclass support, and documentation overhaul. Full audit, consistency pass, and enterprise-grade hardening.

### Added

- **`GradTensor.isGradTensor()`** — static duck-typing method for cross-module `instanceof` compatibility
- **`GradTensor` public constructor** — two overloads: `(data, options?)` for users and `({tensor, requiresGrad, prev, backward})` for internals
- **`Tensor.slice()` instance method** — `t.slice(...)` in addition to the standalone `slice(t, ...)`
- **`ScalarDType` and `ElementOf<D>` types** — enables `tensor([1,2,3]).at(0)` to return `number` instead of `unknown`
- **`DataValue` type export** from `deepbox/dataframe`
- **`loadDigits().images`** — reshaped `[1797, 8, 8]` tensor matching sklearn's `.images` attribute
- **`makeClassification({ flipY })` parameter** — label noise injection (default 1%)
- **`DBSCAN.nClusters` getter** — returns number of discovered clusters (excludes noise)
- **`PolynomialFeatures`** transformer in `deepbox/preprocess`
- **Vector-matrix `dot()` support** — `dot(1D, 2D)` now works correctly
- **`norm()` overloads** — `norm(x)` returns `number`; `norm(x, ord, axis)` returns `Tensor | number`

### Fixed

- **`mseLoss` / `crossEntropyLoss` / `binaryCrossEntropyWithLogitsLoss`** — replaced `instanceof GradTensor` with `GradTensor.isGradTensor()` to fix silent loss-of-gradient bug across module boundaries
- **`GradientBoostingClassifier`** — added multiclass support via One-vs-Rest strategy (was binary-only)
- **`LinearSVC`** — added multiclass support via One-vs-Rest strategy (was binary-only)
- **`crossEntropyLoss`** — 1D GradTensor target now works; overload signatures accept `AnyTensor`
- **`DataLoader` iterator type** — return type now conditional `[Tensor, Tensor] | [Tensor]` instead of `never`
- **`DataFrame.filter()` row type** — changed from `unknown` to `Record<string, any>` for usability
- **`precision()` / `recall()` / `f1Score()`** — auto-detect multiclass and default to `"weighted"` averaging instead of `"binary"`
- **`f1Score()`** — accepts both `string` and `{ average: string }` argument forms
- **`relu()` / `leakyRelu()` / `elu()` return types** — narrowed to `Tensor<Shape, ScalarDType>`
- **`LinearRegression.predict()` return type** — narrowed to `Tensor<Shape, ScalarDType>`
- **`ensureNumericDType()` context parameter** — made optional (default: `"operation"`)
- **JSDoc `@see` links** — all 40+ `deepbox.dev` references verified against actual docs routes
- **Documentation code snippets** — 7 broken snippets fixed across datasets, getting-started, ml, optim, plot, preprocess content files

### Changed

- All documentation, examples, and projects verified against actual API behavior
- All 4,344 tests pass, typecheck clean, lint clean, format clean
- All 33 examples and 6 enterprise projects run successfully
- 542 head-to-head benchmarks validated against Python equivalents
- Test suite expanded from 4,009 to 4,344 tests across 260 test files
- `content.json` version updated from `v0.1.0` to `v0.2.0`
- Regenerated `examples.ts` and `projects.ts` from fresh capture (33 examples, 6 projects)
- Updated copyright year range to 2025-2026
- Updated issue template links to use `deepbox.dev`
- Removed stale loose example files from `docs/examples/` (consolidated into numbered directories)

### Infrastructure

- Zero runtime dependencies confirmed
- ESM + CommonJS dual output with full type declarations
- Strict TypeScript with all checks enabled (`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `noPropertyAccessFromIndexSignature`)
- CI/CD pipeline with GitHub Actions (build, test, coverage, npm publish with provenance)
- Dependabot configured for weekly dependency updates
- Coverage thresholds enforced: 89% lines, 90% functions, 72% branches, 88% statements

## [0.1.0] - 2026-02-12

Initial release.

### Core (`deepbox/core`)

- Type system: `DType`, `Shape`, `Device`, `TypedArray`, `TensorLike`
- Custom error hierarchy: `DeepboxError`, `ShapeError`, `BroadcastError`, `DTypeError`, `IndexError`, `InvalidParameterError`, `NotFittedError`, `NotImplementedError`, `MemoryError`, `DataValidationError`
- Global configuration: `getConfig()`, `setConfig()`, `setDevice()`, `setDtype()`, `setSeed()`
- Validation utilities: `validateShape()`, `validateDtype()`, `shapeToSize()`, `dtypeToTypedArrayCtor()`

### N-Dimensional Arrays (`deepbox/ndarray`)

- `Tensor` class with strided N-d array storage, 7 dtypes (float32, float64, int32, int64, uint8, bool, string)
- Creation: `tensor()`, `zeros()`, `ones()`, `arange()`, `linspace()`, `logspace()`, `geomspace()`, `eye()`, `full()`, `empty()`, `randn()`
- Shape ops: `reshape()`, `transpose()`, `flatten()`, `squeeze()`, `unsqueeze()`, `expandDims()`
- Indexing: `slice()`, `gather()`
- 90+ operations: arithmetic, comparison, logical, trigonometric, activation, reduction, sorting, manipulation
- Activation functions: `relu()`, `sigmoid()`, `softmax()`, `logSoftmax()`, `gelu()`, `mish()`, `swish()`, `elu()`, `leakyRelu()`, `softplus()`
- Automatic differentiation: `GradTensor`, `parameter()`, `noGrad()` with backward support for 20+ ops
- Sparse matrices: `CSRMatrix` (CSR format) with add, sub, scale, multiply, matvec, matmul, transpose
- Broadcasting: full broadcasting semantics

### Linear Algebra (`deepbox/linalg`)

- Decompositions: `svd()`, `qr()`, `lu()`, `cholesky()`, `eig()`, `eigh()`, `eigvals()`, `eigvalsh()`
- Solvers: `solve()`, `lstsq()`, `solveTriangular()`
- Inverse: `inv()`, `pinv()`
- Properties: `det()`, `trace()`, `matrixRank()`, `slogdet()`, `cond()`
- Norms: `norm()` (L1, L2, Frobenius, nuclear, inf)

### DataFrames (`deepbox/dataframe`)

- `DataFrame` and `Series` classes with 50+ operations
- Filtering, grouping, joining, merging, pivoting, sorting, reshaping
- CSV I/O: `readCSV()`, `toCSV()`
- Descriptive statistics: `describe()`, value counts, correlation matrices

### Statistics (`deepbox/stats`)

- Descriptive: `mean()`, `median()`, `mode()`, `std()`, `variance()`, `skewness()`, `kurtosis()`, `quantile()`, `percentile()`, `moment()`, `geometricMean()`, `harmonicMean()`, `trimMean()`
- Correlation: `corrcoef()`, `cov()`, `pearsonr()`, `spearmanr()`, `kendalltau()`
- Hypothesis tests: `ttest_1samp()`, `ttest_ind()`, `ttest_rel()`, `f_oneway()`, `chisquare()`, `mannwhitneyu()`, `wilcoxon()`, `kruskal()`, `friedmanchisquare()`, `shapiro()`, `normaltest()`, `kstest()`, `anderson()`
- Variance tests: `levene()`, `bartlett()`

### Metrics (`deepbox/metrics`)

- Classification: `accuracy()`, `precision()`, `recall()`, `f1Score()`, `fbetaScore()`, `rocAucScore()`, `rocCurve()`, `precisionRecallCurve()`, `confusionMatrix()`, `classificationReport()`, `logLoss()`, `hammingLoss()`, `jaccardScore()`, `matthewsCorrcoef()`, `cohenKappaScore()`, `balancedAccuracyScore()`, `averagePrecisionScore()`
- Regression: `mse()`, `rmse()`, `mae()`, `mape()`, `r2Score()`, `adjustedR2Score()`, `explainedVarianceScore()`, `maxError()`, `medianAbsoluteError()`
- Clustering: `silhouetteScore()`, `silhouetteSamples()`, `daviesBouldinScore()`, `calinskiHarabaszScore()`, `adjustedRandScore()`, `adjustedMutualInfoScore()`, `normalizedMutualInfoScore()`, `homogeneityScore()`, `completenessScore()`, `vMeasureScore()`, `fowlkesMallowsScore()`

### Preprocessing (`deepbox/preprocess`)

- Scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`
- Encoders: `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`
- Splitting: `trainTestSplit()`, `KFold`, `StratifiedKFold`, `GroupKFold`, `LeaveOneOut`, `LeavePOut`

### Machine Learning (`deepbox/ml`)

- Linear models: `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`
- Tree-based: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`
- Ensemble: `GradientBoostingClassifier`, `GradientBoostingRegressor`
- SVM: `LinearSVC`, `LinearSVR`
- Neighbors: `KNeighborsClassifier`, `KNeighborsRegressor`
- Naive Bayes: `GaussianNB`
- Clustering: `KMeans`, `DBSCAN`
- Dimensionality reduction: `PCA`
- Manifold learning: `TSNE`

### Neural Networks (`deepbox/nn`)

- Layers: `Linear`, `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`
- Recurrent: `RNN`, `LSTM`, `GRU`
- Attention: `MultiheadAttention`, `TransformerEncoderLayer`
- Normalization: `BatchNorm1d`, `LayerNorm`
- Regularization: `Dropout`
- Activations (as layers): `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Mish`, `Swish`, `Softmax`, `LogSoftmax`, `ELU`, `LeakyReLU`, `Softplus`
- Losses: `mseLoss()`, `maeLoss()`, `rmseLoss()`, `crossEntropyLoss()`, `binaryCrossEntropyLoss()`, `binaryCrossEntropyWithLogitsLoss()`, `huberLoss()`
- Containers: `Sequential`
- Module system: `Module` base class with parameter management, state dict, train/eval modes, hooks

### Optimization (`deepbox/optim`)

- Optimizers: `SGD` (with momentum), `Adam`, `AdamW`, `Nadam`, `RMSprop`, `Adagrad`, `AdaDelta`
- LR Schedulers: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `OneCycleLR`, `ReduceLROnPlateau`, `WarmupLR`

### Random (`deepbox/random`)

- Basic: `rand()`, `randn()`, `randint()`, `setSeed()`, `getSeed()`, `clearSeed()`
- Distributions: `uniform()`, `normal()`, `binomial()`, `poisson()`, `exponential()`, `gamma()`, `beta()`
- Sampling: `choice()`, `shuffle()`, `permutation()`

### Datasets (`deepbox/datasets`)

- Classic reference loaders: `loadIris()`, `loadDigits()`, `loadBreastCancer()`, `loadDiabetes()`, `loadLinnerud()`
- Classification loaders: `loadFlowersExtended()`, `loadLeafShapes()`, `loadFruitQuality()`, `loadSeedMorphology()`, `loadMoonsMulti()`, `loadConcentricRings()`, `loadSpiralArms()`, `loadGaussianIslands()`, `loadPerfectlySeparable()`
- Regression loaders: `loadPlantGrowth()`, `loadHousingMini()`, `loadEnergyEfficiency()`, `loadCropYield()`
- Clustering loaders: `loadCustomerSegments()`, `loadSensorStates()`
- Multi-output loaders: `loadFitnessScores()`, `loadWeatherOutcomes()`
- Integer-heavy loaders: `loadStudentPerformance()`, `loadTrafficConditions()`
- Synthetic generators: `makeClassification()`, `makeRegression()`, `makeBlobs()`, `makeMoons()`, `makeCircles()`, `makeGaussianQuantiles()`
- Utilities: `DataLoader` (batch iteration with shuffle)

### Visualization (`deepbox/plot`)

- Basic plots: `plot()`, `scatter()`, `bar()`, `hist()`, `boxplot()`, `violinplot()`, `pie()`
- Advanced: `heatmap()`, `contour()`, `contourf()`, `imshow()`
- ML plots: `plotConfusionMatrix()`, `plotRocCurve()`, `plotPrecisionRecallCurve()`, `plotLearningCurve()`, `plotValidationCurve()`, `plotDecisionBoundary()`
- Figure management: `figure()`, `subplot()`, `gca()`, `saveFig()`, `show()`
- Output: SVG (browser + Node.js), PNG (Node.js only)

### Infrastructure

- Zero runtime dependencies
- ESM + CommonJS dual output with type declarations
- Strict TypeScript (strict mode, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`)
- 260 test files, 4,344 tests
- Biome for linting and formatting
- CI/CD with GitHub Actions
- 6 enterprise-grade example projects
- 33 educational examples (00-32)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-10-03

A quality release. Every source file was reviewed line by line, and the results
were checked against NumPy, SciPy, scikit-learn, PyTorch and pandas. About 1,500
issues were fixed, many of them wrong results in 1.0.0. The release also makes
the API more consistent: tensors share one method surface, training works on
plain tensors, mixed dtypes promote instead of throwing, and every export has a
camelCase name.

No export, option or method was removed or renamed. Some results change because
they were wrong before, some dtypes change because of the new dtype rules, and
one type changed (`forward(Tensor)` on layers); all of this is listed under
"Upgrading from 1.0".

### Upgrading from 1.0

- **Dtypes.** Float operations keep the input float dtype: a float32 tensor stays
  float32 through `exp`, `sqrt`, `sum`, `mean`, activations and the rest (1.0.0
  returned float64). Integer input to an operation with fractional results
  (`mean`, `div`, `exp`, `softmax`, ...) gives float32. Integer reductions keep
  the integer dtype. Index results (`argsort`, `argmax`, `digitize`,
  `searchsorted`, `nonzero`) are int32.
- **Promotion.** Binary operations on tensors of different dtypes promote like
  PyTorch (int32 with float32 gives float32, float32 with float64 gives float64)
  instead of throwing `DTypeError`. JavaScript numbers never upcast a tensor.
- **Training.** `module.forward(tensor)` returns a `GradTensor` that tracks the
  weights when grad mode is on and the module has trainable parameters, so data
  no longer has to be wrapped in `parameter()`. Inside `noGrad()` it returns a
  plain `Tensor`. Parameter-free layers (pooling, dropout, activations,
  normalization without affine parameters) return a plain `Tensor` for plain
  input: read the result directly instead of through `.tensor`. In TypeScript,
  `layer.forward(tensor)` is now typed `AnyTensor` (`Tensor | GradTensor`)
  instead of `Tensor`. Code that annotated the result as `Tensor` should use
  `AnyTensor`; custom modules can declare a single
  `forward(x: AnyTensor): AnyTensor`.
- **Layers** compute in their parameter dtype and cast the input, as `Linear`
  already did. Weight initialization now matches PyTorch, so seeded models start
  from different weights.
- **Optimizers** skip parameters that have no gradient (PyTorch behavior)
  instead of throwing `NotFittedError`.
- **Seeded randomness.** Several modules replaced weak private generators with
  the library's seeded generator (trees and forests, bagging, k-means, splitters,
  `QuantileTransformer`, `DataFrame.sample`). The same seed still gives the same
  result, but not the same result as 1.0.0.
- **Stricter input checks.** Many functions now throw a typed error where 1.0.0
  returned silent garbage: NaN or infinite input to estimators, invalid buffers
  in `readParquet` and `readXlsx`, a truncated CIFAR-10 download, string or
  complex input to numeric code, and invalid parameters. A few error classes
  changed to the more precise one (for example `solve_banded` throws
  `InvalidParameterError` for an invalid band width).
- **SpectralNorm** registers the wrapped module as the child `module`, so state
  dicts saved from 1.0.0 models that contain `SpectralNorm` do not load.
- **Statistical tests** follow current SciPy: `mannwhitneyu` and `wilcoxon` use
  exact p-values for small samples, the `mannwhitneyu` statistic is U1, and
  several p-values and statistics were corrected (see Fixed).

### Added

- **Tensor methods.** `Tensor` and `GradTensor` share one method surface, so code
  reads the same with or without gradient tracking: `t.add(1).mul(2).sum()`,
  `t.T`, `t.matmul(w)`, `t.softmax(-1)`, `t.argmax(1)`, comparisons, rounding,
  `sort`, `flip`, `squeeze`, `unsqueeze`, `gather`, `item()` and more. Plain
  tensors also have `requiresGrad` (false), `grad` (null) and a `backward()` that
  explains why it cannot run.
- **ndarray:** `argmax`, `argmin`, `nonzero`, `argwhere`, `countNonzero`,
  `takeAlongAxis`, `putAlongAxis`, `nanvar`, `nanmedian`, `nanprod`,
  `nanargmin`, `nanargmax`, `nancumsum`, `nanquantile`; `unique` options
  (`returnIndex`, `returnInverse`, `returnCounts`, `axis`); batched `cross`;
  `dot` and `matmul` broadcast batch dimensions like `numpy.matmul`;
  activations `relu6`, `selu`, `celu`, `softsign`, `hardsigmoid`, `hardswish`,
  `logSigmoid`, `hardshrink`, `softshrink`; exact GELU through
  `gelu(t, { approximate: "none" })`. New differentiable `GradTensor` methods
  (`sin`, `cos`, `tan`, `log1p`, `expm1`, `maximum`, `minimum`, `cumsum`,
  `prod`, `std`, `var`, `softplus`, `mish`, `swish`, `selu`, `clone`) and
  multi-axis reductions.
- **core:** `promoteTypes`.
- **nn:** `MultiheadAttention` options `needWeights` and `keyPaddingMask`;
  Transformer layers `activation` (`"relu"` or `"gelu"`) and `normFirst`;
  convolution `dilation`, `groups` and `padding: "same" | "valid"`; pooling
  `ceilMode`; layers `ReLU6`, `LogSigmoid`, `CELU`, `Softshrink`, `Hardshrink`
  and `Threshold`; `Trainer` options `accumulationSteps` and
  `restoreBestWeights`; `tripletMarginLoss` supports autograd and a `swap`
  option; loss functions accept the output of `Module.forward` directly.
- **metrics:** multiclass `rocAucScore` (`multiClass: "ovr" | "ovo"`),
  `logLoss`, `jaccardScore` and `matthewsCorrcoef`; an options object
  `{ average, labels, zeroDivision, sampleWeight }` for `precision`, `recall`,
  `f1Score`, `fbetaScore` and `jaccardScore`; `sampleWeight` for the common
  classification and regression metrics; `meanAbsolutePercentageError`
  (scikit-learn semantics, a fraction).
- **dataframe:** `fillna` with per-column values and `method: "ffill" | "bfill"`,
  plus `ffill()` and `bfill()`; `corr` with `method` (`"pearson"`,
  `"spearman"`, `"kendall"`) and `minPeriods`; `sample` with `frac`, `replace`
  and `weights`; `groupBy` with `getGroup`, `nunique`, `quantile`, `transform`
  and named aggregation; `rolling` with `minPeriods` and `center`; `concat` with
  `join` and `ignoreIndex`; `valueCounts` with `normalize` and `dropna`.
- **ml:** trees and forests accept `sampleWeight`, `classWeight`,
  `minImpurityDecrease`, `maxLeafNodes` and `ccpAlpha`; every estimator has
  `clone()`; `Ridge` with `alpha: 0` on rank-deficient input returns the
  minimum-norm solution, as scikit-learn does.
- **stats:** an `alternative` option on `pearsonr`, `spearmanr`, `kendalltau`
  and `pointbiserialr`; `kendalltau` `variant` and `method`; `wilcoxon`
  `zeroMethod`; `benjaminiYekutieli` and `hochberg`.
- **datasets:** `fetch20Newsgroups` and `fetchIMDB` load the official archives
  by default (the 1.0.0 default URLs returned 404).
- **Names.** Every snake_case export now has a camelCase name, for example
  `matrixPower`, `blockDiag`, `solveBanded`, `toDatetime`, `dateRange`,
  `crossValScore`, `crossValidate`, `exportText`, `multivariateNormal`,
  `studentT`, `gaussianKde`, `ttest1samp`, `checkXY` and `checkArray`. The same
  holds for methods (`DataFrame.dropDuplicates`, `resetIndex`, `setIndex`,
  `pctChange`, `valueCounts`, `memoryUsage` and `pivotTable`; the `dt` accessor's
  `isLeapYear`, `dayName`, `dayOfWeek` and friends; `str.getDummies`;
  `style.highlightMax`, `highlightMin`, `highlightNull` and `backgroundGradient`)
  and for options (`leftOn` and `rightOn` in `DataFrame.merge`, `bwMethod` in
  `kdeplot`). When both spellings of an option are given, the camelCase one wins.
- **Tooling:** `typecheck:docs` type-checks every example and project against
  the source, and `prose:check` keeps em dashes out of the repository. Both run
  in `validate:all`.

### Deprecated

- The snake_case names that now have camelCase equivalents, and the lowercase
  `dt` names `dayofweek`, `dayofyear`, `weekofyear` and `daysinmonth`. They keep
  working and are marked `@deprecated` in the type declarations.

### Fixed

The list below names the most significant fixes. Every module received many
smaller ones (edge cases, validation, error messages, strided views, int64
input, documentation).

- **nn:** `Trainer` passed plain tensors to the model, so layers returned
  untracked results, no gradients reached the optimizer, and real models never
  trained. `MultiheadAttention` and the Transformer layers threw for float64 or
  float16 input. `EarlyStopping` kept state between `fit` calls. Init functions
  wrote to the wrong elements of non-contiguous tensors.
- **ml:** `OneClassSVM` flagged most training rows as outliers (88 percent with
  `nu = 0.1`; scikit-learn: 10 percent). `SVC` predictions, `PCA`
  projections, `HuberRegressor`, `LocalOutlierFactor` scores, `IsolationForest`
  thresholds, `CalibratedClassifierCV` (Platt and isotonic), and random forest
  bootstrapping and `maxFeatures` (which was ignored) now match scikit-learn.
  Estimators accept int64 input. Trees and forests return float64 results.
- **ndarray:** `cumsum` and `cumprod` without an axis returned the wrong shape;
  `median` rejected multiple axes; leaf gradients could share one buffer, so
  gradient clipping scaled it several times; `max` and `min` backward failed for
  float32; `floorDiv` and `mod` disagreed with NumPy for large or fractional
  values; int32 multiplication lost low bits; `corrcoef`, `cov` and `tensordot`
  returned silent garbage on bad input.
- **linalg:** non-symmetric `eig` (now a real Francis double-shift QR),
  Hessenberg and QR on matrices with tiny entries, the symmetry test used by
  `eig`, and float64 results for `trace`,
  `matrixPower`, `expm`, `logm` and `sqrtm`.
- **stats:** special functions are accurate to about 1e-14 (the old `erf` was
  accurate to about 1e-7); p-value tails no longer underflow; binomial, Poisson
  and related pmfs use the saddle-point method; `kstest`, `ks_2samp`,
  `anderson`, `mannwhitneyu` and `wilcoxon` match SciPy 1.17, including exact
  small-sample p-values; Welch degrees of freedom are no longer rounded down.
- **metrics:** metrics read strided and transposed tensors correctly, reject NaN
  labels, and match scikit-learn for `hingeLoss`, `coverageError`,
  `adjustedMutualInfoScore` and `fbetaScore` on multiclass input.
- **preprocess:** `TargetEncoder` leaked the target across folds; `RFE` and
  `RFECV` ranked eliminated features in reverse and did not work with the
  library's own tree models; `KBinsDiscretizer` put values on bin edges in the
  wrong bin; the seeded generator cycled after about 16,000 values.
- **dataframe:** `tail(n)` with `n` larger than the frame, `fromTensor` on
  views, sorting with infinities, `query` operator precedence, `eval` parsing,
  cumulative operations with NaN, `round` (now half-to-even), `ewm` with missing
  rows, `str.match` (now anchored like pandas) and nearest interpolation.
- **optim:** state dicts were live references, loaded non-atomically and could
  pair state with the wrong parameter; tied weights were updated twice per step;
  strided parameters were updated in the wrong elements; `LBFGS` ignored
  `lineSearchFn: "strong_wolfe"`; centered `RMSprop` and `RAdam` now match
  PyTorch exactly.
- **random:** shuffling a view changed elements outside it; `categorical`
  without replacement could loop forever; `dirichlet` with small concentrations
  returned uniform rows; `multivariateNormal` accepted invalid covariances.
- **datasets:** one value in the bundled data tables was wrong (all five now
  equal scikit-learn's); `makeFriedman2`, `makeFriedman3`,
  `makeSparseUncorrelated` and `makeLowRankMatrix` follow scikit-learn's
  definitions; `parseCSV` read empty cells as 0; `DataLoader.length` ignored the
  sampler; dataset ids are validated before they reach a URL.
- **plot:** PNG output now draws grids and text and blends translucent colors;
  `fill_between`, `area`, `stackedBar` and `groupedBar` draw correctly; `twinx`
  shares the x axis; log-scale ranges; animated SVG timelines; tick labels
  (2.5 was drawn as "3").
- **core:** `toJSON` turned NaN, Infinity and -0 into other values; `setConfig`
  reseeded the global generator on every call; `WorkerPool.reduce` applied the
  initial value once per chunk, and an invalid `maxWorkers` could hang the
  process. On WebGPU, NaN handling, `tanh` and `gelu` for large inputs,
  average-pool backward, and launches above 16.7 million elements (which
  returned zeros) are fixed.

### Performance

Hot paths in reductions, sorting, trees, metrics, DataFrame operations and
autograd were reworked where results stay identical, for example an
O(n log n) Kendall tau and a three-way quickselect that no longer degrades to
quadratic time on constant input. `binomial` uses BTPE (as NumPy does) for means
of 30 and above: a draw with n = 1e12 went from about 2.5 ms to a few
microseconds, with exact results.

### Verification

Beyond the unit tests (13,000+), every module was checked against its reference
library with randomized differential tests: reductions against NumPy,
estimators and metrics against scikit-learn, layers, gradients and optimizers
against PyTorch, DataFrame operations against pandas, and statistics against
SciPy. The unchanged 1.0.0 test suite was also run against this release, and
every difference is one of the changes listed above.

### Documentation

README, SKILL.md, all examples and projects were updated to the 1.5.0 API and
rewritten in plainer language. Defaults that differ from NumPy, pandas,
scikit-learn and PyTorch are documented where they apply.

## [1.0.0] - 2026-07-02

First stable release.

Compared with `v0.2.0`, `v1.0.0` broadens the public API across every major Deepbox module, hardens the framework around validation and serialization, expands examples and projects, and standardizes the package around the current subpath-export layout.

Ahead of release, the entire framework went through two hardening passes: an
API/consistency pass and a deep correctness audit that fixed silent
wrong-answer bugs across every subsystem, verified against NumPy / SciPy /
scikit-learn / PyTorch references.

### Added

#### Core (`deepbox/core`)

- Backend registry and typed backend interfaces
- **Real device execution**: `WebGpuBackend` now implements the new
  `KernelBackend` contract: tensors created on (or moved to) `webgpu` store
  their data in GPU memory and element-wise arithmetic, activations, matmul
  (`dot`), and full reductions execute as stride/broadcast-aware WGSL compute
  kernels. Includes `DeviceBuffer` handles, a pooled GPU allocator, and an
  injectable `gpu` provider for Node runtimes (`new WebGpuBackend({ gpu })`)
- **End-to-end GPU training**: the device kernel surface now covers everything a
  real training loop needs, all verified on GPU hardware (Dawn) against the CPU
  reference:
  - **Axis reductions** (`sum`/`mean`/`max`/`min` along one or more axes), so
    `softmax`, `logSoftmax`, `layerNorm` and `variance` compose and run on
    device.
  - **Batched matmul** (`ndim > 2`, with batch broadcasting), so attention
    (`Q·Kᵀ`, `·V`) and any batched linear algebra run on device.
  - **2-D convolution** (`im2col`/`col2im` gather/scatter kernels) and **2-D
    pooling** (`max`/`avg`), so `Conv2d` forward and backward run on device and
    CNNs train on the GPU.
  - Extra element-wise kernels: `gelu`, `erf`, `rsqrt`, `reciprocal`, `sign`,
    `expm1`, `log1p`, `softplus`, and a broadcast-aware `where` select.
  - **On-device autograd**: the reverse pass (axis-reduction backward,
    broadcast-sum, matmul/conv backward, strided-view materialization) stays
    resident on the device, and **every practical optimizer runs its step on
    device**: `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adamax`, `Nadam`,
    `RAdam`, `Adadelta`, `ASGD`, `Rprop`, `Lion`, `LAMB`, and `LARS`. A full
    forward → backward → update loop runs without host transfers. (`LBFGS` and
    `SparseAdam` require host-side line search / sparse scatter and throw a
    clear `DeviceError` on device.) MLP, transformer, and CNN training steps are
    covered by device regression tests.
  - **`MaxPool1d`/`MaxPool2d` on device** (forward + backward via a first-argmax
    gather kernel), alongside the already-composable `AvgPool`.
  - **Half precision**: `float16` tensors compute in true on-device half
    (WGSL `shader-f16`, 2 bytes/element, which halves the memory footprint for large
    models) and `bfloat16` tensors carry correct bf16 numerics; `.to('webgpu')`
    round-trips the half dtype. Mixed-precision ops throw rather than silently
    upcast.
  - Reduced per-dispatch overhead: uniform (metadata) buffers are pooled and
    reused across kernel launches instead of allocated and destroyed per op.
- **Working WASM SIMD backend**: `WasmBackend` instantiates embedded,
  build-time-compiled SIMD kernels (`WASM_BINARIES`, generated by
  `scripts/build-wasm.mjs`) and accelerates contiguous float32 arithmetic
  over zero-copy host storage with bit-identical results; `HostAcceleratorBackend`
  is the pluggable contract
- Device dispatch semantics modeled on PyTorch: strict same-device checks
  (`expected all tensors to be on the same device`), CPU-scalar promotion,
  and loud `DeviceError`s with transfer hints for unaccelerated device ops
- `Tensor.to(device)` / `Tensor.cpu()` (async, real data transfer),
  `Tensor.isDeviceTensor`, `Tensor.deviceBuffer`, `Tensor.dispose()`;
  device buffers are also released via a finalization registry
- `nn.Module.to(device)` now actually moves parameters, gradients, and
  buffers between devices (returns a `Promise`) instead of relabeling metadata
- Registry helpers `getKernelBackend()` / `getHostAccelerator()` and type
  guards `isKernelBackend()` / `isHostAcceleratorBackend()`
- `unregisterBackend(device)` removes a registered backend (the CPU backend
  is mandatory and cannot be removed)
- Logger and warning utilities
- Serialization helpers: `save()`, `load()`, `toJSON()`, `fromJSON()`
- Worker-pool helpers for parallel execution
- Expanded error and validation utilities

#### NDArray (`deepbox/ndarray`)

- FFT family: `fft()`, `ifft()`, `rfft()`, `irfft()`, `fft2()`, `ifft2()`, `fftn()`, `ifftn()`
- Einstein summation with `einsum()`
- Numerical helpers such as `interp()`, `trapz()`, `gradient()`, and `digitize()`
- Set, indexing, and manipulation helpers including `meshgrid()`, `index_select()`, `insert()`, `delete_()`, `searchsorted()`, `isin()`, `union1d()`, `intersect1d()`, and `setdiff1d()`
- Additional sparse CSR matrix operations and NaN-aware reductions
- Half-precision and complex array support

#### Linear Algebra (`deepbox/linalg`)

- Decompositions: `schur()`, `polar()`, `hessenberg()`
- Matrix functions: `expm()`, `logm()`, `sqrtm()`, `matrix_power()`, `kron()`, `block_diag()`
- Special matrices: `companion()`, `circulant()`, `hadamard()`, `hankel()`, `hilbert()`, `toeplitz()`, `vandermonde()`
- Additional dense and sparse solvers including `solve_banded()`, `sylvester()`, and sparse solve helpers

#### DataFrame (`deepbox/dataframe`)

- String and datetime accessors
- `MultiIndex` and `Categorical`
- Rolling, expanding, and exponentially weighted operations
- Richer reshaping and query-style workflows such as `assign()`, `query()`, `eval()`, `pivot_table()`, and `crosstab()`
- Excel and Parquet helpers exported from the module barrel, alongside CSV and JSON methods on `DataFrame`

#### Statistics (`deepbox/stats`)

- Confidence interval helpers
- Kernel density estimation via `GaussianKDE` and `gaussian_kde()`
- Multiple-comparison corrections
- Power analysis helpers
- Additional distributions and tests including contingency-style and non-parametric procedures

#### Metrics (`deepbox/metrics`)

- Extra metrics such as `brierScoreLoss()`, `coverageError()`, `meanPinballLoss()`, `zeroOneLoss()`, and `topKAccuracyScore()`
- Pairwise and ranking helpers including `ndcgScore()` and pairwise distance/similarity functions

#### Preprocessing (`deepbox/preprocess`)

- `KBinsDiscretizer`, `SplineTransformer`, `KNNImputer`, `MissingIndicator`
- Feature-selection helpers including `RFE`, `RFECV`, `SelectFromModel`, `SelectKBest`, and mutual-information scoring
- Text vectorizers including `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer`
- More splitters including repeated, grouped, shuffled, and time-series variants

#### Machine Learning (`deepbox/ml`)

- Ensemble methods: AdaBoost, Bagging, Voting, Stacking, and ExtraTrees variants
- Kernel and Nu SVM variants, plus `OneClassSVM`
- Additional neighbors, tree, clustering, decomposition, and manifold learners
- Gaussian processes, discriminant analysis, anomaly detection, multiclass meta-estimators, calibration, and inspection helpers
- Pipeline composition with `Pipeline`, `FeatureUnion`, `ColumnTransformer`, and `makePipeline`
- Hyperparameter search and CV helpers: `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score()`, and `cross_validate()`
- MLP estimators and estimator output/tag helpers

#### Neural Networks (`deepbox/nn`)

- Transformer encoder, decoder, and full-transformer building blocks
- Additional convolution, pooling, normalization, embedding, padding, and utility layers
- Expanded loss function surface including `nllLoss`, `klDivLoss`, `ctcLoss`, `tripletMarginLoss`, `gaussianNLLLoss`, and `poissonNLLLoss`
- Advanced containers, training helpers, initialization helpers, and gradient clipping

#### Optimization (`deepbox/optim`)

- Additional optimizers including `Adamax`, `RAdam`, `LBFGS`, `LARS`, `LAMB`, `SparseAdam`, `ASGD`, and `Rprop`
- Additional learning-rate schedulers including warm restarts, cyclic, polynomial, lambda, and sequential scheduling

#### Random (`deepbox/random`)

- `Generator` class
- Expanded random distributions and sampling helpers including multivariate and categorical-style sampling

#### Datasets (`deepbox/datasets`)

- Additional synthetic generators
- Remote CSV helpers
- Samplers and dataset transforms
- Image, text, and Kaggle-oriented fetching utilities

#### Plotting (`deepbox/plot`)

- Additional plot types such as `kdeplot()`, `stem()`, `strip()`, `radar()`, `waterfall()`, `quiver()`, `polar()`, `pairplot()`, and `jointplot()`
- More ML diagnostics including residuals, feature importance, elbow, silhouette, calibration, and dendrogram plots
- Animation, interactivity, palette helpers, and PDF export support

- **ml/base**: Added optional `clone()` method to the `Estimator` type, allowing estimators to create fresh unfitted copies with the same configuration. Implemented on 17 estimators: `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `KMeans`, `Ridge`, `StandardScaler`, `LinearSVC`, `LinearSVR`, plus `TransformerEncoderLayer`/`TransformerDecoderLayer`.
- **nn/layers/normalization**: Unified `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d` into a shared `_BatchNorm` base class, eliminating ~350 lines of duplicate code.
- **nn/layers/activations**: New activation layers: `Hardtanh`, `Tanhshrink`, `Softmin`, `Softmax2d`.
- **ndarray/autograd**: Added `hardtanh()` and `tanhshrink()` methods on `GradTensor` with full backward pass support.
- **ndarray/ops**: Added `hardtanh()` and `tanhshrink()` functional ops.
- **optim/optimizers**: Added `Lion` optimizer (EvoLved Sign Momentum, Chen et al. 2023) with sign-based updates and single momentum buffer.
- **ml/tree**: Added `ClassificationCriterion` type export.

### Changed

- Version advanced from `0.2.0` to `1.0.0`
- Node.js requirement standardized on `>= 24.13.0`
- Package output remains dual-format (`esm` and `cjs`) with `.d.ts` declarations for each subpath export
- Root package entry continues to expose module namespaces, while named APIs are available from subpath exports such as `deepbox/ndarray` and `deepbox/ml`
- Estimator parameter handling and output configuration were standardized across the expanded ML surface
- `Module.to(device)` now validates backend availability instead of only changing device metadata
- The current release tree contains 315 implementation files under `src/**/*.ts` (excluding `*.d.ts`), 421 Vitest files matching `test/**/*.test.ts`, 8,686 unit and integration tests, 50 example directories (`00`–`49`), and 9 project directories
- CI and release automation now validate the full package, dry-run `npm pack`, and publish with npm provenance

### API

- **Options-object forms for stats reductions.** `mean`, `std`, `variance`,
  `skewness`, and `kurtosis` now accept a named options object in the second
  argument position, such as `std(t, { ddof: 1 })`, `skewness(t, { bias: false })`,
  `kurtosis(t, { fisher: false })`, `mean(t, { axis: 1, keepdims: true })`, as
  the recommended alternative to the trailing positional flags, whose meaning
  differs by function at the same position (`std`'s 3rd arg is `keepdims`,
  `skewness`'s is `bias`). The historical positional signatures are fully
  preserved. New exported types: `MeanOptions`, `VarianceOptions`,
  `SkewnessOptions`, `KurtosisOptions`.
- **camelCase naming aliases (non-breaking).** To converge the public surface on
  camelCase while keeping every existing name working, deprecated aliases were
  added so both spellings resolve to the same function:
  - ndarray: `broadcastTo`, `columnStack`, `atleast1d`, `atleast2d`,
    `zerosLike`, `onesLike`, `emptyLike`, `fullLike`, `indexSelect`, `flipLr`,
    `flipUd` (alongside `broadcast_to`, `column_stack`, …).
  - stats/preprocess: `ttestInd`, `ttestRel`, `fOneway`, `chi2Contingency`,
    `ks2samp`, `fisherExact`, `fClassif`, `fRegression`, `mutualInfoClassif`,
    `mutualInfoRegression` (alongside the SciPy/sklearn snake_case names).
  - nn: `kaimingNormal_`/`kaimingNormal`, `kaimingUniform_`/`kaimingUniform`,
    `xavierNormal_`/`xavierNormal`, `xavierUniform_`/`xavierUniform`,
    `orthogonal`, `clipGradNorm_`/`clipGradNorm`,
    `clipGradValue_`/`clipGradValue`, plus `zeros`/`ones`/`constant`
    (alongside `kaiming_normal_`, `clip_grad_norm_`, …).

  The snake_case originals carry a `@deprecated Prefer <camelCase>` JSDoc tag but
  remain exported and unchanged in behavior.

### Performance

- **Axis-reduction backward is now allocation-free.** The `sum`/`mean`-along-
  axis gradient (the path softmax, cross-entropy, layernorm and attention hit on
  every training step) allocated two short-lived JS arrays *per input element*
  to convert flat indices to coordinates: tens of millions of GC-bound
  allocations per backward on a `[batch, seq, vocab]` logits tensor. It now
  walks the input in row-major order with a single reused coordinate odometer
  and tracks the physical upstream offset incrementally: zero per-element
  allocation. Gradients are unchanged (verified against finite differences for
  every axis and both `keepdims` settings).

- **Whole-tensor statistics full-reduction fast paths**: `skewness`,
  `kurtosis`, `geometricMean` and `harmonicMean` recomputed scalars and ran a
  `forEachIndexOffset` closure + dispatched accessor per element (several
  passes). They now gather one contiguous `Float64Array` and accumulate in a
  single narrowed loop; `skewness`/`kurtosis` fold m2/m3/m4 into one pass.
  `trimMean` further replaces its full `O(n log n)` sort with two `O(n)`
  quickselect partitions (trimmed mean = total − smallest k − largest k).
  Measured vs SciPy: `skewness` 5K 1.2x slower → **10.2x faster**, `kurtosis`
  5K 1.7x slower → **10.1x faster**, `trimMean` 1K 1.7x slower → **5.7x
  faster**, `geometricMean` 5K 1.1x slower → **2.7x faster**, `harmonicMean` 5K
  1.0x → **7.4x faster** (all now beat SciPy at every benched size; values
  match to ~1e-7). `zscore` gets the same contiguous-narrowed path
- **`nansum`/`nanmean` whole-tensor streaming**: the `axis=undefined` case
  went through the generic reducer that pushes every element into a boxed
  `number[]`; it now scans the contiguous typed array with a narrowed load and
  a running sum/count. `nanmean` 1K 1.2x slower → **7.0x faster**, `nansum` 1K
  3.5x slower → **2.9x faster**; the 100K cases are ~10x faster (now within ~2x
  of NumPy instead of ~15-19x)
- **DataFrame `rolling(window).mean()` sliding sum**: was `O(n·window)`,
  rebuilding a values array per position; now an `O(n)` running sum that adds
  the entering value, drops the value leaving the window and tracks the valid
  count so pandas' `min_periods=window` NaN rule is preserved exactly.
  `rolling(5).mean` 1K 1.7x slower → **2.4x faster**
- **NDArray movement/reduction pathologies de-pessimized**: several ops built
  a fresh coordinate array (and, for `nanmax`/`nanmin`, a full boxed `number[]`)
  per element via the generic strided path. Contiguous fast paths now index the
  typed buffers directly with an allocation-free odometer / slab copy:
  `roll` 100K 0.389ms → 0.013ms (~30x, `TypedArray.set`), `nanmin` 100K
  0.644ms → 0.070ms (~9x), `nanmax` 0.655ms → 0.093ms (~7x), `diff` 100K
  0.677ms → 0.111ms (~6x), `fliplr`/`flipud` 100x100 0.125ms → 0.028ms (~4.5x),
  `flip` 100K 0.923ms → 0.268ms (~3.4x), `where` 100K 2.1ms → 1.3ms. NumPy still
  wins the flip family (it returns O(1) negative-stride views) and bulk SIMD,
  but the embarrassing 50–3000x multipliers are gone
- **PCG32 seeded RNG rewritten with 32-bit limb arithmetic** (bit-identical
  sequences, verified across 1.6M samples): the BigInt implementation made
  every sample ~50x slower. Unseeded randomness now batches
  `crypto.getRandomValues` (4096 words/call) and bulk fills. `randn` 100K:
  ~56ms → 2.0ms; `randint` 1M: ~35ms → 7.9ms (SMI-domain modulo); dataset
  generators (`makeClassification`/`makeRegression`/`makeBlobs`/`makeMoons`)
  17–27x faster; `Linear`/`LSTM` layer initialization similarly
- **Same-shape strided binary arithmetic** (e.g. transpose-fed add/sub/mul/
  div) uses an odometer-walk kernel instead of the generic broadcast
  machinery: transposed 100x100 add 278µs → 11µs (~25x). Contiguous views
  with offsets now hit the fast path via `subarray`
- **2-D matmul (`dot`)** uses monomorphic i-k-j kernels with a float64 row
  accumulator (no per-FLOP read-modify-write on float32 outputs)
- **`einsum`** routes two-operand single-contraction patterns
  (`ij,jk->ik` and transposed variants) to the matmul kernel: 100x100
  contraction 75ms → 0.6ms (~125x)
- **`concatenate`/`stack`** use block copies for contiguous axis-0 inputs:
  2x100K 1.4ms → ~0.05ms (~27x)
- **`median`** copies into a typed buffer and runs typed quickselect:
  10K 1.5ms → 0.07ms (~20x)
- **`toeplitz`** fills a typed buffer directly (~9x)
- **`mutual_info_regression` / `mutual_info_classif`** replace the per-point
  full sort in the KSG estimator with an O(n·k) selection scan:
  1Kx20 ~2.2s → ~0.12s (~18x)
- `Tensor.data` is a plain own property again (device tensors get a
  per-instance throwing getter), `slice()`/`toArray()` hoist per-element
  checks (~6x), and device dispatch is guarded at op sites so pure-CPU
  flows pay only two string compares
- `Tensor.astype(dtype)` and `Tensor.item()` are now first-class methods
  (NumPy/PyTorch parity; `astype` was previously only available on
  `GradTensor`, and an error message referenced it on `Tensor` where it did
  not exist). `astype` handles all dtype pairs including string parse /
  stringify, materializes strided views, and throws converting non-finite
  values to `int64`; autograd's internal cast now delegates to it
- 1-D `sort`/`argsort` on lanes >= 8192 use an LSD radix sort on
  order-preserving 64-bit keys (O(n), NaNs last, stable). Sorting 100K values went from
  17.8 to 1.8 ms and argsort from 23.4 to 1.6 ms, verified element-identical
  against NumPy across the threshold with NaN/Inf/tie inputs in both dtypes
- Broad category pass driven by the benchmark suite:
  `CosineAnnealingWarmRestarts` steps are amortized O(1) (~350x on long
  runs); comparisons and logical ops gain contiguous fast paths (7-8x);
  `sort` uses comparator-free typed lane sorts (NaN-last, ~2.8x);
  pairwise Euclidean/Manhattan/cosine densify once (~10x); DataFrame
  `groupBy` aggregations read raw columns instead of constructing Series
  (~17x, now faster than pandas), `eval` compiles expressions once (~6x),
  `expanding()` windows stream in O(n) (~48x, faster than pandas);
  `HashingVectorizer` writes its buffer directly (~8x); t-SNE's exact path
  reuses flat buffers (~3.4x); decision-tree split search uses typed
  columns and packed sorts (~2.3x, compounding through all ensembles);
  KSG mutual information uses an O(n·k) selection scan (~18x).
- Stats and linear-algebra pass:
  `quantile`/`percentile` use a contiguous copy + comparator-free sort
  (~6x, now at parity with NumPy); `DataFrame.corr()` centers each column
  once and exploits matrix symmetry (~8x, now faster than pandas);
  `kendalltau` uses Knight's O(n log n) merge-sort inversion count instead
  of the O(n²) pair loop (scipy-exact output, scales to large n); rank-based
  correlations sort an index array instead of allocating per-element
  objects. Linear-algebra decompositions (LU/QR/SVD/Cholesky/eig/schur and
  the solvers) drop the bounds-checked element accessor in their inner loops.
  `svd`/`svdvals` now use the Golub–Reinsch algorithm (Householder
  bidiagonalization + implicit-shift QR) instead of one-sided Jacobi, which
  ran ~10–30 full O(n³) sweeps: svd 100×100 ~2.8x faster, svdvals ~5.8x, and
  `matrixRank` ~6.9x (cascading into `pinv` and `lstsq`). Reconstruction and
  singular values match NumPy to machine precision, including the
  numerically-singular case where the smallest value is ~machine epsilon (so
  `cond` of a singular matrix is now a very large finite number, matching
  NumPy/LAPACK, rather than exactly Infinity). The remaining large dense
  factorizations stay bounded by native LAPACK/BLAS. Deepbox now
  wins the majority of head-to-head benchmarks against NumPy / pandas /
  scikit-learn / PyTorch equivalents

### Fixed

#### Autograd (`deepbox/ndarray`)

- **`backward()` no longer overflows the stack on deep graphs.** The
  topological-sort that orders the backward pass was built with recursion, so
  the recursion depth equaled the graph depth. An unrolled model such as a
  long RNN over thousands of timesteps or a deep residual stack threw
  `RangeError: Maximum call stack size exceeded` during `backward()`. The
  traversal is now an explicit iterative worklist (identical ordering) and
  handles arbitrary depth. Regression test in `test/autograd.test.ts` exercises
  a 50,000-deep graph.

#### Neural-network training (`deepbox/nn`)

- **`Trainer` validation loop now runs under `noGrad`.** `model.eval()` only
  toggles train-mode layers (dropout/batchnorm); it does not disable gradient
  recording. The validation phase therefore built and immediately discarded a
  full backward graph for every batch, roughly doubling peak memory on large
  validation sets. The forward/loss is now wrapped in `noGrad`, which also
  models the correct inference pattern for users copying the loop.

#### DataFrame IO (`deepbox/dataframe`)

- **Parquet writer/reader rewritten to be genuinely spec-compliant** (verified
  against pyarrow, which could not deserialize the previous output): correct
  Thrift Compact field types and IDs, spec physical-type codes, bit-packed
  PLAIN booleans, UTF8-annotated strings, and metadata-driven reads. Nullable
  columns are written as OPTIONAL with RLE definition levels, so nulls
  round-trip instead of silently becoming `0`/`""`/`false`. Whole-column type
  inference prevents silent int truncation of mixed int/float columns; the
  `columns` read option is now honored; unsupported encodings/compression
  throw instead of returning wrong data
- **XLSX reader** guards against decompression bombs: zip entries are capped
  at 256 MiB decompressed (override with `maxEntryBytes` for trusted files)
- **XLSX reader** now unescapes XML entities (previously `"a & b"` came back
  as `"a &amp; b"`), reads deflate-compressed archives (openpyxl/Excel
  output), inline strings (`t="inlineStr"`), and honors the previously
  ignored `sheet` option (with a descriptive error listing available
  sheets). Null cells are written as empty and read back as `null` instead
  of `0`

#### Neural networks (`deepbox/nn`)

- `Tanhshrink` no longer throws a dtype mismatch for float32 inputs (the
  default dtype)
- `Softmax2d` now normalizes over the channel dimension at each spatial
  location (PyTorch semantics); it previously normalized over spatial
  positions per channel

#### Plotting (`deepbox/plot`)

- `Axes.annotate()`/`Axes.text()` validate that coordinates are finite
  numbers instead of silently emitting `NaN` positions into the SVG

#### Optimization (`deepbox/optim`)

- `LBFGS` rejects unknown `lineSearchFn` values instead of silently running
  without a line search

#### Datasets (`deepbox/datasets`)

- Kaggle helpers accept an `apiBaseUrl` override (mirrors, proxies, tests)


- **ndarray**: non-power-of-2 FFT returned the conjugate spectrum (Bluestein
  chirp sign); `concatenate([view])` ignored strides; `allclose`/`isclose`
  treated NaN and ±Inf as close; `matmul` backward crashed on 1-D operands;
  repeated `backward()` propagated stale intermediate gradients; `sort`,
  `argsort`, `median`, `unique` mishandled NaN; int32 `sum` silently overflowed;
  bool `add` stored out-of-range values; `pow(1, Inf)`; `fftn` ignored strides
  and negative axes; `einsum` ellipsis/int64/output-label validation;
  `Complex.div` (Smith's algorithm) and `abs` (hypot); float16 double-rounding;
  negative-step slice `end:-1`; CSR `fromCOO` now sums duplicates; `round`
  half-to-even; `view()` custom-stride backward.
- **linalg**: `eigh`/`eigvalsh` rewritten as cyclic Jacobi (was one rotation per
  sweep and wrong beyond ~10×10); `solve_banded` general path rewritten with
  LAPACK-style banded LU + pivoting (returned garbage on any row swap); `eig`
  and `schur` gained Wilkinson + exceptional shifts; `sylvester`/`lyapunov` now
  raise on singular systems instead of zero-filling; SVD pre-scales to avoid
  over/underflow at extreme magnitudes; `qr` uses a scale-relative threshold;
  `expm` rewritten with scaling-and-squaring + Padé (handles complex spectra);
  `slogdet`/`lstsq` residuals return float64; negative vector-norm orders.
- **nn**: entire families of layers were silently untrainable (detached
  forward) and now propagate gradients: Conv3d, ConvTranspose1d/2d, all 1D/3D
  and adaptive pooling, Embedding/EmbeddingBag, RNN/LSTM/GRU (rewritten as
  differentiable graphs), Dropout2d/AlphaDropout, Upsample, and all Pad2d
  layers. ConvTranspose2d no longer crashes with default bias; MaxPool2d uses
  −inf padding and backprops in float32; TransformerDecoderLayer applies a
  causal mask; `crossEntropyLoss` accepts float64 logits.
- **ml**: GaussianProcessClassifier (was worse than chance) now uses the correct
  Laplace posterior mean; Ridge `sag` no longer diverges to ±Infinity; BallTree
  `query` returns correct neighbors; LogisticRegression regularization is no
  longer n_samples× too strong and the solver runs to convergence; seeded
  DecisionTree/RandomForest no longer freeze the feature subset per tree;
  `feature_importances_` uses mean-impurity-decrease (MDI); tree `criterion`
  (entropy/log_loss) is honored; GradientBoostingClassifier applies the Newton
  leaf update; ComplementNB drops the class prior for >1 class; FastICA whitens
  with the √n factor (was returning a random rotation); LocalOutlierFactor
  `predict` computes true novelty LOF; PCA randomized `explainedVarianceRatio`
  no longer sums to 1; cross-validation folds are shuffled (were degenerate on
  sorted data); Isomap raises on disconnected graphs; Birch/FastICA getters no
  longer read past their buffers.
- **dataframe**: `Date` values now hash by timestamp (all dates previously
  collapsed to one join/group key); the datetime accessor parses naive strings
  in local time to match extraction; `eval("a == 3")` filters instead of
  overwriting the column; `query` supports column-vs-column comparisons; CSV
  reading strips a BOM, keeps all-empty rows as nulls, and no longer coerces
  whitespace to 0 or `" 007 "` to 7; rolling windows require a full window
  (min_periods); descending sort places nulls last; `pivot_table` no longer
  collapses distinct keys that stringify alike.
- **stats/random/metrics**: `F.ppf` and `beta.ppf` (and chi²/gamma tails) use
  bracketed inversion instead of a diverging Newton iteration; Poisson (large
  μ) and negative-binomial (small p) samplers no longer cap due to `exp(-λ)`
  underflow; `binom.pmf` at p∈{0,1}; NDCG uses linear gains (sklearn parity);
  `mannwhitneyu` applies the continuity correction for all sample sizes;
  `norm.sf` computes the upper tail directly; quantile/percentile/pairwise/ROC
  outputs are float64.
- **optim**: schedulers perform the construction-time step (PyTorch semantics),
  fixing a one-epoch shift that made OneCycleLR/WarmupLR run the first epoch at
  the wrong LR; ReduceLROnPlateau uses a relative threshold by default with a
  corrected cooldown; ASGD applies the decoupled decay term; SGD skips
  dampening on the first momentum step.
- **plot**: XSS in interactive HTML output (data-point labels are escaped);
  log-scale axes place decade ticks with correct labels; data series are
  clipped to the axes viewport; box/violin quartiles use linear interpolation
  (matplotlib parity).
- **preprocess/datasets**: SimpleImputer, KNNImputer, MissingIndicator,
  PolynomialFeatures, Binarizer, and SplineTransformer are now stride-aware
  (correct on transposed/sliced views); StratifiedShuffleSplit handles string
  labels; stratified `trainTestSplit` validates singleton classes regardless of
  `randomState`; TargetEncoder `fitTransform` cross-fits to prevent target
  leakage; the remote-dataset default timeout is now applied and its timer
  cleared.

- **nn/layers**: `TransformerEncoder` and `TransformerDecoder` now create distinct layer instances per stack layer instead of reusing the same object. Fixes a critical bug where all encoder/decoder layers shared identical weight parameters.
- **nn/losses**: All loss functions now properly support `GradTensor` inputs with overload signatures. Previously only `mseLoss` preserved the computation graph for autograd; `maeLoss`, `rmseLoss`, `binaryCrossEntropyLoss`, and `marginRankingLoss` now also support gradient tracking.
- **optim/schedulers**: `WarmupLR` no longer desynchronizes the after-scheduler's epoch counter. The transition from warmup to the wrapped scheduler now properly delegates without stale intermediate LR values.
- **ml/model_selection**: `cross_val_score` now prefers the `clone()` method on estimators when available, with an improved constructor-based fallback for estimators that accept an options object.
- **nn/module**: `freezeParameters()` and `unfreezeParameters()` now scan array properties to update stale parameter references in modules that store parameters in arrays (e.g., RNN, LSTM, GRU weight arrays).
- **core/utils**: Extracted `shapesEqual()` to a shared utility in `deepbox/core`, eliminating three duplicate implementations across the codebase.
- **nn/layers**: `SiLU` now extends `Swish` instead of duplicating its implementation.
- **ml/tree**: Restored `criterion` parameter (gini/entropy/log_loss) on `DecisionTreeClassifier`, working `setParams()` for both Classifier and Regressor, and `export_text()` for ASCII tree visualization.

- Export coverage gaps in `deepbox/nn`, including `gaussianNLLLoss` and `poissonNLLLoss`
- GLU activation slicing behavior
- `plotDendrogram()` implementation
- Vote normalization in `OneVsOneClassifier.predictProba()`
- Reconstruction behavior in `LatentDirichletAllocation.inverseTransform()`
- Numerous audit-tracked issues from the `v0.2.0` parity and consistency pass

### Documentation

- Refreshed release-facing docs for `v1.0.0`
- Replaced `LLMs.txt` with `SKILL.md` as the agent guide for writing Deepbox code, examples, and projects
- Updated README, contributing guidance, changelog language, and security policy to reflect the current package layout

## [0.2.0] - 2026-02-14

78 source files changed (1,418 insertions, 674 deletions). Bug fixes, type safety improvements, multiclass support, and documentation overhaul. Consistency pass, and enterprise-grade hardening.

_Numbers in the `0.2.0` sections below (tests, examples, projects, coverage) describe the **v0.2.0** tree; see **[1.0.0]** above for the current package._

### Added

- **`GradTensor.isGradTensor()`**: static duck-typing method for cross-module `instanceof` compatibility
- **`GradTensor` public constructor**: two overloads: `(data, options?)` for users and `({tensor, requiresGrad, prev, backward})` for internals
- **`Tensor.slice()` instance method**: `t.slice(...)` in addition to the standalone `slice(t, ...)`
- **`ScalarDType` and `ElementOf<D>` types**: enables `tensor([1,2,3]).at(0)` to return `number` instead of `unknown`
- **`DataValue` type export** from `deepbox/dataframe`
- **`loadDigits().images`**: reshaped `[1797, 8, 8]` tensor matching sklearn's `.images` attribute
- **`makeClassification({ flipY })` parameter**: label noise injection (default 1%)
- **`DBSCAN.nClusters` getter**: returns number of discovered clusters (excludes noise)
- **`PolynomialFeatures`** transformer in `deepbox/preprocess`
- **Vector-matrix `dot()` support**: `dot(1D, 2D)` now works correctly
- **`norm()` overloads**: `norm(x)` returns `number`; `norm(x, ord, axis)` returns `Tensor | number`

### Fixed

- **`mseLoss` / `crossEntropyLoss` / `binaryCrossEntropyWithLogitsLoss`**: replaced `instanceof GradTensor` with `GradTensor.isGradTensor()` to fix silent loss-of-gradient bug across module boundaries
- **`GradientBoostingClassifier`**: added multiclass support via One-vs-Rest strategy (was binary-only)
- **`LinearSVC`**: added multiclass support via One-vs-Rest strategy (was binary-only)
- **`crossEntropyLoss`**: 1D GradTensor target now works; overload signatures accept `AnyTensor`
- **`DataLoader` iterator type**: return type now conditional `[Tensor, Tensor] | [Tensor]` instead of `never`
- **`DataFrame.filter()` row type**: changed from `unknown` to `Record<string, any>` for usability
- **`precision()` / `recall()` / `f1Score()`**: auto-detect multiclass and default to `"weighted"` averaging instead of `"binary"`
- **`f1Score()`**: accepts both `string` and `{ average: string }` argument forms
- **`relu()` / `leakyRelu()` / `elu()` return types**: narrowed to `Tensor<Shape, ScalarDType>`
- **`LinearRegression.predict()` return type**: narrowed to `Tensor<Shape, ScalarDType>`
- **`ensureNumericDType()` context parameter**: made optional (default: `"operation"`)
- **JSDoc `@see` links**: all 40+ `deepbox.dev` references verified against actual docs routes
- **Documentation code snippets**: 7 broken snippets fixed across datasets, getting-started, ml, optim, plot, preprocess content files

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
- Coverage thresholds enforced at release time: 89% lines, 90% functions, 72% branches, 88% statements (thresholds were later relaxed in `v1.0.0`; see `vitest.config.ts` on `main`)

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

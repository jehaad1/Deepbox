---
name: deepbox-agent-guide
description: Use when writing Deepbox code, examples, or full projects so imports, module selection, dtypes, training loops, errors, backends, docs pages, and framework-specific patterns stay accurate.
---

# Deepbox Skill Guide

This file is for AI agents, code generators, and automation that need to build applications and projects with Deepbox.

Use it when generating Deepbox code, selecting modules, following the website docs, or turning a task into a runnable Deepbox example or project.

## Project Facts

- Package: `deepbox`
- Release line described by this file: `v1.5.0`
- Runtime requirement: Node.js `>= 24.13.0`
- Package shape: ESM + CommonJS + `.d.ts`
- Runtime dependencies: `0`
- Root import behavior: `deepbox` exports namespaces, not direct named APIs
- Compatibility: 1.5.0 is backward compatible with 1.0.0. No export was removed or renamed. Deprecated snake_case names still work. Some results changed because 1.0.0 returned wrong values (see "Upgrading from 1.0.0").

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

The root namespaces are `core`, `ndarray`, `linalg`, `dataframe`, `stats`, `metrics`, `preprocess`, `ml`, `nn`, `optim`, `random`, `datasets`, and `plot`.

### Avoid

Do not generate this:

```ts
import { tensor, LinearRegression } from "deepbox";
```

The root entry does not export those names directly.

### Naming

- Use camelCase for every function, option, and method: `matrixPower`, `toDatetime`, `ttestInd`, `crossValScore`, `multivariateNormal`, `checkXY`, `clipGradNorm_`, `kaimingNormal_`.
- The snake_case names (`matrix_power`, `to_datetime`, `ttest_ind`, `cross_val_score`, `multivariate_normal`, `check_X_y`, `clip_grad_norm_`, `kaiming_normal_`) still exist and are marked `@deprecated`. The same holds for methods (`dropDuplicates`, `resetIndex`, `setIndex`, `pctChange`, `memoryUsage`, `pivotTable`, `valueCounts`, `dt.dayOfWeek()`, `dt.isLeapYear()`, `str.getDummies()`, `style.highlightMax()`) and options (`leftOn` and `rightOn` in `merge`, `bwMethod` in `gaussianKde` and `kdeplot`). When both spellings of an option are given, the camelCase one wins. Write the snake_case names only when migrating old code.
- Trailing underscores on `nn` initializers (`xavierUniform_`, `zeros_`) and gradient clippers (`clipGradNorm_`) follow the PyTorch convention and are part of the canonical name.

## Source of Truth

Assume the agent may only have the package repo and the public website.

When you need the real current Deepbox feature set for writing code, use this precedence order:

1. the installed package types (`node_modules/deepbox/dist/<module>/index.d.ts`), which describe the exact version in use
2. the live website at `https://deepbox.dev`
3. `https://deepbox.dev/docs`
4. `https://deepbox.dev/examples`
5. `https://deepbox.dev/projects`
6. local examples and projects if this repo is present
7. local source barrels such as `src/<module>/index.ts` as a verification step

Important:

- The website is the primary discovery surface for agents writing Deepbox code.
- Examples and projects matter as much as API pages because they show intended usage.
- If local source or the installed `.d.ts` files are available, use them to verify exports and signatures before making strong claims about named APIs.
- Do not use a name from memory of NumPy, scikit-learn, pandas, or PyTorch without checking that Deepbox exports it. Several names differ (for example `matmul` is a `Tensor` method and has no `ndarray` function; use `dot()` for the function form).

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
| `deepbox/core` | shared types, errors, config, validation, serialization, backends | `DType`, `Shape`, `InvalidParameterError`, `promoteTypes()`, `save()`, `registerBackend()` |
| `deepbox/ndarray` | tensors, autograd, sparse arrays, general numerical ops | `Tensor`, `GradTensor`, `tensor()`, `parameter()`, `noGrad()`, `einsum()` |
| `deepbox/linalg` | decompositions, inverse problems, matrix functions | `svd()`, `solve()`, `matrixPower()` |
| `deepbox/dataframe` | tabular data workflows | `DataFrame`, `Series`, `toDatetime()` |
| `deepbox/stats` | descriptive stats, tests, distributions, KDE, power analysis | `mean()`, `ttestInd()`, `norm`, `GaussianKDE` |
| `deepbox/metrics` | evaluation metrics | `accuracy()`, `mse()`, `silhouetteScore()`, `meanAbsolutePercentageError()` |
| `deepbox/preprocess` | feature prep and data splitting | `StandardScaler`, `OneHotEncoder`, `KFold`, `trainTestSplit()` |
| `deepbox/ml` | classical ML estimators and pipelines | `RandomForestClassifier`, `Pipeline`, `GridSearchCV`, `crossValScore()` |
| `deepbox/nn` | neural-network modules, losses, training helpers | `Module`, `Sequential`, `Linear`, `crossEntropyLoss()`, `Trainer` |
| `deepbox/optim` | optimizers and schedulers | `Adam`, `SGD`, `CosineAnnealingLR` |
| `deepbox/random` | seeded randomness and distributions | `setSeed()`, `Generator`, `multivariateNormal()` |
| `deepbox/datasets` | built-in datasets, samplers, remote fetchers | `loadIris()`, `DataLoader`, `makeBlobs()` |
| `deepbox/plot` | figures and plots | `figure()`, `plot()`, `heatmap()`, `saveFig()` |

Placement rules that are easy to get wrong:

- `Pipeline`, `makePipeline`, `ColumnTransformer`, `FeatureUnion`, `GridSearchCV`, `crossValScore` live in `deepbox/ml`, not `deepbox/preprocess`.
- `trainTestSplit` and the cross-validation splitters (`KFold`, `StratifiedKFold`, ...) live in `deepbox/preprocess`.
- Loss functions (`mseLoss`, `crossEntropyLoss`) live in `deepbox/nn`. Evaluation metrics (`mse`, `accuracy`) live in `deepbox/metrics`.
- `corrcoef` and `cov` exist in both `deepbox/ndarray` and `deepbox/stats`.
- Backend classes (`WasmBackend`, `WebGpuBackend`) and `registerBackend()` live in `deepbox/core`.

## Website Page Map

Use this table when you need exact places to fetch current features quickly from the website while writing code.

| Area | Website pages | Local source to verify exports |
| --- | --- | --- |
| Home | `https://deepbox.dev/` | `src/index.ts` if available |
| Docs index | `https://deepbox.dev/docs` | module barrels in `src/` if available |
| Examples index | `https://deepbox.dev/examples` | `docs/examples/*` if available |
| Projects index | `https://deepbox.dev/projects` | `docs/projects/*` if available |
| Core docs | `/docs/core-types`, `/docs/core-config`, `/docs/core-errors`, `/docs/core-utils` | `src/core/index.ts` |
| Devices and backends | `/docs/devices-and-execution` | `src/core/backend/`, `src/ndarray/ops/device_dispatch.ts` |
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
- `devices-and-execution`
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

- `Tensor` is the plain tensor. `GradTensor` wraps a `Tensor` (available as `.tensor`) and records operations for reverse-mode autodiff.
- `Tensor` and `GradTensor` share one method surface, in the style of PyTorch: `t.add(1).mul(2).sum()`, `t.T`, `t.matmul(w)`, `t.mean()`, `t.argmax(1)`, `t.softmax(-1)`, `t.item()`. Prefer chaining in new code.
- The functional API still works and is the way to reach ops that are not methods: `add(a, b)`, `einsum(...)`, `where(...)`, `fft(...)`.
- Plain `Tensor` exposes `requiresGrad` (always `false`), `grad` (always `null`), `detach()` (returns itself) and `backward()` (throws `DeepboxError`). This lets code that receives `Tensor | GradTensor` call them without narrowing. `AnyTensor` is the union type.
- `GradTensor` returns plain `Tensor` for results that carry no gradient: `argmax`, `argmin`, `argsort`, comparisons (`eq`, `gt`, ...), `any`, `all`, `isnan`.
- `parameter(data)` creates a trainable leaf `GradTensor` (`requiresGrad: true`). Use it for a hand-written optimization problem, not for training data.
- `noGrad(fn)` runs a synchronous callback with graph recording off and returns its result. An async callback throws.

### Training uses plain tensors

When grad mode is on and a module has trainable parameters, `model.forward(x)` on a plain `Tensor` returns a `GradTensor` that tracks the weights. The input does not need `parameter(...)`.

- A loss computed from that output has `backward()`.
- Inside `noGrad()` the same call returns a plain `Tensor`.
- `model.eval()` changes layer behavior (dropout, batch norm) but does not turn tracking off. Wrap inference in `noGrad()`.
- A module with no trainable parameters, or with all parameters frozen, returns a plain `Tensor`. Calling `backward()` on it throws `DeepboxError`.
- Parameter-free layers (`ReLU`, `Dropout`, pooling, `LayerNorm` without affine) return a plain `Tensor` for plain input. Read the result directly, not through `.tensor`.
- Optimizers skip parameters that have no gradient instead of throwing.

### Dtype Rules

- `tensor()` creates `float32` by default (`getDtype()` returns the global default; `setDtype()` changes it). NumPy defaults to float64, so pass `{ dtype: "float64" }` when matching NumPy numbers exactly.
- Tensor-tensor binary ops promote like PyTorch instead of throwing. The ladder is `bool < uint8 < int32 < int64 < float16, bfloat16 < float32 < float64`. In the same category the wider type wins, an integer with a float gives the float type, and `float16` with `bfloat16` gives `float32`. `promoteTypes(a, b)` from `deepbox/core` returns the result type.
- A JavaScript number never upcasts a tensor: `int32 + 1` stays `int32`, `float32 + 1.5` stays `float32`, `int32 + 1.5` gives `float32`.
- Float ops keep the input float dtype: `exp`, `sqrt`, `log`, `sum`, `mean`, softmax, and activations on `float32` return `float32`, and on `float64` return `float64`.
- Integer input to ops with fractional results (`mean`, `div`, `exp`, `log`, `softmax`, `reciprocal`) gives `float32`.
- Index results are `int32`: `argmax`, `argmin`, `argsort`, `digitize`, `searchsorted`, `nonzero`.
- Layers compute in their parameter dtype and cast the input to it. Parameter-free layers keep the input float dtype.
- `complex64` and `complex128` appear in `DType` but a tensor cannot be created with them yet (`tensor()` and `zeros()` throw `DTypeError`). `Complex`, `Complex64Array` and `Complex128Array` exist as standalone array types.
- `int64` tensors are backed by `BigInt64Array`, so their elements are `bigint`.
- Convert explicitly with `t.astype("float64")`.

### Classical ML

- Estimators generally follow `fit()`, `predict()`, `transform()`, and `fitTransform()` style patterns.
- Many estimators also support `getParams()` and `setParams()`. Every estimator has `clone()`, which returns an unfitted copy with the same parameters.
- Calling `predict()` or `transform()` before `fit()` throws `NotFittedError`.
- Trees and forests accept `sampleWeight` in `fit()` and the options `classWeight`, `minImpurityDecrease`, `maxLeafNodes`, `ccpAlpha`.
- Pipeline-style composition lives in `deepbox/ml`, not `deepbox/preprocess`.
- Averaged classification metrics (`precision`, `recall`, `f1Score`, `fbetaScore`, `jaccardScore`) accept a positional `average` or an options object `{ average, labels, zeroDivision, sampleWeight }`.

### Neural Networks

- `deepbox/nn` follows a `Module`/`forward()` mental model similar to PyTorch.
- Parameters come from modules (`model.parameters()`) and are passed to optimizers in `deepbox/optim`.
- Training helpers such as `Trainer`, `EarlyStopping`, `ModelCheckpoint`, and `GradientAccumulator` live in `deepbox/nn`. `Trainer` supports `accumulationSteps`, `restoreBestWeights`, `maxGradNorm`, and `earlyStopping`.
- Newer layer options: `MultiheadAttention` takes `needWeights` and `keyPaddingMask`; Transformer layers take `activation` and `normFirst`; conv layers take `dilation`, `groups`, and `padding: "same" | "valid"`; pooling layers take `ceilMode`.

### Plotting

- `deepbox/plot` is figure-oriented.
- `figure()`, `subplot()`, and `gca()` manage state.
- `show()` renders to SVG by default.
- `saveFig()` writes SVG, PNG, or PDF and returns a `Promise`.

### DataFrames

- `DataFrame` and `Series` are object-oriented APIs.
- CSV and JSON helpers largely live as `DataFrame` methods.
- Excel and Parquet helpers are exported from the module barrel.

## Defaults That Differ From Reference Libraries

These defaults stay as they are in 1.x for compatibility and may change in 2.0. When a task says "match scikit-learn" or "match PyTorch", pass the option shown.

| API | Deepbox default | Reference | To match the reference |
| --- | --- | --- | --- |
| `tensor()` dtype | `float32` | NumPy: `float64` | `{ dtype: "float64" }` |
| `gelu`, `GELU` | tanh approximation | PyTorch: exact | `{ approximate: "none" }` |
| `mape` | percentage (10 means 10%) | scikit-learn: fraction | use `meanAbsolutePercentageError` |
| `PowerTransformer` `standardize` | `false` | scikit-learn: `true` | `{ standardize: true }` |
| `DataFrame.ewm` `adjust`, `bias` | `false`, `true` | pandas: `true`, `false` | `{ adjust: true, bias: false }` |
| `DataFrame.groupBy` | keys in first-appearance order, missing keys kept | pandas: sorted, missing dropped | `{ sort: true, dropna: true }` |
| `DataFrame.valueCounts` | missing values counted | pandas: dropped | `{ dropna: true }` |
| `Series.str.replace` `regex` (third argument) | `true` | pandas: `false` | `replace(pat, repl, false)` |
| `InstanceNorm` `affine` | `true` | PyTorch: `false` | `{ affine: false }` |
| `Upsample` `alignCorners` (bilinear) | `true` | PyTorch: `false` | `{ alignCorners: false }` |
| `RandomForest*` `maxDepth` | `10` | scikit-learn: unlimited | `{ maxDepth: Infinity }` |
| `LinearSVC` `loss` | `"hinge"` | scikit-learn: squared hinge | set `loss` explicitly |
| `GaussianMixture` `covarianceType` | `"diag"` | scikit-learn: `"full"` | `{ covarianceType: "full" }` |
| `KernelRidge` `gamma` | `1.0` | scikit-learn: `1 / nFeatures` | pass `gamma` |
| `RMSNorm` `eps` | `1e-5` | PyTorch: dtype machine epsilon | pass `eps` |
| Averaged metrics, no `average` | `"binary"` for two classes, `"weighted"` for more | scikit-learn: `"binary"` (errors on multiclass) | pass `average` |
| `imshow` row 0 | bottom of the y axis | matplotlib: top | `{ origin: "upper" }` |
| `clip(t, min, max)` with `min > max` | throws `InvalidParameterError` | NumPy, PyTorch: returns `max` | n/a |
| `FullTransformer` stacks | no final LayerNorm | PyTorch `nn.Transformer`: final LayerNorm | `{ finalNorm: true }` |

Other facts that differ from NumPy and SciPy:

- Seeded random streams are not bit-identical to NumPy or PyTorch. Results are reproducible within Deepbox for a fixed seed, not across libraries.
- `mannwhitneyu` and `wilcoxon` use exact p-values for small samples, like SciPy's default `method="auto"`.

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

Every Deepbox error extends `DeepboxError` directly, and `DeepboxError` extends `Error`. No error class extends another one, so catch each class you care about (or `DeepboxError` for all of them).

- `DeepboxError`: base class
- `InvalidParameterError`: bad function or constructor arguments, invalid options
- `ShapeError`: incompatible or invalid shapes. Broadcast mismatches in tensor ops throw `ShapeError`
- `BroadcastError`: exported for user code and carries `shape1` and `shape2`. Library ops currently report broadcast failures as `ShapeError`
- `DTypeError`: unsupported dtype, such as a complex or string tensor where it is not defined, or a non-float tensor moved to a kernel device
- `IndexError`: invalid indexing
- `NotFittedError`: estimator used before `fit()`
- `ConvergenceError`: iterative algorithm failed to converge
- `DeviceError`: unavailable device, no registered backend, unsupported op on a device, mixed-device operands, use of a disposed tensor
- `MemoryError`: allocation or memory-bound failure
- `DataValidationError`: invalid input data, such as a corrupt Parquet or XLSX buffer or an unknown aggregation name
- `NotImplementedError`: declared surface not implemented

```ts
import { tensor } from "deepbox/ndarray";
import { DeepboxError, ShapeError } from "deepbox/core";

try {
  tensor([[1, 2], [3, 4]]).add(tensor([1, 2, 3]));
} catch (error) {
  if (error instanceof ShapeError) console.log("shape problem");
  else if (error instanceof DeepboxError) console.log("other Deepbox error");
  else throw error;
}
```

## Public API Inventory

This section lists exports that exist in the 1.5.0 barrels. It is still not a replacement for the barrel files; treat it as a fast map of what exists and where to fetch the exhaustive list. Deprecated snake_case aliases are left out unless noted.

### `deepbox/core`

- Types and constants: `Axis`, `Device`, `DType`, `ElementOf`, `ExtendedTypedArray`, `ScalarDType`, `Shape`, `TensorLike`, `TensorStorage`, `TypedArray`, `DEVICES`, `DTYPES`, `isDevice()`, `isDType()`
- Dtype helpers: `promoteTypes()`, `isFloatDType()`, `toFloatDType()`, `dtypeToTypedArrayCtor()`, `ensureNumericDType()`
- Config: `getConfig()`, `getDevice()`, `getDtype()`, `getSeed()`, `resetConfig()`, `setConfig()`, `setDevice()`, `setDtype()`, `setSeed()`
- Errors: `DeepboxError`, `BroadcastError`, `ConvergenceError`, `DataValidationError`, `DeviceError`, `DTypeError`, `IndexError`, `InvalidParameterError`, `MemoryError`, `NotFittedError`, `NotImplementedError`, `ShapeError`
- Backend: `CpuBackend`, `WasmBackend`, `WebGpuBackend`, `registerBackend()`, `unregisterBackend()`, `getBackend()`, `getKernelBackend()`, `getHostAccelerator()`, `isBackendAvailable()`, `isKernelBackend()`, `isHostAcceleratorBackend()`, `listBackends()`, `WGSL_SHADERS`, `WAT_MODULES`, `WASM_BINARIES`
- Validation and utilities: `checkArray()`, `checkXY()`, `checkIsFitted()`, `validateArray()`, `validateShape()`, `validateDtype()`, `validateDevice()`, `validateInteger()`, `validatePositive()`, `validateNonNegative()`, `validateRange()`, `validateOneOf()`, `shapeToSize()`, `shapesEqual()`, `normalizeAxis()`, `normalizeAxes()`, typed-array access helpers
- Serialization, logging, warnings, parallelism: `save()`, `load()`, `toJSON()`, `fromJSON()`, `Logger`, `setLogHandler()`, `getLogHandler()`, `warn()`, `filterWarnings()`, `catchWarnings()`, `resetWarnings()`, `setWarningHandler()`, `WorkerPool`, `createWorkerPool()`, `availableCores()`

### `deepbox/ndarray`

- Tensor types and autograd: `Tensor`, `GradTensor`, `AnyTensor`, `parameter()`, `noGrad()`, `customOp()`, `detach()`
- Creation: `tensor()`, `zeros()`, `ones()`, `empty()`, `full()`, `eye()`, `arange()`, `linspace()`, `logspace()`, `geomspace()`, `zerosLike()`, `onesLike()`, `emptyLike()`, `fullLike()`, `randn()`, `diag()`, `tril()`, `triu()`
- Element types: `Complex`, `Complex64Array`, `Complex128Array`, `Float16Array`, `BFloat16Array`, `roundToFloat16()`, `roundToBFloat16()`
- Arithmetic and math: `add`, `sub`, `mul`, `div`, `pow`, `mod`, `floorDiv`, `neg`, `abs`, `sign`, `reciprocal`, `maximum`, `minimum`, `clip`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `square`, `cbrt`, `rsqrt`, `floor`, `ceil`, `round`, `trunc`, `gcd`, `lcm`, trigonometric and hyperbolic functions, `addScalar`, `mulScalar`
- Reductions: `sum`, `mean`, `median`, `prod`, `min`, `max`, `std`, `variance`, `cumsum`, `cumprod`, `argmax`, `argmin`, `argsort`, `nonzero`, `argwhere`, `countNonzero`, `all`, `any`
- NaN-aware reductions: `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanstd`, `nanvar`, `nanmedian`, `nanprod`, `nanargmin`, `nanargmax`, `nancumsum`, `nanquantile`
- Comparison and logic: `equal`, `notEqual`, `greater`, `greaterEqual`, `less`, `lessEqual`, `isclose`, `allclose`, `arrayEqual`, `isnan`, `isinf`, `isfinite`, `logicalAnd`, `logicalOr`, `logicalNot`, `logicalXor`, `where`
- Shape: `reshape`, `flatten`, `transpose`, `squeeze`, `unsqueeze`, `expandDims`, `swapaxes`, `moveaxis`, `concatenate`, `stack`, `vstack`, `hstack`, `columnStack`, `split`, `tile`, `repeat`, `flip`, `flipLr`, `flipUd`, `roll`, `rot90`, `pad`, `broadcastTo`, `atleast1d`, `atleast2d`, `meshgrid`, `slice`, `contiguous`, `copy`, `clone`
- Indexing: `booleanIndex`, `fancyIndex`, `indexSelect`, `gather`, `scatter`, `takeAlongAxis`, `putAlongAxis`, `insert`, `delete_`
- Sorting and sets: `sort`, `argsort`, `searchsorted`, `digitize`, `bincount`, `histogram`, `unique` (with `returnIndex`, `returnInverse`, `returnCounts`), `union1d`, `intersect1d`, `setdiff1d`, `isin`
- Linalg-style helpers: `dot()` (batch-broadcasting), `einsum()`, `tensordot()`, `cross()` (batched), `corrcoef()`, `cov()`
- Activations: `relu`, `relu6`, `leakyRelu`, `elu`, `selu`, `celu`, `gelu` (with `{ approximate: "none" }`), `sigmoid`, `softmax`, `logSoftmax`, `softplus`, `softsign`, `mish`, `swish`, `hardsigmoid`, `hardswish`, `hardtanh`, `logSigmoid`, `hardshrink`, `softshrink`, `tanhshrink`
- FFT, signal, numerical: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `convolve`, `correlate`, `interp`, `trapz`, `gradient`, `diff`, window functions (`hannWindow`, `hammingWindow`, `blackmanWindow`, `bartlettWindow`, `kaiserWindow`)
- Differentiable building blocks used by `nn`: `im2col`, `col2im`, `logSoftmaxGrad`, `softmaxGrad`, `concatGrad`, `stackGrad`, `varianceGrad`, `dropoutGrad`
- Sparse: `CSRMatrix`

### `deepbox/linalg`

- Decompositions: `svd()`, `svdvals()`, `qr()`, `lu()`, `cholesky()`, `eig()`, `eigh()`, `eigvals()`, `eigvalsh()`, `schur()`, `polar()`, `hessenberg()`
- Inverses and properties: `inv()`, `pinv()`, `det()`, `trace()`, `matrixRank()`, `slogdet()`, `norm()`, `cond()`
- Matrix functions and constructors: `expm()`, `logm()`, `sqrtm()`, `matrixPower()`, `kron()`, `blockDiag()`, `companion()`, `circulant()`, `hadamard()`, `hankel()`, `hilbert()`, `toeplitz()`, `vandermonde()`
- Solvers: `solve()`, `lstsq()`, `solveTriangular()`, `solveBanded()`, `lyapunov()`, `sylvester()`, `sparseSolve()`, `sparseCholeskySolve()`, `denseToCSR()`

### `deepbox/dataframe`

- Core structures: `DataFrame`, `DataFrameGroupBy`, `Series`, `Rolling`, `Expanding`, `EWM`
- Accessors and types: `StringAccessor`, `DateTimeAccessor`, `PlotAccessor`, `StyleAccessor`, `MultiIndex`, `Categorical`
- Date/time helpers: `toDatetime()`, `dateRange()`, `timedelta()`
- Exported IO helpers: `readParquet()`, `writeParquet()`, `readXlsx()`, `writeXlsx()`
- New in 1.5.0 on `DataFrame`: `fillna` with a per-column object and `method`, `ffill()`, `bfill()`, `corr` with `method` and `minPeriods`, `sample` with `frac`, `replace`, `weights`, `groupBy` helpers `getGroup`, `nunique`, `quantile`, `transform` and named aggregation, `rolling` with `minPeriods` and `center`, `concat` with `join` and `ignoreIndex`, `valueCounts` with `normalize` and `dropna`
- Important note: CSV and JSON workflows are primarily implemented as `DataFrame` methods such as `fromCsvString()`, `toCsvString()`, `toCsv()`, `fromJsonString()`, `toJsonString()`, and `toJson()`

### `deepbox/stats`

- Confidence intervals: `meanConfidenceInterval()`, `meanConfidenceIntervalZ()`, `meanDiffConfidenceInterval()`, `proportionConfidenceInterval()`
- Correlation: `corrcoef()`, `cov()`, `pearsonr()`, `spearmanr()`, `kendalltau()`, `partialcorr()`, `pointbiserialr()`. The correlation tests take an `alternative` option, and `kendalltau` takes `variant` and `method`
- Descriptive statistics: `mean()`, `median()`, `mode()`, `variance()`, `std()`, `sem()`, `iqr()`, `zscore()`, `geometricMean()`, `harmonicMean()`, `trimMean()`, `percentile()`, `quantile()`, `bootstrap()`, `cohenD()`, `moment()`, `skewness()`, `kurtosis()`
- Distributions: `beta`, `binom`, `cauchy`, `chi2`, `expon`, `f`, `gamma`, `geom`, `hypergeom`, `laplace`, `lognorm`, `nbinom`, `norm`, `pareto`, `poisson`, `t`, `uniform`, `weibull`
- KDE and power analysis: `GaussianKDE`, `gaussianKde()`, `tTestPower()`
- Multiple testing: `bonferroni()`, `holm()`, `sidak()`, `hochberg()`, `benjaminiHochberg()`, `benjaminiYekutieli()`
- Tests: `ttest1samp()`, `ttestInd()`, `ttestRel()`, `fOneway()`, `fTwoway()`, `chisquare()`, `chi2Contingency()`, `fisherExact()`, `mannwhitneyu()`, `wilcoxon()`, `kruskal()`, `friedmanchisquare()`, `shapiro()`, `normaltest()`, `kstest()`, `ks2samp()`, `anderson()`, `levene()`, `bartlett()`, `fligner()`, `lilliefors()`, `runsTest()`, `medianTest()`

### `deepbox/metrics`

- Classification: `accuracy()`, `precision()`, `recall()`, `f1Score()`, `fbetaScore()`, `rocAucScore()` (multiclass with `multiClass: "ovr" | "ovo"`), `rocCurve()`, `precisionRecallCurve()`, `confusionMatrix()`, `classificationReport()`, `logLoss()` (accepts a probability matrix), `hammingLoss()`, `jaccardScore()`, `matthewsCorrcoef()` (multiclass), `cohenKappaScore()`, `balancedAccuracyScore()`, `averagePrecisionScore()`
- Regression: `mse()`, `meanSquaredError()`, `rmse()`, `rootMeanSquaredError()`, `mae()`, `meanAbsoluteError()`, `mape()` (percentage), `meanAbsolutePercentageError()` (fraction), `r2Score()`, `adjustedR2Score()`, `explainedVarianceScore()`, `maxError()`, `medianAbsoluteError()`. Common metrics take `sampleWeight`
- Clustering: `silhouetteScore()`, `silhouetteSamples()`, `daviesBouldinScore()`, `calinskiHarabaszScore()`, `adjustedRandScore()`, `randScore()`, `mutualInfoScore()`, `adjustedMutualInfoScore()`, `normalizedMutualInfoScore()`, `homogeneityScore()`, `completenessScore()`, `vMeasureScore()`, `fowlkesMallowsScore()`
- Extra and pairwise: `brierScoreLoss()`, `coverageError()`, `d2TweedieScore()`, `detCurve()`, `hingeLoss()`, `labelRankingLoss()`, `meanGammaDeviance()`, `meanPinballLoss()`, `meanPoissonDeviance()`, `meanSquaredLogError()`, `multilabelConfusionMatrix()`, `smape()`, `topKAccuracyScore()`, `zeroOneLoss()`, `ndcgScore()`, `pairwiseCosine()`, `pairwiseEuclidean()`, `pairwiseManhattan()`, `reciprocalRank()`

### `deepbox/preprocess`

- Discretization: `KBinsDiscretizer`
- Encoders: `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`, `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`
- Feature selection: `fClassif()`, `fRegression()`, `RFE`, `RFECV`, `SelectFromModel`, `SelectKBest`, `VarianceThreshold`
- Imputation: `SimpleImputer`, `KNNImputer`, `MissingIndicator`
- Mutual information: `mutualInfoClassif()`, `mutualInfoRegression()`
- Feature transformers: `Binarizer`, `FunctionTransformer`, `PolynomialFeatures`, `SplineTransformer`
- Scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`
- Splitting: `trainTestSplit()`, `KFold`, `StratifiedKFold`, `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneOut`, `LeavePOut`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `PredefinedSplit`, `RepeatedKFold`, `RepeatedStratifiedKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplit`, `TimeSeriesSplit`
- Text: `CountVectorizer`, `HashingVectorizer`, `TfidfVectorizer`

### `deepbox/ml`

- Base and output utilities: estimator types, `assertEstimator()`, `getEstimatorTags()`, `getOutput()`, `setOutput()`, `resetOutput()`
- Anomaly detection: `IsolationForest`, `LocalOutlierFactor`
- Calibration: `CalibratedClassifierCV`, `calibrationCurve()`
- Clustering: `AffinityPropagation`, `AgglomerativeClustering`, `Birch`, `DBSCAN`, `GaussianMixture`, `KMeans`, `MeanShift`, `MiniBatchKMeans`, `OPTICS`, `SpectralClustering`
- Decomposition: `PCA`, `FastICA`, `NMF`, `TruncatedSVD`, `LatentDirichletAllocation`
- Discriminant analysis: `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`
- Ensemble: `AdaBoostClassifier`, `AdaBoostRegressor`, `BaggingClassifier`, `BaggingRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `StackingClassifier`, `StackingRegressor`, `VotingClassifier`, `VotingRegressor`
- Gaussian processes: `GaussianProcessClassifier`, `GaussianProcessRegressor`
- Inspection: `permutationImportance()`
- Linear models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`, `BayesianRidge`, `HuberRegressor`, `IsotonicRegression`, `KernelRidge`, `QuantileRegressor`, `RANSACRegressor`
- Manifold learning: `TSNE`, `Isomap`, `MDS`, `SpectralEmbedding`
- MLP: `MLPClassifier`, `MLPRegressor`
- Model selection: `GridSearchCV`, `RandomizedSearchCV`, `crossValScore()`, `crossValidate()`
- Multiclass: `OneVsOneClassifier`, `OneVsRestClassifier`
- Naive Bayes: `GaussianNB`, `BernoulliNB`, `CategoricalNB`, `ComplementNB`, `MultinomialNB`
- Neighbors: `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestNeighbors`, `NearestCentroid`, `KDTree`, `BallTree`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`
- Pipeline and feature composition: `Pipeline`, `FeatureUnion`, `ColumnTransformer`, `makePipeline()`
- Random projection: `GaussianRandomProjection`, `johnsonLindenstraussMinDim()`
- Semi-supervised: `LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`
- SVM: `LinearSVC`, `LinearSVR`, `SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM`
- Trees: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `exportText()`

### `deepbox/nn`

- Gradient clipping: `clipGradNorm_()`, `clipGradValue_()`
- Containers: `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`
- Initialization: `constant_()`, `zeros_()`, `ones_()`, `eye_()`, `uniform_()`, `normal_()`, `truncNormal_()`, `xavierUniform_()`, `xavierNormal_()`, `kaimingUniform_()`, `kaimingNormal_()`, `orthogonal_()`, `sparse_()`, `calculateGain()`, `calculateFanInOut()`
- Linear and embedding: `Linear`, `Identity`, `Flatten`, `Unflatten`, `Embedding`, `EmbeddingBag`
- Activation layers: `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `GELU`, `GLU`, `Sigmoid`, `LogSigmoid`, `Tanh`, `Tanhshrink`, `Softmax`, `LogSoftmax`, `Softmin`, `Softmax2d`, `Softplus`, `Softsign`, `SiLU`, `Swish`, `Mish`, `Hardsigmoid`, `Hardswish`, `Hardtanh`, `Hardshrink`, `Softshrink`, `Threshold`
- Attention and transformer: `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`, `FullTransformer`, `PositionalEncoding`, `causalMask()`
- Convolution and pooling: `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`
- Recurrent: `RNN`, `LSTM`, `GRU`, plus `packSequence()`, `packPaddedSequence()`, `padPackedSequence()`, `unpackSequence()`
- Normalization: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LocalResponseNorm`, `SpectralNorm`
- Regularization, padding, upsampling: `Dropout`, `Dropout2d`, `AlphaDropout`, `ZeroPad2d`, `ConstantPad2d`, `ReflectionPad2d`, `ReplicationPad2d`, `Upsample`
- Losses: `binaryCrossEntropyLoss()`, `binaryCrossEntropyWithLogitsLoss()`, `cosineEmbeddingLoss()`, `crossEntropyLoss()`, `ctcLoss()`, `gaussianNLLLoss()`, `huberLoss()`, `klDivLoss()`, `maeLoss()`, `marginRankingLoss()`, `mseLoss()`, `nllLoss()`, `poissonNLLLoss()`, `rmseLoss()`, `smoothL1Loss()`, `tripletMarginLoss()`
- Base module and training: `Module`, `Trainer`, `EarlyStopping`, `GradientAccumulator`, `ModelCheckpoint`

### `deepbox/optim`

- Base classes: `Optimizer`, `LRScheduler`
- Optimizers: `SGD`, `Adam`, `AdamW`, `Nadam`, `Adamax`, `RAdam`, `RMSprop`, `Adagrad`, `AdaDelta`, `LBFGS`, `LARS`, `LAMB`, `Lion`, `SparseAdam`, `ASGD`, `Rprop`
- Schedulers: `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `CyclicLR`, `LambdaLR`, `LinearLR`, `OneCycleLR`, `PolynomialLR`, `ReduceLROnPlateau`, `SequentialLR`, `WarmupLR`

### `deepbox/random`

- Seed management: `setSeed()`, `getSeed()`, `clearSeed()`
- `Generator`: a seeded stream that is independent of the global seed. `new Generator(seed)` with `random()`, `normal()`, `uniform()`, `randint()`, `exponential()`, `bernoulli()`, `choice(weights)`, `permutation(n)`, and array forms such as `normalArray()` and `uniformArray()` that return typed arrays
- Basic sampling (return tensors, take a shape): `rand()`, `randn()`, `randint()`, `uniform()`, `normal()`, `choice()`, `permutation()`, `shuffle()`
- Distributions: `bernoulli`, `beta`, `binomial`, `categorical`, `cauchy`, `chi2`, `dirichlet`, `exponential`, `fDistribution`, `gamma`, `geometric`, `gumbelSoftmax`, `hypergeometric`, `laplace`, `lognormal`, `multinomial`, `multivariateNormal`, `negativeBinomial`, `pareto`, `poisson`, `rayleigh`, `studentT`, `triangular`, `vonmises`, `weibull`, `zipf`
- Source barrel inspection is still the way to get option shapes, because many functions are defined directly in `src/random/index.ts`

### `deepbox/datasets`

- Data loading core: `DataLoader`, `StreamingDataset`, `iterableDataset()`, `asyncIterableDataset()`, `defaultCollate()`
- Synthetic generators: `makeBiclusters()`, `makeBlobs()`, `makeCheckerboard()`, `makeCircles()`, `makeClassification()`, `makeFriedman1()`, `makeFriedman2()`, `makeFriedman3()`, `makeGaussianQuantiles()`, `makeLowRankMatrix()`, `makeMoons()`, `makeRegression()`, `makeSCurve()`, `makeSPDMatrix()`, `makeSparseUncorrelated()`, `makeSwissRoll()`
- Image helpers: `fetchMNIST()`, `fetchCIFAR10()`
- Kaggle helpers: `readKaggleCredentials()`, `searchKaggleDatasets()`, `fetchKaggleDatasetInfo()`, `listKaggleFiles()`, `fetchKaggleDataset()`
- Built-in loaders: `loadIris()`, `loadWine()`, `loadDigits()`, `loadBreastCancer()`, `loadDiabetes()`, `loadLinnerud()`, plus domain-specific datasets such as `loadHousingMini()`, `loadStudentPerformance()` and `loadWeatherOutcomes()`. Each returns `{ data, target, featureNames, targetNames, description }`
- Remote CSV: `fetchCSVDataset()`, `parseCSV()`
- Samplers and transforms: `SequentialSampler`, `SubsetRandomSampler`, `WeightedRandomSampler`, `filterDataset()`, `mapDataset()`, `randomSplit()`, `Subset`
- Text helpers: `fetch20Newsgroups()`, `fetchIMDB()`. Both download the official archives, so they need network access

### `deepbox/plot`

- Core figure API: `Figure`, `Axes`, `figure()`, `gcf()`, `gca()`, `sca()`, `subplot()`, `show()`, `saveFig()`
- Animation, interactivity, themes and palettes: `Animation`, `createAnimation()`, `InteractivePlot`, `createInteractivePlot()`, `setTheme()`, `getTheme()`, `listThemes()`, `resetTheme()`, `getPalette()`, `listPalettes()`
- Base plotting: `plot()`, `scatter()`, `bar()`, `barh()`, `hist()`, `boxplot()`, `violinplot()`, `pie()`, `heatmap()`, `imshow()`, `contour()`, `contourf()`, `errorbar()`, `area()`, `fillBetween()`, `step()`, `legend()`, `title()`, `xlabel()`, `ylabel()`, `xlim()`, `ylim()`, `grid()`, `text()`, `annotate()`, `twinx()`
- Plot variants and helpers: `axhline()`, `axvline()`, `stackedBar()`, `groupedBar()`
- 3-D: `scatter3d()`, `surface()`, `wireframe()`, `Scatter3D`, `Surface3D`, `Wireframe3D`
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

Every recipe below runs as written against 1.5.0 (ESM, Node 24).

### Tensor and Autograd

```ts
import { parameter } from "deepbox/ndarray";

const x = parameter([1, 2, 3]);
const y = x.mul(x).sum();
y.backward();

console.log(x.grad?.toString());
```

### Fluent Tensor Methods and Dtypes

```ts
import { tensor } from "deepbox/ndarray";

const a = tensor([[1, 2], [3, 4]]);
console.log(a.add(1).mul(2).sum().item()); // 28
console.log(a.T.toString());
console.log(a.argmax(1).dtype); // int32
console.log(a.softmax(-1).toString());

const ints = tensor([1, 2, 3], { dtype: "int32" });
console.log(ints.add(tensor([0.5, 0.5, 0.5])).dtype); // float32
console.log(ints.mean().dtype); // float32
```

### Classical ML

```ts
import { loadIris } from "deepbox/datasets";
import { accuracy, f1Score } from "deepbox/metrics";
import { LogisticRegression } from "deepbox/ml";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

const { data, target } = loadIris();
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(data, target, {
  testSize: 0.25,
  randomState: 42,
});

const scaler = new StandardScaler();
const XTrainScaled = scaler.fitTransform(XTrain);
const XTestScaled = scaler.transform(XTest);

const model = new LogisticRegression();
model.fit(XTrainScaled, yTrain);
const pred = model.predict(XTestScaled);

console.log(accuracy(yTest, pred), f1Score(yTest, pred, "macro"));
```

Fit scalers on the training split only, then call `transform` on the test split.

### Pipeline and Cross-Validation

`crossValScore(estimator, X, y, cv, scoring?)` takes the number of folds as `cv` (an integer of at least 2). Classifiers get stratified folds.

```ts
import { loadIris } from "deepbox/datasets";
import { LogisticRegression, Pipeline, crossValScore } from "deepbox/ml";
import { StandardScaler } from "deepbox/preprocess";

const { data, target } = loadIris();

const pipe = new Pipeline([
  ["scale", new StandardScaler()],
  ["clf", new LogisticRegression()],
]);
console.log(crossValScore(pipe, data, target, 5));
```

### Neural Network Training

Train on plain tensors. `model.forward(x)` returns a `GradTensor` when the model has trainable parameters, so `loss.backward()` works. Evaluate inside `noGrad()`.

```ts
import { noGrad, tensor } from "deepbox/ndarray";
import { Linear, ReLU, Sequential, mseLoss } from "deepbox/nn";
import { Adam } from "deepbox/optim";
import { setSeed } from "deepbox/random";

setSeed(0);

const X = tensor([[1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1]]);
const y = tensor([[1], [2], [3], [4], [5], [5]]);

const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));
const optimizer = new Adam(model.parameters(), { lr: 0.01 });

for (let epoch = 0; epoch < 200; epoch++) {
  optimizer.zeroGrad();
  const loss = mseLoss(model.forward(X), y);
  loss.backward();
  optimizer.step();
  if (epoch % 50 === 0) console.log(epoch, loss.item());
}

model.eval();
const pred = noGrad(() => model.forward(X)); // plain Tensor, no graph
console.log(pred.toString());
```

### Trainer

```ts
import { type Tensor, tensor } from "deepbox/ndarray";
import { Linear, ReLU, Sequential, Trainer, mseLoss } from "deepbox/nn";
import { Adam } from "deepbox/optim";

const batches: [Tensor, Tensor][] = [
  [tensor([[1, 0], [0, 1]]), tensor([[1], [2]])],
  [tensor([[1, 1], [2, 1]]), tensor([[3], [4]])],
];

const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const optimizer = new Adam(model.parameters(), { lr: 0.01 });
const trainer = new Trainer(model, optimizer, (out, target) => mseLoss(out, target), {
  epochs: 20,
  restoreBestWeights: true,
});

const result = trainer.fit(batches);
console.log(result.history.length, result.bestEpoch);
```

### DataFrames

```ts
import { DataFrame } from "deepbox/dataframe";

const df = new DataFrame({
  team: ["b", "a", "b", "a"],
  score: [10, 20, 30, 40],
});

console.log(df.groupBy("team", { sort: true }).sum().toString());
console.log(df.groupBy("team").agg({ best: ["score", "max"] }).toString());
```

### Plotting

```ts
import { tensor } from "deepbox/ndarray";
import { figure, plot, saveFig } from "deepbox/plot";

figure({ width: 800, height: 500 });
plot(tensor([1, 2, 3]), tensor([2, 4, 8]), { label: "growth" });
await saveFig("growth.svg");
```

### Seeded Randomness

```ts
import { tensor } from "deepbox/ndarray";
import { Generator, randn, setSeed } from "deepbox/random";

setSeed(42); // global stream, used by randn, rand, shuffle and the model initializers
console.log(randn([2, 2]).toString());

const rng = new Generator(7); // independent stream, returns numbers and typed arrays
const noise = tensor(Array.from(rng.normalArray(0, 1, 4)));
console.log(noise.toString());
```

### Opting In to the WASM Backend

```ts
import { WasmBackend, isBackendAvailable, registerBackend } from "deepbox/core";
import { zeros } from "deepbox/ndarray";

const wasm = new WasmBackend();
await wasm.init(); // completes without error even when WASM SIMD is missing
if (wasm.info().available) registerBackend("wasm", wasm);

if (isBackendAvailable("wasm")) {
  const a = zeros([1024], { device: "wasm" }).add(1);
  console.log(a.add(a).sum().item()); // 2048, computed on host memory
}
```

## Devices and Backends

Every tensor is on the CPU unless you create or move it to another device after registering a backend. Only the `cpu` backend is registered at load time. `listBackends()` shows what is registered and `isBackendAvailable(device)` shows what is usable.

- Creating a tensor on a device with no registered, available backend throws `DeviceError`. The same applies to `await t.to(device)` and `await model.to(device)`.
- Register a backend after initializing it: `await backend.init()`, then `registerBackend("wasm" | "webgpu", backend)`. `init()` does not throw when the runtime lacks support. Check `backend.info().available` before registering.

`wasm` (host accelerator):

- `wasm` tensors keep ordinary host storage. Moving between `cpu` and `wasm` (`await t.to("wasm")`) only relabels the tensor and shares memory.
- Only `add`, `sub`, `mul` and `div` on same-shape, contiguous `float32` tensors with at least 512 elements use the SIMD kernels. Every other op on a `wasm` tensor runs the normal CPU code and returns the same values. It does not throw.
- Treat `wasm` as an optional speed-up for large float32 arithmetic, not as a different execution environment. Other dtypes (such as `float64`) are accepted and computed on the CPU.

`webgpu` (kernel device):

- `webgpu` tensors live in GPU memory. Only `float32`, `float16` (needs the WebGPU `shader-f16` feature) and `bfloat16` tensors can be moved there. Convert with `astype("float32")` first, otherwise `to()` throws `DTypeError`.
- Ops that are wired to the GPU: element-wise `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `neg`, `abs`, `sign`, `reciprocal`, `exp`, `log`, `sqrt`, `square`, `rsqrt`, `expm1`, `log1p`, `tanh`, `relu`, `sigmoid`, `gelu`, `softplus`, `where`, the reductions `sum`, `mean`, `min`, `max` (full or along axes), `dot` and `matmul` (including batched), and the conv and pooling kernels used by `Conv2d` and the 2-D pooling layers.
- Any other op on a `webgpu` tensor (for example `sort`, `cumsum`, comparisons, `astype`, `fill`, reading `.data`, `toArray()`, `item()`) throws `DeviceError` that tells you to move the tensor with `await t.cpu()`.
- Operands on different devices throw `DeviceError`, except a 0-D host tensor, which is uploaded automatically. Move tensors explicitly with `await t.to(device)`.
- Reductions of empty tensors and negative-step slices throw `DeviceError` on the device.
- Free large device tensors with `t.dispose()`. A disposed tensor throws `DeviceError` when used.

Modules:

- `Module.to(device)` returns a `Promise`. Always `await` it. It validates the device immediately and throws `DeviceError` if no backend is available.
- Kernel devices take `float32`, `float16` and `bfloat16` parameters only. Other parameters and buffers stay on the host.
- `Embedding`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, the recurrent layers, `PReLU` and `SpectralNorm` keep their weights on the host, so those layers still work after `model.to("webgpu")`.

Do not describe a WASM or WebGPU speed-up without a registered, initialized backend, and do not assume an op is accelerated without checking `src/ndarray/ops/device_dispatch.ts` and the `devices-and-execution` docs page.

## Agent Guardrails

- Prefer subpath imports unless you intentionally want namespace imports.
- Use camelCase names and fluent `Tensor` methods in new code.
- Do not wrap training data in `parameter(...)`. Pass plain tensors to `model.forward()`.
- Run inference inside `noGrad(() => ...)`. `model.eval()` alone does not stop graph recording.
- Do not pass an async function to `noGrad()`. It throws. Keep only the synchronous tensor work inside.
- Do not expect `complex64` or `complex128` tensors. Creating one throws `DTypeError`.
- Use `setSeed()` or `Generator` in reproducible examples that depend on randomness.
- Mention remote/network behavior when using dataset fetch helpers (`fetchMNIST`, `fetchCIFAR10`, `fetch20Newsgroups`, `fetchIMDB`, `fetchCSVDataset`, Kaggle helpers).
- Do not claim an API exists at the root package unless it is actually exported there.
- When documenting `dataframe` IO, distinguish between `DataFrame` methods and module-barrel helpers.
- When building wrappers or utilities around Deepbox, prefer Deepbox custom errors over generic `Error`.
- Prefer website docs and examples over maintainer-oriented repo files.
- Register and initialize a backend before describing or using a `wasm` or `webgpu` device (see "Devices and Backends").

## Upgrading From 1.0.0

Point agents at this section when they migrate older code.

- Nothing was removed or renamed. snake_case names still work and are marked deprecated.
- Float ops now keep the input float dtype. Code that read float64 back from `float32` input now gets `float32`. Integer input to `mean`, `div`, `exp`, `log` and similar gives `float32`.
- Mixed dtypes promote instead of throwing `DTypeError`.
- Parameter-free layers return a plain `Tensor` for plain input. Remove `.tensor` from their results.
- Layers compute in their parameter dtype and cast the input, so recurrent layers accept integer input.
- Optimizers skip parameters without a gradient. Frozen parameters no longer throw.
- `parameter(...)` created inside `noGrad()` keeps `requiresGrad: true`.
- Seeded random streams changed in several modules. Re-check any test that hard-codes sampled values.
- Many functions return correct values where 1.0.0 was wrong. Examples: `OneClassSVM` outlier rates, plot tick labels, `digitize` with decreasing bins, `mannwhitneyu` and `wilcoxon` p-values for small samples. Re-run numerical tests before trusting old expected values.
- `readParquet` and `readXlsx` throw `DataValidationError` on an invalid buffer instead of returning an empty result, and `fetchCIFAR10` throws when a training batch is missing instead of loading fewer batches.

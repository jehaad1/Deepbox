<img src="./Banner.png" alt="Deepbox" />

# Deepbox

## The TypeScript Toolkit for AI & Numerical Computing

[![CI](https://github.com/jehaad1/Deepbox/actions/workflows/ci.yml/badge.svg)](https://github.com/jehaad1/Deepbox/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/deepbox)](https://www.npmjs.com/package/deepbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!--
[![Bench](https://img.shields.io/badge/Bench-Leaderboard-amber)](https://bench.deepbox.dev)
-->

Deepbox is a TypeScript library for tensors, linear algebra, tabular data, machine learning, neural networks, statistics, datasets and plotting. It has no runtime dependencies and ships ESM and CommonJS builds with type declarations. The API follows NumPy, pandas, scikit-learn and PyTorch, so code written against those libraries translates directly.

> Docs: [deepbox.dev/docs](https://deepbox.dev/docs)
> Examples: [deepbox.dev/examples](https://deepbox.dev/examples)
> Projects: [deepbox.dev/projects](https://deepbox.dev/projects)
<!--
> Benchmarks: [bench.deepbox.dev](https://bench.deepbox.dev)
-->

## Installation

```bash
npm install deepbox
```

## Requirements

- Node.js `>= 24.13.0`, as declared in the `engines` field of `package.json`. Deepbox 1.x is built and tested on Node 24 with a TypeScript `ES2024` target. Older Node versions are not supported.

## Import Model

Each module has its own subpath export. Import named APIs from the module you use:

```ts
import { tensor, parameter } from "deepbox/ndarray";
import { LinearRegression } from "deepbox/ml";
import { DataFrame } from "deepbox/dataframe";
```

The root package exports namespaces, not named symbols:

```ts
import * as db from "deepbox";

const x = db.ndarray.tensor([1, 2, 3]);
const model = new db.ml.LinearRegression();
```

## Quick Start

The samples below run as written. They import from the subpaths above.

### Tensors and Autograd

`Tensor` and `GradTensor` share one method surface, so operations chain the way they do in PyTorch. The functional form (`add(a, b)`) still works. `tensor()` creates `float32` data by default.

```ts
import { noGrad, parameter, tensor } from "deepbox/ndarray";

const t = tensor([
  [1, 2],
  [3, 4],
]);
console.log(t.add(1).mul(2).sum().item()); // 28
console.log(t.T.toString());
console.log(t.argmax(1).toString()); // int32 indices
console.log(t.mean().item()); // 2.5

// parameter() creates a tensor that records operations for backward().
const x = parameter([
  [1, 2],
  [3, 4],
]);
const w = parameter([[0.5], [0.25]]);
const y = x.matmul(w).sum();
y.backward();
console.log(x.grad?.toString());
console.log(w.grad?.toString());

// noGrad() turns tracking off for everything computed inside it.
const pred = noGrad(() => x.matmul(w));
console.log(pred.requiresGrad); // false
```

### Neural Network

Training data stays in plain tensors. When gradient tracking is on and a module has trainable parameters, `model.forward(x)` returns a `GradTensor` that tracks the weights, so `loss.backward()` fills the gradients that the optimizer reads. `loss.item()` returns the loss as a number.

```ts
import { tensor } from "deepbox/ndarray";
import { Linear, mseLoss, ReLU, Sequential } from "deepbox/nn";
import { Adam } from "deepbox/optim";
import { setSeed } from "deepbox/random";

setSeed(42);

// Learn y = x1 + 2 * x2 from four points.
const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const y = tensor([[0], [2], [1], [3]]);

const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));
const optimizer = new Adam(model.parameters(), { lr: 0.01 });

for (let epoch = 0; epoch < 200; epoch++) {
  optimizer.zeroGrad();
  const loss = mseLoss(model.forward(X), y);
  loss.backward();
  optimizer.step();
  if (epoch % 50 === 0 || epoch === 199) console.log(epoch, loss.item());
}
```

### Classical ML

Estimators follow the scikit-learn pattern: `fit`, `predict`, `score`.

```ts
import { loadIris } from "deepbox/datasets";
import { accuracy } from "deepbox/metrics";
import { RandomForestClassifier } from "deepbox/ml";
import { trainTestSplit } from "deepbox/preprocess";

const { data, target } = loadIris();
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(data, target, {
  testSize: 0.25,
  randomState: 42,
});

const model = new RandomForestClassifier({ nEstimators: 50, randomState: 42 });
model.fit(XTrain, yTrain);

console.log(accuracy(yTest, model.predict(XTest)));
```

### DataFrames

```ts
import { DataFrame } from "deepbox/dataframe";

const df = new DataFrame({
  name: ["Alice", "Bob", "Charlie"],
  team: ["A", "A", "B"],
  score: [91, 84, 96],
});

console.log(df.groupBy("team").mean().toString());
console.log(df.filter((row) => Number(row.score) > 90).toString());
```

## GPU and WASM Backends

Tensors carry a device: `cpu`, `webgpu` or `wasm`. The CPU backend is always registered. The other two are opt-in: create the backend, call `await backend.init()`, check `info().available`, then pass it to `registerBackend`.

Operands of one op must be on the same device, except that a 0-D host tensor is moved to the device of the other operand. An op that a device does not implement throws a `DeviceError` that tells you to move the tensor with `await t.cpu()`. Deepbox does not copy `webgpu` data to the CPU on its own.

### WebGPU

The WebGPU backend runs these ops as WGSL compute kernels, on views, transposes and broadcasts without copying:

- Element-wise binary ops (`add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`) and the common unary ops and activations (`exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `gelu`, `erf` and others), plus `where`.
- `matmul` and `dot`, including batched matmul with batch broadcasting.
- `sum`, `mean`, `max` and `min`, over the whole tensor or along axes.
- 2-D convolution (through `im2col` and `col2im`) and 2-D max and average pooling.

Supported element types are `float32`, `float16` and `bfloat16`. `float16` needs the WebGPU `shader-f16` feature. `bfloat16` is rounded to bfloat16 on upload and download and computed in float32. Reductions of empty tensors throw on device.

Autograd records through these ops, and the optimizers keep their state on the device when the parameters live there. `LBFGS` and `SparseAdam` are the exceptions: they throw a `DeviceError` for device parameters. The test suite checks the device paths against an in-process reference backend, so CI does not need a GPU. Anything outside the list above needs `await t.cpu()` first.

```ts
import { registerBackend, WebGpuBackend } from "deepbox/core";
import { dot, relu, tensor } from "deepbox/ndarray";

// In a browser, navigator.gpu is used. In Node, pass a WebGPU binding: new WebGpuBackend({ gpu }).
const gpu = new WebGpuBackend();
await gpu.init();
if (gpu.info().available) {
  registerBackend("webgpu", gpu);

  const a = tensor(
    [
      [1, 2],
      [3, 4],
    ],
    { device: "webgpu" }
  );
  const y = relu(dot(a, a)); // runs as WGSL compute kernels
  console.log((await y.cpu()).toString()); // read back to host memory
} else {
  console.log("WebGPU is not available in this runtime");
}
```

### WASM

The WASM backend is a host accelerator. Tensors on the `wasm` device keep ordinary host memory. Element-wise `add`, `sub`, `mul` and `div` run through embedded SIMD kernels when both operands are contiguous `float32` tensors of the same shape with at least 512 elements. Every other op runs the normal CPU code, and results are identical to the CPU. If the runtime lacks WASM SIMD, `info().available` is `false`.

```ts
import { registerBackend, WasmBackend } from "deepbox/core";
import { add, tensor } from "deepbox/ndarray";

const wasm = new WasmBackend();
await wasm.init();
if (wasm.info().available) {
  registerBackend("wasm", wasm);

  const a = tensor(new Array(4096).fill(1), { dtype: "float32", device: "wasm" });
  const b = add(a, a); // SIMD kernel
  console.log(b.device, b.at(0)); // wasm 2
} else {
  console.log("WASM SIMD is not available in this runtime");
}
```

## Modules

| Module | Includes |
| --- | --- |
| `deepbox/core` | Types, errors, config, backends, logging, warnings, validation, serialization, worker pool |
| `deepbox/ndarray` | Tensor creation, 100+ operations, autograd, sparse CSR, FFT, einsum, numerical utilities |
| `deepbox/linalg` | Decompositions, matrix functions, solvers, norms, special matrices |
| `deepbox/dataframe` | `DataFrame`, `Series`, string and datetime accessors, MultiIndex, Categorical, CSV/JSON methods, Excel/Parquet helpers |
| `deepbox/stats` | Descriptive stats, correlations, distributions, hypothesis tests, KDE, confidence intervals, power analysis |
| `deepbox/metrics` | Classification, regression, clustering, pairwise, ranking and calibration metrics |
| `deepbox/preprocess` | Scalers, encoders, imputers, feature selection, text vectorizers, splitters |
| `deepbox/ml` | Linear models, trees, ensembles, SVM, neighbors, Naive Bayes, clustering, manifold, pipelines, model selection |
| `deepbox/nn` | Modules, layers, recurrent models, transformers, losses, training utilities, initialization |
| `deepbox/optim` | Optimizers and learning-rate schedulers |
| `deepbox/random` | Seed control, `Generator`, distributions, sampling utilities |
| `deepbox/datasets` | Built-in datasets, synthetic generators, loaders, samplers, remote and Kaggle helpers |
| `deepbox/plot` | Figure API, SVG/PNG/PDF output, statistical plots, ML diagnostic plots, palettes, animation |

## Upgrading from 1.0

Deepbox 1.5 is backward compatible with 1.0: no export was removed or renamed, and older names still work. Results changed in the places listed below. The full list is in [CHANGELOG.md](CHANGELOG.md).

- Dtype rules: Float ops keep the input float dtype, so `float32` stays `float32` (several ops returned `float64` in 1.0). Integer input to an op with a fractional result (`mean`, `exp`, `softmax` and similar) gives `float32`. Index results (`argmax`, `argsort`, `digitize`, `searchsorted`, `nonzero`) are `int32`. Call `t.astype("float64")` if you need double precision.
- Promotion: Mixed dtypes promote like PyTorch instead of throwing: `int32` with `float32` gives `float32`, `float32` with `float64` gives `float64`. A JavaScript number never changes a tensor's dtype.
- Training API: Data no longer needs `parameter(...)`. `model.forward(x)` returns a `GradTensor` that tracks the weights, and `noGrad()` turns tracking off. Optimizers skip parameters that have no gradient. In 1.0 they threw `NotFittedError`.
- Names: All public names are camelCase (`matrixPower`, `toDatetime`, `ttestInd`, `crossValScore`, `multivariateNormal`, `checkXY`). The snake_case names from 1.0 still work and are marked deprecated.
- Parameter-free layers: Layers without trainable parameters (pooling, `Dropout2d`, `LayerNorm` without affine parameters) return a plain `Tensor` for plain input. Read the result directly instead of through `.tensor`.
- Seeded streams: The same seed gives different numbers than 1.0 in several modules. Tests that pin exact random values need new expected values.
- Corrected results: About 1,500 fixes landed, many of them wrong results in 1.0. Calls that used to throw (mixed dtypes, batched broadcasting, frozen parameters in optimizers) now work.

## Defaults that Differ from NumPy, pandas, scikit-learn and PyTorch

These defaults are kept in 1.x for compatibility and may change in 2.0.

| API | Deepbox default | Reference |
| --- | --- | --- |
| `PowerTransformer` `standardize` | `false` | scikit-learn: `true` |
| `DataFrame.ewm` `adjust` / `bias` | `false` / `true` | pandas: `true` / `false` |
| `gelu`, `GELU` | tanh approximation | PyTorch: exact (`approximate: "none"`) |
| `mape` | percentage | scikit-learn returns a fraction; use `meanAbsolutePercentageError` for that |
| `InstanceNorm` `affine` | `true` | PyTorch: `false` |
| `tensor()` default dtype | `float32` | NumPy: `float64` (PyTorch: `float32`) |

## Known Limits

- `complex64` and `complex128` appear in the `DType` type, but tensors cannot be created with them yet.
- WebGPU and WASM accelerate a subset of ops, as described above. The rest throw a `DeviceError`.

<!--
## Performance

Deepbox is pure TypeScript with no native addons and no C bindings. Operations run on V8's JIT compiler with `TypedArray` backing. The Python libraries it is compared with use hand-tuned C and Fortran (BLAS, LAPACK, ATen).

930 head-to-head benchmarks across 12 categories, run on the same machine with identical data sizes and median-based winner selection. Deepbox-only local cases are tracked separately and excluded from the win totals.

Two win rates are reported. Overall: 586/930 (63.0%). Realized-work, which excludes 120 sub-microsecond lazy-view and spec-build cases that only rewrite shape and stride metadata: 483/810 (59.6%). See benchmarks/RESULTS.md for the per-case breakdown.

| Category | Deepbox Wins | Python Package Wins | Competing Against |
| --- | ---: | ---: | --- |
| DataFrames | 43 | 28 | Pandas (C / Cython) |
| Datasets | 51 | 0 | scikit-learn |
| Linear Algebra | 15 | 50 | NumPy + SciPy (LAPACK) |
| Metrics | 125 | 4 | scikit-learn (C / Cython) |
| ML Training | 60 | 28 | scikit-learn (C / Cython) |
| NDArray Ops | 41 | 169 | NumPy (C / BLAS) |
| Neural Networks | 28 | 19 | PyTorch (C++ ATen) |
| Optimizers | 46 | 3 | PyTorch (C++ ATen) |
| Plotting | 6 | 0 | Matplotlib (C / Agg) |
| Preprocessing | 48 | 16 | scikit-learn (C / Cython) |
| Random | 47 | 6 | NumPy (C) |
| Statistics | 76 | 21 | SciPy (C / Fortran) |
| Total | 586 | 344 | n/a |

The gap is largest for BLAS-bound operations (matmul, decompositions) and smallest for memory-layout operations (transpose, reshape, indexing), where the lazy-view design helps.

Run `npm run bench:all` to reproduce. Full results are in benchmarks/RESULTS.md.
-->

## Repository Contents

- [`src`](./src) holds the library, one folder per module.
- [`test`](./test) holds the Vitest suite. Regression tests for the 1.5.0 audit are in [`test/v150`](./test/v150).
- [`docs/examples`](./docs/examples) holds the numbered examples, each runnable on its own.
- [`docs/projects`](./docs/projects) holds larger end-to-end projects.
- [`CHANGELOG.md`](./CHANGELOG.md) lists the changes in each release.
- [`SKILL.md`](./SKILL.md) is the guide for AI agents.
<!--
- [`benchmarks`](./benchmarks) holds the Deepbox and Python benchmark harnesses.
-->

## For AI Agents

[`SKILL.md`](./SKILL.md) is the repository guide for coding agents. It is also installed at `node_modules/deepbox/SKILL.md`. It covers:

- import patterns
- module selection
- core types
- the error hierarchy
- common coding patterns and pitfalls

## Development

```bash
npm ci
npm run validate:all
```

`validate:all` runs the format, lint and type checks, builds the package, and runs the tests, benchmarks smoke run, examples, projects and coverage. See [CONTRIBUTING.md](CONTRIBUTING.md) for the individual scripts and the writing rules. Security reporting instructions are in [SECURITY.md](SECURITY.md).

## License

Deepbox is released under the [MIT License](LICENSE).

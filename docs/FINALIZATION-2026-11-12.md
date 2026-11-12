# Finalization ledger, 2026-11-12

Single source of truth for the 1.5.x finalization pass over Deepbox, DeepboxDocs and the two benchmark repos. Updated after every accepted piece of work.

Priorities: P0 breaks users or loses data, P1 users will notice, P2 quality, P3 polish.

## Phase 0: ground truth

| Item | State |
| --- | --- |
| Library branch | `main` at `7a82acb`, equal to `origin/main` and `origin/release/1.5.0`. Local `main` was 7 commits behind; fast-forwarded. |
| npm | `deepbox@1.5.0` is `latest`. `v1.5.0` tag exists. `src/` changes since the tag are JSDoc and three small `nn` edits. |
| Git identity | `Jehaad Aljohani`, GitHub `jehaad1`. Matches the owner. |
| CI | Last 3 runs on `main` green (CI and Release). |
| Baseline `validate:all` (clean clone, `npm ci`) | Every gate green except one test: `test/v150/c22-ml-manifold.test.ts` "handles a few hundred samples quickly" asserts wall-clock `< 5000 ms` and took 6178 ms under coverage on a loaded machine. Same test passed in the plain `npm test` run. 13399 tests, 506 files. Total time 6m13s. |
| Install | `npm ci` warns that npm 11 blocks the `esbuild` postinstall script. Build and tests still work. |
| DeepboxDocs | `main` at `8ccb1f1`, clean, deployed by Vercel's Git integration (`deepbox-docs` project, aliases `deepbox.dev`, `www.deepbox.dev`). |
| deepbox-bench-private | 12 modified files and new `bench/engine/`: the all-or-nothing scoring engine (package version 2.0.0). Under review. |
| deepbox-bench-web | `LabLogo.tsx` reformat that corrupts SVG path data (`Math.LN10`, `Math.PI` pasted into numbers) and a stray `bun.lock` (repo uses npm). |

## Decisions

Each decision: what, why, how to undo.

- D3. bench-private: kept and committed the all-or-nothing scoring engine (local commit, not pushed). Evidence: `validate:all` there passes format, lint, typecheck, `next build`, `bench:check`, audit, strict discrimination, coverage, API coverage, policy, scorecard, calibration, rubrics and self-check (200 of 200 references score full marks). Only `docs:check` fails, because the generated prompt pack is older than the live docs; it is regenerated after the docs pass (Core, then Docs, then Bench). Undo: `git revert` that commit.
- D2. bench-web: dropped the uncommitted `LabLogo.tsx` reformat (its only content change was corrupt SVG path data) and the stray `bun.lock` (the repo uses npm and `package-lock.json`). Undo: `git apply` the backup patch kept outside the repo during this session; the committed file is the correct one.
- D1. Ledger replaces older audit and plan notes. No older audit or plan files remain in the library repo (`RELEASE-AUDIT-v1.0.0.md` is gone); nothing to consolidate there.

## Findings

Format: id, priority, area, evidence, status.

| Id | P | Area | Evidence | Status |
| --- | --- | --- | --- | --- |
| BASE-1 | P2 | tests | `test/v150/c22-ml-manifold.test.ts:346` wall-clock assertion fails under load (6178 ms vs 5000 ms). Flaky by design. | fixed 6b9f026 |
| BASE-2 | P3 | docs | `CONTRIBUTING.md` says `prose:check` is not part of `validate:all`; `package.json` runs it first. | fixed 6b9f026 |
| BENCHPRIV-1 | P2 | bench-private | `npm run docs:check` fails: `bench/docs.ts` older than DeepboxDocs. | open (after docs pass) |
| BENCHWEB-1 | P1 | bench-web | Uncommitted `src/app/components/LabLogo.tsx` corrupts the DeepSeek and MiniMax SVG paths. | dropped (D2) |
| PKG-1 | P1 | packaging | Emitted `.d.ts` use extensionless relative imports; `moduleResolution: node16/nodenext` users get TS2834 errors, or untyped imports with `skipLibCheck`. attw flags all 14 entry points. Reproduced. | fixed bb0672f |
| PKG-2 | P2 | packaging | `import {tensor} from "deepbox/ndarray"` bundles 342 KB minified; namespace `__export` objects share chunks with code. Marking them pure gives 127 KB. | fixed (tensor bundle 342,456 to 127,321 bytes) |
| PKG-3 | P2 | packaging | `engines.node >=24.13.0` blocks Node 22 with `--engine-strict`, while all entry points, WASM and a fit/predict smoke test pass on Node 22.23.4. | fixed ce103a0 (CI confirms 22.12.0 on push) |
| PKG-4 | P2 | packaging | Dual package hazard: ESM and CJS copies do not share `Tensor` class, error classes, seed or backend registry. | documented (README Import Model) |
| PKG-5 | P2 | packaging | Browser bundling of `deepbox/plot`, `deepbox/datasets` and the root fails on `fs/promises`, `path`, `os` dynamic imports (tsup strips `node:`). | fixed 99ea22e |
| PKG-6 | P3 | packaging | No `./package.json` export. | fixed bb0672f |
| PKG-7 | P3 | packaging | Published 1.5.0 differs from `main` in README numbers, five `.d.ts` JSDoc edits and `bench:python`; no `[Unreleased]` in CHANGELOG; published without provenance. | open |
| PKG-8 | P3 | types | Public `.d.ts` need `WebAssembly` and `AbortSignal` types without referencing them. | kept: consumers get WebAssembly and AbortSignal from @types/node or the DOM lib; a triple-slash DOM reference would force DOM types on Node users |
| PKG-9 | P3 | types | `any` in public signatures: `DataFrame.filter` row, `DataLoader` collate types. | owner decision: Record<string, unknown> rows would break typed user code such as row.age > 30; a generic row type is a 2.0 change |
| SEC-1 | P2 | CI | `release.yml` grants `contents: write` and `id-token: write` during `npm ci` and the full test run; long-lived `NPM_TOKEN`; no check that the tag is on `main`; `ci.yml` has no `permissions` block; actions not SHA-pinned. | fixed 9a6b355 (trusted publishing is an owner action) |
| SEC-2 | P3 | deps | `npm audit`: 7 dev-only advisories (vitest, vite, postcss, nanoid, source-map-js); 0 in the runtime tree. Three Dependabot branches unmerged since March. | open |
| SEC-3 | P3 | history | Five published commits on `main` (73a7d90, 0befc08, 2761140, fd8bac9, 49180bb) carry agent co-author trailers; author name spelled two ways. History is not rewritten (owner rule); `.mailmap` can unify names. | owner decision |
| SEC-4 | P3 | repo | Issue chooser has no private security report link. | fixed 6b9f026 |
| SEC-5 | P3 | deps | `allowScripts` covers `webgpu` only; `npm ci` warns about `esbuild`. | fixed ce103a0 |
| DOC-1 | P2 | docs | Same as BASE-2; also `validate:fix` omits `prose:check`. | fixed 6b9f026, ce103a0 |
| DOC-2 | P2 | repo | Five dead old-layout benchmark files (`benchmarks/0{2,3,5,6}-*/deepset.ts`, `benchmarks/07-scalability/deepbox.ts`) referenced nowhere. | fixed 6b9f026 |
| DOC-3 | P2 | dataframe | `df.concat([df])` throws raw `TypeError` (`other._data` undefined). Reproduced. | fixed da0e738 |
| DOC-4 | P3 | prose | `prose:check` skips root config files; `tsup.config.ts:23` has an em dash; 22 en dashes in tracked files. | open |
| DOC-5 | P3 | docs | `benchmarks/README.md:115` matmul range does not match `RESULTS.md`. | open |
| DOC-6 | P3 | JSDoc | `src/ndarray/ops/radix.ts:21` links to a 404 docs page. | open |
| DOC-7 | P3 | dataframe | `DataFrame.corr` returns `0.9999999999999998` for perfect correlation; pandas returns exactly 1. | fixed ae00f04 |
| DF-1 | P0 | dataframe io | `readXlsx`: one far-right cell per row allocates 16384-wide rows; 46 KB file uses 1.2 GB RSS (verified), 233 KB file crashes the process with heap OOM. `src/dataframe/io/xlsx.ts:765`. | fixed 99aa89a |
| DF-2 | P0 | dataframe io | `readXlsx`: unterminated `<row>` tags make lazy regexes quadratic (1.2 MB file, 43 s) and return 0 rows with no error. `xlsx.ts:740,758,721`. Verified at 289 KB, 2.1 s, 0 rows. | fixed 99aa89a |
| DF-3 | P1 | dataframe io | `fromCsvString`: a quote inside an unquoted field toggles quote mode; `12" steel` throws, `say "hi" there` loses its quotes. Verified. `DataFrame.ts:2396`. | fixed 712d8e7 |
| DF-4 | P1 | dataframe io | CSV integers beyond 2^53 silently rounded. Verified. `DataFrame.ts:2481`. | fixed 712d8e7 |
| DF-5 | P1 | dataframe dt | `dt.isoformat()` returns UTC with `Z` while other accessors use local wall-clock fields. `DateTimeAccessor.ts:597`. | fixed d40d610 (behavior change) |
| DF-6 | P2 | dataframe dt | Naive datetimes stored in local time: DST gap shifts 02:30 to 03:30, `date_range` skips an hour. | owner decision: switch naive datetimes to UTC fields (breaking) |
| DF-7 | P2 | dataframe | `pctChange` returns null on a zero previous value (pandas: inf). | fixed 03d7397 |
| DF-8 | P2 | dataframe | `Expanding` lacks `median` (Rolling has it). | fixed 3aa2964 |
| DF-9 | P2 | dataframe | `filter`/`assign(fn)` rows built on `{}`: a `__proto__` column is unreachable. | fixed 8368df3 |
| DF-10 | P2 | dataframe | `groupBy` on an empty frame drops value columns. | fixed 53561c2 |
| DF-11 | P2 | dataframe io | CSV sign plus leading zero inconsistent (`007` text, `-007` number); `1e400` becomes Infinity. | fixed 712d8e7 |
| DF-12 | P2 | dataframe | `merge`/`join` accept one key column only; `on: ""` gives a misleading error. | fixed 8cf476a |
| DF-13 | P2 | dataframe | `Series([]).sum()` throws; boolean `sum` throws (pandas: 0, count of true). | fixed 8e51b4b (behavior change) |
| DF-14 | P2 | datasets | `mapDataset` flattens N-D data and drops `images`. | fixed 8f656fa |
| DF-15 | P2 | datasets | Archive loaders: no timeout, download size cap, inflate cap, checksum or cache. | fixed 0874c84 |
| DF-16 | P2 | datasets | `fetchMNIST` does not validate labels. | fixed 0874c84 |
| DF-17 | P2 | datasets | `makeClassification` draws labels i.i.d., not balanced as documented by sklearn parity. | fixed e867cf8 (behavior change) |
| DF-18 | P2 | dataframe io | CSV reader: no `NA`/`N/A`/`TRUE` handling, duplicate headers throw (xlsx dedupes), short rows throw. | fixed 712d8e7 (behavior change: NA markers become null) |
| PL-1 | P2 | plot | `Axes.surface/wireframe/scatter3d` given Tensors draw nothing silently. | fixed c969759 |
| DF-19 | P3 | dataframe | `diff`/`pctChange` reject negative periods. | fixed 03d7397 |
| DF-20 | P3 | dataframe | `valueCounts({dropna:false})` counts NaN and null separately. | fixed b58f5fa |
| DF-21 | P3 | dataframe | Mixed-type sort gives an inconsistent order instead of an error. | fixed 0bd6121 |
| DF-22 | P3 | dataframe | groupBy `first`/`last` return missing values (pandas skips them). | option added 996c2d0; default flip is owner decision |
| DF-23 | P3 | dataframe | `astype("number")` coerces silently (`"0x10"` to 16); `astype("boolean")` maps `"no"` to true. | fixed c6fbdc2 (behavior change) |
| DF-24 | P3 | dataframe | `StyleAccessor.toANSI` passes control characters; CJK width wrong. | fixed 131f567 |
| DF-25 | P3 | dataframe | `Categorical` not integrated with DataFrame/Series. | deferred: needs its own design (construction, astype, groupBy, sort) |
| DF-26 | P3 | dataframe | `sample({replace:true})` resets row labels. | documented 96dd247 |
| NN-1 | P1 | nn | `Trainer` with a loss that is a plain Tensor trains nothing and reports a constant loss. Verified. `Trainer.ts:379`. | fixed d3ef700 (behavior change: throws) |
| NN-2 | P1 | nn | `gaussianNLLLoss`, `poissonNLLLoss`, `cosineEmbeddingLoss`, `ctcLoss` return non-differentiable Tensors for GradTensor inputs. Verified. | fixed 9619307 |
| NN-3 | P1 | nn | `crossEntropyLoss`/`nllLoss` have no default `ignoreIndex` (torch: -100); padded batches throw. Verified. | fixed d08f0c8 (behavior change: -100 ignored) |
| NN-4 | P2 | nn | `stateDict()` loses a parameter named `__proto__`. | fixed a94bf71 |
| CORE-1 | P2 | core | `unregisterBackend()` leaves `defaultDevice` on the removed backend; `registerBackend("cpu", x)` accepted. | fixed 9241cd7 |
| NN-6 | P2 | nn | `crossEntropyLoss`/`binaryCrossEntropyWithLogitsLoss` return a JS number for plain Tensors; other losses return Tensors. | owner decision: return 0-d Tensor instead of number (breaking) |
| NN-7 | P2 | nn | `binaryCrossEntropyLoss` clips out-of-range predictions silently; log clamp differs from torch (-100). | fixed 67aa3f5 |
| NN-8 | P2 | nn | `crossEntropyLoss`/`nllLoss` reject `(N,C,d1...)` and unbatched `(C)` inputs. | fixed d08f0c8 |
| NN-9 | P2 | nn | `TransformerDecoderLayer` always applies a causal mask; no way to disable. | fixed 50eef96 |
| NN-10 | P2 | nn | Unknown or unsupported constructor options silently ignored (LSTM `dropout`/`projSize`, Conv2d `paddingMode`, ConvTranspose2d `groups`). | fixed 71e7a3e, e6ad6fd |
| NN-11 | P3 | optim | `MultiStepLR` rejects duplicate milestones (torch applies gamma twice). | fixed 0a59f17 |
| NN-12 | P3 | optim | `OneCycleLR`/`CyclicLR` do not cycle momentum. | fixed 0a59f17 (behavior change: momentum cycles by default, torch parity) |
| NN-13 | P3 | nn | Dropout rejects `p = 1`. | fixed ee72e03 |
| NN-14 | P3 | nn | `Module.train(mode)` does not validate `mode`. | fixed a94bf71 |
| NN-15 | P3 | nn | `loadStateDict(null)` throws raw TypeError. | fixed a94bf71 |
| NN-16 | P3 | optim | Optimizers accept an empty parameter list. | fixed 4188e2d (behavior change: throws) |
| NN-17 | P3 | nn | `Trainer` leaves model mode inconsistent after `fit`; verbose output bypasses `setLogHandler`. | fixed 9776715 |
| NN-18 | P3 | nn | LSTM `forwardWithState(x, hx)` without `cx` silently zeroes cell state. | fixed 71e7a3e |
| NN-19 | P3 | nn | RNN layers cannot consume `PackedSequence`. | deferred: needs PackedSequence.data to carry gradients (API widening) |
| NN-20 | P3 | nn | `PositionalEncoding` stores a 5000x512 float64 buffer in every `stateDict` (73 MB JSON). | fixed a94bf71 |
| NN-21 | P3 | core | `WorkerPool` is sequential; `taskTimeout` stored but not enforced. | fixed cb5c615 |
| NN-22 | P3 | core | `SerializedEstimator` unused; `save()` not atomic; checkpoints pretty-printed. | fixed 6a741a2 |
| NN-23 | P3 | nn | Conv2d and pooling reject unbatched `(C,H,W)` input. | fixed 63d599d |
| ND-1 | P1 | linalg | `eig`/`eigvals` throw for complex eigenvalues: 91 of 100 random 4x4 matrices. Verified. Documented limitation. | open (additive option; default is owner decision) |
| ND-2 | P1 | random | `shuffle`/`permutation`/`choice` on N-D tensors work over flattened elements and scramble rows (numpy/torch use axis 0). Verified. Documented, so changing the default is breaking. | owner decision; additive `axis` option |
| ND-3 | P2 | ndarray | `matmul` accepts 2-D only; `dot` is batched. | open |
| ND-4 | P2 | ndarray | int64 beyond 2^53: `searchsorted`, `isclose`, `allclose` compare as Number. Verified. | open |
| ND-5 | P2 | ndarray | int32/uint8/bool casts silently produce garbage (`NaN` to 0, 3e9 wraps, `"0x10"` to 16); int64 path throws `DTypeError`. Verified. | open |
| ND-6 | P2 | ndarray | String dtype unsupported in `equal`, `where`, `unique`, `flip`, `roll`, `booleanIndex`, `fancyIndex`, `union1d`, `sort`. | open |
| ND-7 | P2 | linalg | `sparseSolve` allocates a dense N x N workspace with no guard (n=20000 needs 3.2 GB). | open |
| ND-8 | P2 | ndarray | Unknown option keys ignored: `slice({start, stop})` drops `stop`; `histogram(t, {bins})` raw RangeError. Verified. | open |
| ND-9 | P2 | ndarray | `nanmax`/`nanmin` on empty return NaN while `max`/`min` throw (numpy raises). | open |
| ND-10 | P2 | ndarray | `astype` to the same dtype returns `this` (aliasing). Documented. | owner decision |
| ND-11 | P3 | linalg | `norm`, `cond`, `det`, `matrixRank` throw on NaN/Inf instead of propagating. | open |
| ND-12 | P3 | ndarray | `logspace` parameter order differs from `linspace`; `base` unvalidated. | open |
| ND-13 | P3 | ndarray | `@internal` `dropoutMask` exported; dropout rejects p = 1. | open |
| ND-14 | P3 | ndarray | `rot90` 3-D, set ops on N-D, `expandDims` axis tuple, 0-d axis 0, `repeat` message, `diff(bool)` narrower than numpy. | open |
| ND-15 | P3 | ndarray | `eye(3, {dtype})` misleading error; `multinomial` counts float32; integer GradTensor accepts `requiresGrad`. | open |
| ND-16 | P3 | ndarray | `gcd`/`lcm` truncate non-integer floats silently. | open |
| ND-17 | P3 | ndarray | `sort(x, undefined)` sorts last axis instead of flattening. | open |
| ND-18 | P3 | linalg | `trace` returns shape `[1]`; `cov` of 1-D returns `[1,1]`; one-observation `cov` throws. | open |
| ML-1 | P1 | ml | `crossValScore`/`crossValidate`/`GridSearchCV`/`RandomizedSearchCV` accept only integer `cv`; the shipped splitters (`KFold`, `TimeSeriesSplit`, `GroupKFold`) cannot be used, so time-ordered data leaks. Verified. | open |
| ML-2 | P1 | ml | `MLPClassifier` wraps labels outside int32 (3e9 becomes -1294967296). Verified. | open |
| ML-3 | P2 | ml | 74 estimator constructors ignore unknown keys (`KMeans({n_clusters})`, `SVC({probability})`, `TSNE({maxIter})`). | open |
| ML-4 | P2 | ml | `setParams(getParams())` throws for Voting, Stacking, CalibratedClassifierCV. | open |
| ML-5 | P2 | ml | LDA/QDA use unbiased covariance; JSDoc claims sklearn parity; probabilities differ by 1e-2. | open |
| ML-6 | P2 | ml | `Ridge` builds the p x p Gram matrix when p > n (p=3000: 10.9 s vs 0.6 s for svd). | open |
| ML-7 | P2 | ml | LDA/QDA use Jacobi eigen (x400 features: 12.5 s). | open |
| ML-8 | P2 | ml | 17 estimators silently ignore `sampleWeight`; `LinearSVR` has none. | open |
| ML-9 | P2 | ml | `coef`/`intercept` types differ across linear models (Tensor, Float64Array, number, array). | owner decision (unify with deprecated aliases) |
| ML-10 | P3 | ml | Defaults differ from sklearn (tree `maxDepth` 10, GMM `diag`, LinearSVC `hinge`, ...). Changing them is breaking. | owner decision; document in one table |
| ML-11 | P3 | ml | `KDTree`/`BallTree` keep a reference to the caller's buffer. | open |
| ML-12 | P3 | ml | Lasso/ElasticNet/Ridge/SGD/RANSAC/KernelRidge validate hyperparameters only in `fit`. | open |
| ML-13 | P3 | ml | TSNE `nIter` vs `maxIter`; predict label dtype varies; `crossValScore` positional vs options. | open |
| PP-1 | P1 | preprocess | Encoders and `KBinsDiscretizer` lack `getParams`/`setParams`/`clone`; Pipeline clone and CV fail. Verified. | open |
| MT-1 | P1 | metrics | `classificationReport` accepts only binary 0/1 labels. Verified. | open |
| MT-2 | P1 | metrics | No `posLabel`; `{-1,1}` and `{1,2}` labels throw by default. Verified. | open |
| MT-3 | P2 | metrics | Two-class string labels silently use "weighted" while the JSDoc says "binary". Verified. | open |
| PP-2 | P2 | preprocess | `ngramRange: [1, 1e9]` loops to 1e9 per document (8.5 s for 5 tokens). | open |
| PP-3 | P2 | preprocess | Scaler `setParams` accept unknown keys; `MaxAbsScaler.setParams` ignores all input. | open |
| PP-4 | P2 | preprocess | Text vectorizers lack `fit`/`transform`/`clone`; cannot be Pipeline steps. | open |
| MT-4 | P2 | metrics | `daviesBouldinScore`/`calinskiHarabaszScore` return a score for a single cluster (sklearn raises). | open |
| MT-5 | P2 | metrics | No multioutput regression or multilabel classification metrics. | open |
| MT-6 | P3 | metrics | `cohenKappaScore` lacks `sampleWeight`; error prints `[object Object]`. | open |
| MT-7 | P3 | metrics | String labels ordered by UTF-16 unit, not code point; `LabelEncoder` disagrees. | open |
| PP-5 | P3 | preprocess | `LabelBinarizer` returns 2 columns for 2 classes (sklearn 1). Changing is breaking. | owner decision; accept 1-column inverse |
| PP-6 | P3 | preprocess | Transformers upcast float32 to float64. | open |
| ST-1 | P3 | stats | `kendalltau` constant input returns NaN while `pearsonr`/`spearmanr` throw (scipy: NaN). | open |
| ST-2 | P3 | stats | `poisson(0)` rejected. | open |
| PP-7 | P3 | preprocess | `HashingVectorizer` uses FNV-1a, not MurmurHash3. | open |
| TEST-1 | P2 | tests | `test/plot-png-zlib-mock.test.ts` timed out at the 5 s default on an unchanged tree under load. | fixed b9ac513 |
| BUILD-1 | P3 | tooling | `benchmarks/tsconfig.json` had 143 type errors (no Node types) and was not checked. | fixed ce103a0 (`typecheck:bench` in `validate:all`) |
| DOCS-1 | P1 | docs site | `deepbox.dev` 307-redirects to `www`, but canonical, og:url, sitemap and robots use the apex. Verified. | in progress (canonical moves to `www`) |
| DOCS-2 | P1 | docs site | Example and project pages show a one-line README description in a "Console output" frame; 0 of 59 match real output. | in progress |
| DOCS-3 | P1 | docs site | "How to run" fails for 6 pages that use top-level await without `"type": "module"`. | in progress |
| DOCS-4 | P1 | JSDoc | Doc examples that throw: `partialcorr` (collinear data), `inv` (imports `matmul` from ndarray), random `choice` (imports `tensor` from random), `AlphaDropout` (`Sequential([...])`). | open (library JSDoc) |
| DOCS-5 | P2 | JSDoc | 253 of 667 `@example` blocks do not run as pasted (195 have no import). | open (library JSDoc) |
| DOCS-6 | P2 | docs site | og/twitter tags missing or site-default on subpages; 10000x5000 image declared 512x512; 5000x5000 favicon. | in progress |
| DOCS-7 | P2 | docs site | No Content-Security-Policy; `/favicon.ico` 404; HTML sends `access-control-allow-origin: *`. | in progress |
| DOCS-8 | P2 | docs site | Sitemap lists a redirect (`/docs`) and request-time `lastmod`; robots has `Host:`. | in progress |
| DOCS-9 | P2 | docs site | Search dialog swallows "/" (reads `event.currentTarget`). | in progress |
| DOCS-10 | P2 | JSDoc | British spellings in JSDoc (initialisation, neighbours, centres, materialised, recognised). | open (library JSDoc) |
| DOCS-11 | P3 | JSDoc | En dashes in JSDoc names (Golub-Reinsch, Bartels-Stewart, BFGS, 0-9). | open (library JSDoc, prose check extended) |
| DOCS-12 | P3 | docs site | Home lacks canonical; markdown in meta descriptions; duplicate charset; case-variant paths return 200. | in progress |
| DOCS-13 | P3 | docs site | No `llms.txt`; TTF fonts preloaded. | in progress |
| DOCS-14 | P3 | datasets | IMDB default URL refused from this network; figshare fine. | unverified (network) |

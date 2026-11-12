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
| BASE-1 | P2 | tests | `test/v150/c22-ml-manifold.test.ts:346` wall-clock assertion fails under load (6178 ms vs 5000 ms). Flaky by design. | open |
| BASE-2 | P3 | docs | `CONTRIBUTING.md` says `prose:check` is not part of `validate:all`; `package.json` runs it first. | open |
| BENCHPRIV-1 | P2 | bench-private | `npm run docs:check` fails: `bench/docs.ts` older than DeepboxDocs. | open (after docs pass) |
| BENCHWEB-1 | P1 | bench-web | Uncommitted `src/app/components/LabLogo.tsx` corrupts the DeepSeek and MiniMax SVG paths. | dropped (D2) |
| PKG-1 | P1 | packaging | Emitted `.d.ts` use extensionless relative imports; `moduleResolution: node16/nodenext` users get TS2834 errors, or untyped imports with `skipLibCheck`. attw flags all 14 entry points. Reproduced. | open |
| PKG-2 | P2 | packaging | `import {tensor} from "deepbox/ndarray"` bundles 342 KB minified; namespace `__export` objects share chunks with code. Marking them pure gives 127 KB. | open |
| PKG-3 | P2 | packaging | `engines.node >=24.13.0` blocks Node 22 with `--engine-strict`, while all entry points, WASM and a fit/predict smoke test pass on Node 22.23.4. | open |
| PKG-4 | P2 | packaging | Dual package hazard: ESM and CJS copies do not share `Tensor` class, error classes, seed or backend registry. | open |
| PKG-5 | P2 | packaging | Browser bundling of `deepbox/plot`, `deepbox/datasets` and the root fails on `fs/promises`, `path`, `os` dynamic imports (tsup strips `node:`). | open |
| PKG-6 | P3 | packaging | No `./package.json` export. | open |
| PKG-7 | P3 | packaging | Published 1.5.0 differs from `main` in README numbers, five `.d.ts` JSDoc edits and `bench:python`; no `[Unreleased]` in CHANGELOG; published without provenance. | open |
| PKG-8 | P3 | types | Public `.d.ts` need `WebAssembly` and `AbortSignal` types without referencing them. | open |
| PKG-9 | P3 | types | `any` in public signatures: `DataFrame.filter` row, `DataLoader` collate types. | open |
| SEC-1 | P2 | CI | `release.yml` grants `contents: write` and `id-token: write` during `npm ci` and the full test run; long-lived `NPM_TOKEN`; no check that the tag is on `main`; `ci.yml` has no `permissions` block; actions not SHA-pinned. | open |
| SEC-2 | P3 | deps | `npm audit`: 7 dev-only advisories (vitest, vite, postcss, nanoid, source-map-js); 0 in the runtime tree. Three Dependabot branches unmerged since March. | open |
| SEC-3 | P3 | history | Five published commits on `main` (73a7d90, 0befc08, 2761140, fd8bac9, 49180bb) carry agent co-author trailers; author name spelled two ways. History is not rewritten (owner rule); `.mailmap` can unify names. | owner decision |
| SEC-4 | P3 | repo | Issue chooser has no private security report link. | open |
| SEC-5 | P3 | deps | `allowScripts` covers `webgpu` only; `npm ci` warns about `esbuild`. | open |
| DOC-1 | P2 | docs | Same as BASE-2; also `validate:fix` omits `prose:check`. | open |
| DOC-2 | P2 | repo | Five dead old-layout benchmark files (`benchmarks/0{2,3,5,6}-*/deepset.ts`, `benchmarks/07-scalability/deepbox.ts`) referenced nowhere. | open |
| DOC-3 | P2 | dataframe | `df.concat([df])` throws raw `TypeError` (`other._data` undefined). Reproduced. | open |
| DOC-4 | P3 | prose | `prose:check` skips root config files; `tsup.config.ts:23` has an em dash; 22 en dashes in tracked files. | open |
| DOC-5 | P3 | docs | `benchmarks/README.md:115` matmul range does not match `RESULTS.md`. | open |
| DOC-6 | P3 | JSDoc | `src/ndarray/ops/radix.ts:21` links to a 404 docs page. | open |
| DOC-7 | P3 | dataframe | `DataFrame.corr` returns `0.9999999999999998` for perfect correlation; pandas returns exactly 1. | open |
| DF-1 | P0 | dataframe io | `readXlsx`: one far-right cell per row allocates 16384-wide rows; 46 KB file uses 1.2 GB RSS (verified), 233 KB file crashes the process with heap OOM. `src/dataframe/io/xlsx.ts:765`. | open |
| DF-2 | P0 | dataframe io | `readXlsx`: unterminated `<row>` tags make lazy regexes quadratic (1.2 MB file, 43 s) and return 0 rows with no error. `xlsx.ts:740,758,721`. Verified at 289 KB, 2.1 s, 0 rows. | open |
| DF-3 | P1 | dataframe io | `fromCsvString`: a quote inside an unquoted field toggles quote mode; `12" steel` throws, `say "hi" there` loses its quotes. Verified. `DataFrame.ts:2396`. | open |
| DF-4 | P1 | dataframe io | CSV integers beyond 2^53 silently rounded. Verified. `DataFrame.ts:2481`. | open |
| DF-5 | P1 | dataframe dt | `dt.isoformat()` returns UTC with `Z` while other accessors use local wall-clock fields. `DateTimeAccessor.ts:597`. | open |
| DF-6 | P2 | dataframe dt | Naive datetimes stored in local time: DST gap shifts 02:30 to 03:30, `date_range` skips an hour. | open |
| DF-7 | P2 | dataframe | `pctChange` returns null on a zero previous value (pandas: inf). | open |
| DF-8 | P2 | dataframe | `Expanding` lacks `median` (Rolling has it). | open |
| DF-9 | P2 | dataframe | `filter`/`assign(fn)` rows built on `{}`: a `__proto__` column is unreachable. | open |
| DF-10 | P2 | dataframe | `groupBy` on an empty frame drops value columns. | open |
| DF-11 | P2 | dataframe io | CSV sign plus leading zero inconsistent (`007` text, `-007` number); `1e400` becomes Infinity. | open |
| DF-12 | P2 | dataframe | `merge`/`join` accept one key column only; `on: ""` gives a misleading error. | open |
| DF-13 | P2 | dataframe | `Series([]).sum()` throws; boolean `sum` throws (pandas: 0, count of true). | open |
| DF-14 | P2 | datasets | `mapDataset` flattens N-D data and drops `images`. | open |
| DF-15 | P2 | datasets | Archive loaders: no timeout, download size cap, inflate cap, checksum or cache. | open |
| DF-16 | P2 | datasets | `fetchMNIST` does not validate labels. | open |
| DF-17 | P2 | datasets | `makeClassification` draws labels i.i.d., not balanced as documented by sklearn parity. | open |
| DF-18 | P2 | dataframe io | CSV reader: no `NA`/`N/A`/`TRUE` handling, duplicate headers throw (xlsx dedupes), short rows throw. | open |
| PL-1 | P2 | plot | `Axes.surface/wireframe/scatter3d` given Tensors draw nothing silently. | open |
| DF-19 | P3 | dataframe | `diff`/`pctChange` reject negative periods. | open |
| DF-20 | P3 | dataframe | `valueCounts({dropna:false})` counts NaN and null separately. | open |
| DF-21 | P3 | dataframe | Mixed-type sort gives an inconsistent order instead of an error. | open |
| DF-22 | P3 | dataframe | groupBy `first`/`last` return missing values (pandas skips them). | open |
| DF-23 | P3 | dataframe | `astype("number")` coerces silently (`"0x10"` to 16); `astype("boolean")` maps `"no"` to true. | open |
| DF-24 | P3 | dataframe | `StyleAccessor.toANSI` passes control characters; CJK width wrong. | open |
| DF-25 | P3 | dataframe | `Categorical` not integrated with DataFrame/Series. | open |
| DF-26 | P3 | dataframe | `sample({replace:true})` resets row labels. | open |
| NN-1 | P1 | nn | `Trainer` with a loss that is a plain Tensor trains nothing and reports a constant loss. Verified. `Trainer.ts:379`. | open |
| NN-2 | P1 | nn | `gaussianNLLLoss`, `poissonNLLLoss`, `cosineEmbeddingLoss`, `ctcLoss` return non-differentiable Tensors for GradTensor inputs. Verified. | open |
| NN-3 | P1 | nn | `crossEntropyLoss`/`nllLoss` have no default `ignoreIndex` (torch: -100); padded batches throw. Verified. | open |
| NN-4 | P2 | nn | `stateDict()` loses a parameter named `__proto__`. | open |
| CORE-1 | P2 | core | `unregisterBackend()` leaves `defaultDevice` on the removed backend; `registerBackend("cpu", x)` accepted. | open |
| NN-6 | P2 | nn | `crossEntropyLoss`/`binaryCrossEntropyWithLogitsLoss` return a JS number for plain Tensors; other losses return Tensors. | open |
| NN-7 | P2 | nn | `binaryCrossEntropyLoss` clips out-of-range predictions silently; log clamp differs from torch (-100). | open |
| NN-8 | P2 | nn | `crossEntropyLoss`/`nllLoss` reject `(N,C,d1...)` and unbatched `(C)` inputs. | open |
| NN-9 | P2 | nn | `TransformerDecoderLayer` always applies a causal mask; no way to disable. | open |
| NN-10 | P2 | nn | Unknown or unsupported constructor options silently ignored (LSTM `dropout`/`projSize`, Conv2d `paddingMode`, ConvTranspose2d `groups`). | open |
| NN-11 | P3 | optim | `MultiStepLR` rejects duplicate milestones (torch applies gamma twice). | open |
| NN-12 | P3 | optim | `OneCycleLR`/`CyclicLR` do not cycle momentum. | open |
| NN-13 | P3 | nn | Dropout rejects `p = 1`. | open |
| NN-14 | P3 | nn | `Module.train(mode)` does not validate `mode`. | open |
| NN-15 | P3 | nn | `loadStateDict(null)` throws raw TypeError. | open |
| NN-16 | P3 | optim | Optimizers accept an empty parameter list. | open |
| NN-17 | P3 | nn | `Trainer` leaves model mode inconsistent after `fit`; verbose output bypasses `setLogHandler`. | open |
| NN-18 | P3 | nn | LSTM `forwardWithState(x, hx)` without `cx` silently zeroes cell state. | open |
| NN-19 | P3 | nn | RNN layers cannot consume `PackedSequence`. | open |
| NN-20 | P3 | nn | `PositionalEncoding` stores a 5000x512 float64 buffer in every `stateDict` (73 MB JSON). | open |
| NN-21 | P3 | core | `WorkerPool` is sequential; `taskTimeout` stored but not enforced. | open |
| NN-22 | P3 | core | `SerializedEstimator` unused; `save()` not atomic; checkpoints pretty-printed. | open |
| NN-23 | P3 | nn | Conv2d and pooling reject unbatched `(C,H,W)` input. | open |

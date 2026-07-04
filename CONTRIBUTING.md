# Contributing to Deepbox

> Website: https://deepbox.dev
> Docs: https://deepbox.dev/docs
> Examples: https://deepbox.dev/examples
> Projects: https://deepbox.dev/projects

Thanks for contributing. This document covers the workflow and standards used in the `v1.0.0` repository.

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before participating.

## Requirements

- Node.js `>= 24.13.0` (matches `package.json` `engines` and CI; TypeScript targets ES2024)
- npm `>= 11`

## Setup

```bash
git clone https://github.com/jehaad1/Deepbox.git
cd Deepbox
npm ci
```

## Development Commands

```bash
npm run build
npm run dev
npm test
npm run test:coverage
npm run typecheck
npm run lint:check
npm run lint:fix
npm run format
npm run format:check
npm run validate:all
npm run validate:fix
npm run all
```

`npm run all` is the CI/release validation alias and should stay green before a PR is merged.

## Project Layout

| Path | Purpose |
| --- | --- |
| `src/core` | Types, errors, config, validation, logging, warnings, serialization, backends |
| `src/ndarray` | Tensor core, autograd, sparse matrices, operations |
| `src/linalg` | Decompositions, matrix functions, solvers |
| `src/dataframe` | `DataFrame`, `Series`, accessors, IO helpers |
| `src/stats` | Descriptive stats, distributions, tests, KDE, confidence intervals |
| `src/metrics` | Evaluation metrics |
| `src/preprocess` | Scalers, encoders, text, splitters, feature selection |
| `src/ml` | Classical ML estimators and composition utilities |
| `src/nn` | Modules, layers, losses, training helpers |
| `src/optim` | Optimizers and schedulers |
| `src/random` | RNG, distributions, sampling |
| `src/datasets` | Built-in datasets, generators, samplers, remote loaders |
| `src/plot` | Figure API, plot types, renderers |
| `test` | Vitest suite |
| `docs/examples` | 50 numbered examples (`00`–`49`) in the current tree |
| `docs/projects` | 9 larger end-to-end projects |
| `benchmarks` | Deepbox and Python benchmark harnesses (`npm run bench:deepbox`, `bench:python`, `bench:all`; optional `npm run bench:tensor` writes `deepbox-tensor.json` only) |

## API and Import Rules

- Prefer subpath imports in examples, tests, docs, and user-facing snippets:
  - `deepbox/ndarray`
  - `deepbox/ml`
  - `deepbox/dataframe`
- The root package exports namespaces only. Do not document or test direct named imports from `deepbox` unless you are using the namespace form:

```ts
import * as db from "deepbox";

db.ndarray.tensor([1, 2, 3]);
db.ml.LinearRegression;
```

## Coding Standards

- TypeScript runs in strict mode with additional safety flags enabled.
- Do not introduce `any`, `@ts-ignore`, `@ts-expect-error`, or unnecessary unsafe casts.
- Follow the existing naming conventions:
  - classes: PascalCase
  - functions and methods: camelCase
  - constants: UPPER_SNAKE_CASE when truly constant
- Keep public APIs documented with concise JSDoc where the surrounding module already follows that pattern.
- Preserve the zero-runtime-dependency policy unless an explicit architectural change is intended and reviewed.

## Error Handling

When extending Deepbox internals, prefer the framework's custom errors from `deepbox/core` instead of generic `Error`:

- `InvalidParameterError`
- `ShapeError`
- `BroadcastError`
- `DTypeError`
- `IndexError`
- `NotFittedError`
- `ConvergenceError`
- `DeviceError`
- `MemoryError`
- `DataValidationError`
- `NotImplementedError`

Error messages should be specific and include the offending value, shape, dtype, or parameter when practical.

## Tests

- Add or update tests for every user-visible behavior change.
- Place tests in `test/` and use the `*.test.ts` naming pattern. Vitest’s `include` pattern is `test/**/*.test.ts`; other `.ts` files in `test/` (for example `*_helpers.ts` or `*.bench.ts`) are not picked up as test files.
- Cover edge cases, invalid inputs, and regression paths when changing numerical code.
- Do not commit focused tests.
- If you change exports, add or update export coverage tests as needed.

## Docs Expectations

If a change affects the public surface area, update the relevant docs in the same PR:

- For new or substantially revised public APIs, prefer JSDoc with `@see https://deepbox.dev/docs/<slug>` when a matching docs page exists. Cross-links are still being expanded across the tree; add them where you touch code.
- Run `npm run jsdoc:docs-gaps` for a repo-local list of `src/**/*.ts` files that still omit a DeepboxDocs URL (`deepbox.dev/docs`). The script fails the process when gaps exist so `validate:all` stays honest.
- Run `npm run jsdoc:core-slugs` to ensure `src/core/**` files reference the correct docs slugs (`core-types`, `core-config`, `core-errors`, `core-utils`, or `devices-and-execution` for optional GPU/WASM backends), matching `DeepboxDocs` `core.json`.
- [README.md](README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [SKILL.md](SKILL.md)
- examples under `docs/examples`
- projects under `docs/projects`
- any related docs-site content maintained elsewhere in the monorepo

Do not document APIs that are not actually exported.

## Pull Requests

1. Branch from `main`.
2. Make the smallest coherent change that solves the problem.
3. Run `npm run validate:all`.
4. Update tests and docs for public changes.
5. Open a PR with a clear description, scope, and validation notes.

The PR template in [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) should be completed accurately.

## Issues and Questions

- Bugs: use the [bug report template](https://github.com/jehaad1/Deepbox/issues/new?template=bug_report.yml)
- Features: use the [feature request template](https://github.com/jehaad1/Deepbox/issues/new?template=feature_request.yml)
- Questions: use the [question template](https://github.com/jehaad1/Deepbox/issues/new?template=question.yml)

## Coordinated release (Deepbox + DeepboxDocs + DeepboxBench)

When cutting a version across the three products:

1. Bump `version` in each repo’s `package.json` and any public version strings (for example `DeepboxDocs/public/LLMs.txt` and docs navigation meta if you version them).
2. Update [CHANGELOG.md](CHANGELOG.md) in Deepbox.
3. Run `npm run build` in Deepbox, then `npm run validate:all` in **Deepbox**, **DeepboxDocs**, and **DeepboxBench** (docs and bench CI expect a built Deepbox `dist/` for snippet checks).
4. Publish the npm package from Deepbox; deploy the docs and benchmark sites per your hosting setup.

## Security

Do not report vulnerabilities in public issues. Follow [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

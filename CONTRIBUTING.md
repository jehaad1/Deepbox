# Contributing to Deepbox

> Website: https://deepbox.dev
> Docs: https://deepbox.dev/docs
> Examples: https://deepbox.dev/examples
> Projects: https://deepbox.dev/projects

Thanks for contributing. This document covers the workflow and standards for the Deepbox repository (release line 1.5).

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
npm run build            # tsup bundle plus type declarations
npm run dev              # tsup in watch mode
npm test                 # Vitest, whole suite
npm run test:coverage    # Vitest with coverage thresholds
npm run format           # biome format --write
npm run format:check     # biome format, no writes
npm run lint:check       # biome check, no writes
npm run lint:fix         # biome check --write
npm run typecheck        # tsc on src
npm run typecheck:test   # tsc on test (tsconfig.test.json)
npm run typecheck:docs   # tsc on docs/examples and docs/projects
npm run prose:check      # fails on em dashes in project text
npm run validate:all     # the full CI gate
npm run validate:fix     # same gate, but formats and fixes lint first
```

`npm run validate:all` is the CI and release gate. It runs `format:check`, `lint:check`, `typecheck`, `typecheck:test`, `typecheck:docs`, the two JSDoc checks, then a build, the tests, a benchmark smoke run, every example, every project and coverage. `npm run all` is an alias for it. Keep it green before a PR is merged. `prose:check` is not part of `validate:all`, so run it yourself before you push.

For a quick loop, run one file with `npx vitest run test/<file>.test.ts` and format only what you changed with `npx biome check --write <files>`.

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
| `test` | Vitest suite. `test/v150` holds the regression tests for the 1.5.0 audit |
| `docs/examples` | Numbered, runnable examples |
| `docs/projects` | Larger end-to-end projects |
| `benchmarks` | Deepbox and Python benchmark harnesses (`npm run bench:deepbox`, `bench:python`, `bench:all`) |
| `scripts` | Repository checks, such as `prose:check` and the JSDoc link checks |

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
- Put tests in `test/` and name them `*.test.ts`. Vitest's `include` pattern is `test/**/*.test.ts`. Other `.ts` files in `test/` (helpers such as `*_helpers.ts`, benches such as `*.bench.ts`) are not run as tests.
- Regression tests for bugs fixed in 1.5.0 and tests for new 1.5.0 behavior live in `test/v150`. Add new tests of that kind there and name the file after the area it covers. Tests for older, unchanged behavior stay next to their module's existing test file.
- Cover edge cases, invalid inputs and regression paths when changing numerical code. Check expected values against NumPy, SciPy, scikit-learn, PyTorch or pandas when one applies, and say which in a comment.
- Do not commit focused tests (`.only`).
- If you change exports, add or update export coverage tests.
- Coverage thresholds are set in `vitest.config.ts`. `npm run test:coverage` fails below them.

## Docs Expectations

If a change affects the public surface, update the relevant docs in the same PR:

- For new or substantially revised public APIs, prefer JSDoc with `@see https://deepbox.dev/docs/<slug>` when a matching docs page exists. Cross-links are still being added; add them where you touch code.
- Run `npm run jsdoc:docs-gaps` for a list of `src/**/*.ts` files that still lack a `deepbox.dev/docs` URL. The script fails when gaps exist, so `validate:all` stays honest.
- Run `npm run jsdoc:core-slugs` to check that `src/core/**` files reference the correct docs slugs (`core-types`, `core-config`, `core-errors`, `core-utils`, or `devices-and-execution` for the optional GPU and WASM backends), matching `DeepboxDocs` `core.json`.
- [README.md](README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [SKILL.md](SKILL.md)
- examples under `docs/examples`
- projects under `docs/projects`
- any related docs-site content maintained elsewhere in the monorepo

Do not document APIs that are not exported. Every code sample must run as written: run it before you commit it, and use `npm run typecheck:docs` for examples and projects.

### Writing rules

These apply to code comments, JSDoc, console output, Markdown and any other text in the repository. `npm run prose:check` enforces the first rule.

- Do not write em dashes (U+2014). Rewrite the sentence. Use a colon between a label and its description, a comma or a period between two clauses, "vs" for comparisons and "n/a" for an empty table cell. Do not swap in an en dash or a double hyphen.
- Write plain, specific, formal English in short sentences. Say what the code does and how to use it. Prefer concrete facts to adjectives.
- Avoid marketing adjectives and filler phrases. State the fact instead.
- No emoji, including in console output. No rhetorical questions. Do not bold every other phrase.
- Use canonical camelCase API names. Mention a deprecated snake_case name only to help someone migrate.

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

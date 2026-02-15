# Contributing to Deepbox

> **Website:** https://deepbox.dev · **Docs:** https://deepbox.dev/docs · **Examples:** https://deepbox.dev/examples · **Projects:** https://deepbox.dev/projects

Thanks for helping improve Deepbox! This guide covers the full development workflow.

## Code of Conduct

Please read and follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Requirements

- Node.js `>= 24.13.0`
- npm `>= 11.6.2`

## Setup

```bash
git clone https://github.com/jehaad1/Deepbox.git
cd Deepbox
npm install
```

## Development Commands

```bash
npm run build          # Build with tsup (ESM + CJS + .d.ts)
npm run dev            # Watch mode
npm test               # Run all tests (vitest)
npm run test:coverage  # Run tests with V8 coverage
npm run typecheck      # tsc --noEmit
npm run lint           # Biome check
npm run lint:fix       # Biome check --write
npm run format         # Biome format --write
npm run format:check   # Biome format (check only)
npm run all            # format:check + lint:check + typecheck + build + test + coverage
npm run all:fix        # format + lint:fix + typecheck + build + test + coverage
```

## Project Structure

```
src/
  core/         # Types, errors, config, validation
  ndarray/      # Tensors, autograd, sparse, ops
  linalg/       # Decompositions, solvers, norms
  dataframe/    # DataFrame, Series
  stats/        # Descriptive stats, correlations, hypothesis tests
  metrics/      # Classification, regression, clustering metrics
  preprocess/   # Scalers, encoders, splits
  ml/           # Classical ML models
  nn/           # Neural network layers, losses, module system
  optim/        # Optimizers, LR schedulers
  random/       # Random distributions, sampling
  datasets/     # Built-in datasets, generators, DataLoader
  plot/         # SVG/PNG visualization
test/           # 260 test files
docs/examples/  # 33 educational examples (00-32) → https://deepbox.dev/examples
docs/projects/  # 6 enterprise-grade projects → https://deepbox.dev/projects
```

## Writing Code

### Style

- **Formatter/Linter**: Biome (configured in `biome.json`)
- **Indent**: 2 spaces
- **Quotes**: double
- **Semicolons**: always
- **Line width**: 100 characters
- **Trailing commas**: ES5

### TypeScript Rules

- **Strict mode** with all checks enabled
- `noUncheckedIndexedAccess`: every indexed access must be guarded
- `exactOptionalPropertyTypes`: optional properties cannot be `undefined` unless explicitly typed
- No `any`, no unsafe `as` casts, no `!` non-null assertions, no `@ts-ignore`
- All public APIs must have JSDoc with `@param`, `@returns`, and `@throws` tags

### Error Handling

- Never use generic `throw new Error()`
- Use the appropriate custom error from `core/errors/`:
  - `ShapeError` — shape mismatches
  - `DTypeError` — dtype incompatibilities
  - `BroadcastError` — broadcasting failures
  - `IndexError` — out-of-bounds access
  - `InvalidParameterError` — bad constructor/function arguments
  - `NotFittedError` — calling predict before fit
  - `DataValidationError` — invalid input data
- Error messages must be descriptive and include actual values

### Naming Conventions

- **Files**: camelCase for modules (`LinearRegression.ts`), lowercase for utilities (`_internal.ts`)
- **Functions**: camelCase (`trainTestSplit`, `f1Score`)
- **Classes**: PascalCase (`StandardScaler`, `GradTensor`)
- **Constants**: UPPER_SNAKE_CASE for true constants (`DTYPES`, `DEVICES`)
- **Internal APIs**: prefix with `_` or place in `_internal.ts`

## Writing Tests

- Test files go in `test/` with `.test.ts` extension
- Use `vitest` (`describe`, `it`, `expect`)
- Cover edge cases: empty input, single element, NaN, large values, wrong dtypes
- Do not use `test.only` or `test.skip` in committed code
- Run `npm test` to verify all 4,344 tests pass before submitting

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes with tests
3. Run `npm run all` — it must pass completely
4. Write a clear PR description explaining what changed and why
5. One approval required for merge

## Reporting Bugs

Use the [bug report template](https://github.com/jehaad1/Deepbox/issues/new?template=bug_report.yml). Include:

- Deepbox version and Node.js version
- Minimal reproduction code
- Expected vs actual behavior

## Requesting Features

Use the [feature request template](https://github.com/jehaad1/Deepbox/issues/new?template=feature_request.yml).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

# Random Sampling & Distributions

> **View online:** https://deepbox.dev/examples/21-random-sampling

Draw random numbers from common distributions. `setSeed` makes every later draw reproducible, which helps with simulations, tests and repeatable experiments.

## Deepbox Modules Used

| Module            | Features Used                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `deepbox/random`  | `setSeed`, `rand`, `randn`, `randint`, `uniform`, `normal`, `binomial`, `poisson`, `exponential`, `gamma`, `beta`, `multivariateNormal`, `choice`, `permutation`, `Generator` |
| `deepbox/ndarray` | `tensor`                                                                                                              |

## What It Shows

- Seeding twice with the same value repeats the same numbers.
- Thirteen sampling calls, each printing its result with a short description.
- `multivariateNormal(mean, cov, n)` draws `n` correlated rows. The camelCase name replaces `multivariate_normal`, which still works.
- `choice(items, k)` samples with replacement, `choice(items, k, false)` without. `permutation` returns a shuffled copy.
- A `Generator` has its own state and is not affected by `setSeed`.
- Random tensors are `float32` by default. Integer draws are `int32`.

## Usage

```bash
npm run example:21
```

## Output

Console output only. The numbers are the same on every run because of the seed. They differ from 1.0.0 because the random streams changed in 1.5.0.

## Files

```
21-random-sampling/
├── index.ts     # Main entry point
└── README.md    # This file
```

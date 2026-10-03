# Learning Rate Schedulers

> **View online:** https://deepbox.dev/examples/16-lr-schedulers

A scheduler changes the optimizer's learning rate as training goes on. This example steps eight schedulers for a few epochs and prints the learning rate at each one.

## Deepbox Modules Used

| Module          | Features Used                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `deepbox/nn`    | `Linear`, `ReLU`, `Sequential`                                                                                      |
| `deepbox/optim` | `Adam`, `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `ReduceLROnPlateau`, `WarmupLR`, `OneCycleLR` |

## What It Shows

- `StepLR` multiplies the rate by `gamma` every `stepSize` epochs. `MultiStepLR` does it at the listed milestones.
- `ExponentialLR` multiplies by `gamma` every epoch. `CosineAnnealingLR` follows a half cosine from the base rate down to `etaMin` over `tMax` epochs.
- `LinearLR` and `WarmupLR` ramp the rate up. `OneCycleLR` rises to `maxLr` and then falls.
- `ReduceLROnPlateau` is driven by a metric: `step(loss)` lowers the rate once the loss has not improved for `patience` epochs.
- Call `scheduler.step()` once per epoch, after `optimizer.step()`. `getLastLr()` returns one rate per parameter group.
- Example 41 covers more schedulers: warm restarts, lambda, cyclic, polynomial and sequential.

## Usage

```bash
npm run example:16
```

## Output

Console output only: the learning rate per epoch for each scheduler.

## Files

```
16-lr-schedulers/
├── index.ts     # Main entry point
└── README.md    # This file
```

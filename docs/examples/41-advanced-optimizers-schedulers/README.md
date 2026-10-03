# Advanced Optimizers & Schedulers

> **View online:** https://deepbox.dev/examples/41-advanced-optimizers-schedulers

Trains a small network with the `RAdam`, `LAMB` and `LARS` optimizers, then prints the learning rate curves of the `CyclicLR`, `CosineAnnealingWarmRestarts`, `PolynomialLR`, `LambdaLR` and `SequentialLR` schedulers.

## Deepbox Modules Used

| Module            | Features Used                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `deepbox/optim`   | RAdam, LAMB, LARS, CyclicLR, CosineAnnealingWarmRestarts, PolynomialLR, LambdaLR, SequentialLR, Adam, StepLR |
| `deepbox/nn`      | Linear, Sequential, ReLU, mseLoss                                                                            |
| `deepbox/ndarray` | tensor                                                                                                       |
| `deepbox/random`  | setSeed                                                                                                      |

## Usage

```bash
npm run example:41
```

## Output

- Console output only: loss and learning rate at epochs 1, 10, 20 and 30 for each optimizer, and the learning rate over 20 steps for each scheduler.
- Training uses plain tensors. `model.forward(x)` returns a tensor that tracks the weights, so `loss.backward()` and `loss.item()` need no wrapping of the data.
- `CosineAnnealingWarmRestarts` takes `t0` and `tMult`. The older `T_0` and `T_mult` still work.

## Files

```
41-advanced-optimizers-schedulers/
├── index.ts     # Example script
└── README.md    # This file
```

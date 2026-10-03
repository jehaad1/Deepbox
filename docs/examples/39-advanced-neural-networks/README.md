# Advanced Neural Networks

> **View online:** https://deepbox.dev/examples/39-advanced-neural-networks

Eight parts on neural network building blocks: weight initialization, GELU and PReLU, `LayerNorm` and `GroupNorm`, `Embedding`, `ModuleList` and `ModuleDict`, a `Sequential` model with dropout, and the `Trainer` with early stopping, gradient accumulation and best-weight restore. `GELU` uses the tanh approximation by default, while PyTorch uses the exact form. Pass `{ approximate: "none" }` for the exact form.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/nn`      | Trainer, EarlyStopping, ModelCheckpoint, Embedding, GroupNorm, LayerNorm, PReLU, GELU, ModuleList, ModuleDict, Sequential, Linear, ReLU, Dropout, xavierUniform_, kaimingNormal_ |
| `deepbox/optim`   | Adam                                                                                                                                                                             |
| `deepbox/ndarray` | tensor, noGrad                                                                                                                                                                   |
| `deepbox/random`  | setSeed                                                                                                                                                                          |

## Usage

```bash
npm run example:39
```

## Output

- Console output only: layer outputs and shapes, parameter counts, the `Trainer` history summary, and `EarlyStopping` and `ModelCheckpoint` decisions for fixed loss sequences. The script calls `setSeed(42)`, so the weights and the training results repeat between runs.
- Training uses plain tensors. `model.forward(x)` returns a `GradTensor` that tracks the weights, so data does not need `parameter(...)`. Inference runs inside `noGrad()`.

## Files

```
39-advanced-neural-networks/
├── index.ts     # Example script
└── README.md    # This file
```

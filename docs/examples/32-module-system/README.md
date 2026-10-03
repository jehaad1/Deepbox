# Neural Network Module System

> **View online:** https://deepbox.dev/examples/32-module-system

Covers the `Module` base class: writing a custom module, listing parameters, saving and loading a state dict, train and eval modes, freezing parameters, forward hooks and the `Sequential` container. It also shows when `forward` returns a `GradTensor` and when it returns a plain `Tensor`.

## Deepbox Modules Used

| Module            | Features Used                                                                    |
| ----------------- | -------------------------------------------------------------------------------- |
| `deepbox/ndarray` | `tensor`, `noGrad`, `GradTensor.isGradTensor`                                    |
| `deepbox/nn`      | `Module`, `Linear`, `ReLU`, `Sequential`, `stateDict`, `freezeParameters`, hooks |

## Usage

```bash
npm run example:32
```

## Notes

- With gradient tracking on and trainable weights, `forward` on a plain tensor returns a `GradTensor` that tracks the weights. Data does not need `parameter(...)`.
- Inside `noGrad()`, or when every parameter is frozen, `forward` returns a plain `Tensor`.
- `eval()` switches layer behavior (dropout, batch norm). It does not stop gradient tracking.
- `parameters()` yields `GradTensor` values, so reading `.shape` or `.requiresGrad` needs no type check.

## Output

- Console output only: parameter names and shapes, state dict keys, the `training` flag, counts of trainable parameters after freezing, the type returned by `forward` in each situation, and a forward hook firing once.

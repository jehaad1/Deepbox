# Neural Network Training

> **View online:** https://deepbox.dev/examples/13-neural-network-training

Train small networks with plain tensors. Covers `Sequential` models, a custom `Module`, the `mseLoss` function, `Adam` and `SGD` with momentum.

## Deepbox Modules Used

| Module            | Features Used                                          |
| ----------------- | ------------------------------------------------------ |
| `deepbox/ndarray` | `tensor`, `noGrad`                                     |
| `deepbox/nn`      | `Linear`, `ReLU`, `Sequential`, `Module`, `mseLoss`    |
| `deepbox/optim`   | `Adam`, `SGD`                                          |

## What It Shows

- The data are plain tensors. You do not wrap them in `parameter(...)`.
- When gradient mode is on and a module has trainable parameters, `model.forward(x)` returns a `GradTensor` that tracks the weights. `mseLoss(model.forward(X), y)` is then a scalar you can call `backward()` on.
- The training step is `optimizer.zeroGrad()`, forward and loss, `loss.backward()`, `optimizer.step()`.
- `loss.item()` reads the scalar loss. Its type is a union of number, bigint and string, so wrap it in `Number(...)` before calling `toFixed`.
- `noGrad(() => ...)` turns tracking off for evaluation, and the results are plain tensors.
- A custom network extends `Module`, registers its children with `registerModule`, and implements `forward`. `stateDict()` lists the parameter names.

## Usage

```bash
npm run example:13
```

## Output

Console output only: loss every 50 epochs, predictions next to targets, the state dict keys, the evaluation loss and the final SGD loss.

## Files

```
13-neural-network-training/
├── index.ts     # Main entry point
└── README.md    # This file
```

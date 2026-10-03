# Automatic Differentiation (Autograd)

> **View online:** https://deepbox.dev/examples/14-autograd

Deepbox records the operations done on tracked tensors (`GradTensor`) as a computation graph. `backward()` walks the graph in reverse and fills in the gradient of each tracked input.

## Deepbox Modules Used

| Module            | Features Used                                                  |
| ----------------- | -------------------------------------------------------------- |
| `deepbox/ndarray` | `parameter`, `noGrad`, `tensor`, `GradTensor` methods          |
| `deepbox/nn`      | `Linear`                                                       |

## What It Shows

- `parameter(...)` creates a tracked tensor. Use it only for values you want gradients for. Training data stays a plain tensor.
- A `GradTensor` has the same method surface as `Tensor`, and plain numbers work in a chain: `p.mul(2).sub(3).relu().sum()`.
- `x.grad` holds the gradient after `backward()`. `item()` reads a scalar result.
- Gradients accumulate across `backward()` calls until you call `zeroGrad()`.
- `noGrad(() => ...)` skips recording, so results have `requiresGrad` set to `false`.
- A layer called with a plain tensor still returns a tracked result, because its weights are trainable. Part 6 reads the weight and bias gradients with `namedParameters()`.

## Usage

```bash
npm run example:14
```

## Output

Console output only: values and gradients for each part. The expected gradients are given in comments next to the code.

## Files

```
14-autograd/
├── index.ts     # Main entry point
└── README.md    # This file
```

# Automatic Differentiation (Autograd)

> **View online:** https://deepbox.dev/examples/14-autograd

Deepbox's autograd system tracks operations on GradTensors to build a computation graph, then computes gradients via reverse-mode differentiation.

## Deepbox Modules Used

| Module            | Features Used                         |
| ----------------- | ------------------------------------- |
| `deepbox/ndarray` | GradTensor, parameter, noGrad, tensor |

## Usage

```bash
npm run example:14
```

## Output

- Console output demonstrating basic gradients, multi-variable gradients, chained operations, noGrad inference, and gradient accumulation

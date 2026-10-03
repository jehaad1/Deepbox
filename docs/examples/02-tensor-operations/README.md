# Tensor Operations

> **View online:** https://deepbox.dev/examples/02-tensor-operations

Arithmetic, math functions and reductions on tensors, including broadcasting.

## Deepbox Modules Used

| Module            | Features Used                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `deepbox/ndarray` | `add`, `sub`, `mul`, `div`, `sqrt`, `exp`, `log`, `sin`, `cos`, `sum`, `mean`, `max`, `min`, `argmax`, `item` |

## What It Shows

- Every operation exists as a function (`add(a, b)`) and as a tensor method (`a.add(b)`). Both forms give the same result. Methods chain: `a.mul(10).add(1)`.
- A JavaScript number broadcasts against every element and does not change the tensor's dtype.
- Two tensors with different shapes broadcast following the NumPy rules, so `[2, 1] + [3]` gives `[2, 3]`.
- Reductions (`sum`, `mean`, `max`, `min`) take an optional axis. They return tensors, and `item()` converts a one-element result to a number.
- Index results such as `argmax` are `int32`. Float results keep the float dtype of the input.

## Usage

```bash
npm run example:02
```

## Output

Console output only.

## Files

```
02-tensor-operations/
├── index.ts     # Main entry point
└── README.md    # This file
```

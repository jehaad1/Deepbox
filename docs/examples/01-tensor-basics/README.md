# Tensor Basics

> **View online:** https://deepbox.dev/examples/01-tensor-basics

How to create and inspect tensors, the N-dimensional arrays that every other module is built on.

## Deepbox Modules Used

| Module            | Features Used                                                           |
| ----------------- | ----------------------------------------------------------------------- |
| `deepbox/ndarray` | `tensor`, `zeros`, `ones`, `eye`, `arange`, `linspace`, `reshape`, `.T`, `at`, `item` |

## What It Shows

- Build 1D, 2D and 3D tensors from nested JavaScript arrays and read `shape` and `size`.
- Create filled tensors (`zeros`, `ones`, `eye`) and sequences (`arange`, `linspace`).
- Reshape with `t.reshape([2, 3])` and transpose with `t.T`.
- Read one element with `at(...)` and a reduction result with `item()`.
- `tensor()` creates `float32` tensors by default. Pass `{ dtype: "float64" }` when you need double precision.

## Usage

```bash
npm run example:01
```

## Output

Console output only: each tensor is printed with its shape and dtype.

## Files

```
01-tensor-basics/
├── index.ts     # Main entry point
└── README.md    # This file
```

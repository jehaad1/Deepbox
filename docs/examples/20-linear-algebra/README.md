# Linear Algebra Operations

> **View online:** https://deepbox.dev/examples/20-linear-algebra

Determinant, inverse, norms, matrix power, SVD, QR, LU, symmetric eigendecomposition and linear systems. Each decomposition is checked by multiplying the factors back together.

## Deepbox Modules Used

| Module            | Features Used                                                        |
| ----------------- | -------------------------------------------------------------------- |
| `deepbox/linalg`  | `det`, `inv`, `trace`, `norm`, `matrixPower`, `svd`, `qr`, `lu`, `eigh`, `solve` |
| `deepbox/ndarray` | `tensor`, `diag`, `matmul`, `dot`, `.T`                              |

## What It Shows

- `det` and `norm` return plain numbers. `trace` returns a tensor.
- `svd(B, false)` gives the reduced factors: for a `[3, 2]` matrix, `U` is `[3, 2]`. The default is the full SVD, where `U` is `[3, 3]`.
- `lu(D)` returns `[P, L, U]` with `D = P * L * U`.
- `eigh` is for symmetric matrices. It returns the eigenvalues in ascending order and the eigenvectors as columns.
- `matmul` needs two 2D tensors. To multiply a matrix by a vector, use `dot`.
- `matrixPower(A, n)` is the camelCase name. The old `matrix_power` still works.
- Every decomposition prints the largest error of the reconstruction, which should be at rounding level.

## Usage

```bash
npm run example:20
```

## Output

Console output only.

## Files

```
20-linear-algebra/
├── index.ts     # Main entry point
└── README.md    # This file
```

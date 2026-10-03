# Advanced Linear Algebra Toolkit

> **View online:** https://deepbox.dev/examples/49-advanced-linear-algebra

Linear algebra routines beyond SVD, QR and LU: Hessenberg and Schur decompositions, polar decomposition, matrix functions (`expm`, `logm`, `sqrtm`, `matrixPower`), banded and sparse solvers, Sylvester and Lyapunov equations, and special matrix builders. Reconstruction errors are computed with the fluent `Tensor` methods `matmul`, `T`, `sub`, `square`, `sum` and `sqrt`.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/linalg`  | `hessenberg`, `schur`, `polar`, `expm`, `logm`, `sqrtm`, `matrixPower`, `solveBanded`, `denseToCSR`, `sparseSolve`, `sylvester`, `lyapunov`, `toeplitz`, `hadamard`, `blockDiag` |
| `deepbox/ndarray` | `tensor`                                                                                                                                                                         |

## Usage

```bash
npm run example:49
```

## Output

- Console output only: reconstruction errors, solver results, matrix equation solutions and special matrices.
- The older names `matrix_power`, `solve_banded` and `block_diag` still work. Use the camelCase names.

## Files

```text
49-advanced-linear-algebra/
├── index.ts
└── README.md
```

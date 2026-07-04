# Advanced Linear Algebra Toolkit

> **View online:** https://deepbox.dev/examples/49-advanced-linear-algebra

A focused v1.0.0 linear algebra example covering the new advanced routines missing from the earlier decomposition walkthrough: Hessenberg and Schur decompositions, polar decomposition, matrix functions, structured solvers, sparse CSR solving, and Sylvester/Lyapunov equations.

## Deepbox Modules Used

| Module           | Features Used |
| ---------------- | ------------- |
| `deepbox/linalg` | `hessenberg`, `schur`, `polar`, `expm`, `logm`, `sqrtm`, `matrix_power`, `solve_banded`, `denseToCSR`, `sparseSolve`, `sylvester`, `lyapunov`, `toeplitz`, `hadamard`, `block_diag` |
| `deepbox/ndarray`| `tensor` |

## Usage

```bash
npm run example:49
```

## Output

- Console walkthrough with reconstruction errors, solver results, and special-matrix examples

## Architecture

```text
49-advanced-linear-algebra/
├── index.ts
└── README.md
```

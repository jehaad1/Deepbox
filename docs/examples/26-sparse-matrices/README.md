# Sparse Matrix Operations

> **View online:** https://deepbox.dev/examples/26-sparse-matrices

`CSRMatrix` (Compressed Sparse Row) stores only the non-zero entries of a matrix. This example builds one from COO triplets and runs addition, scaling, element-wise and matrix products, transpose and conversion back to a dense tensor.

## Deepbox Modules Used

| Module            | Features Used                                                                         |
| ----------------- | ------------------------------------------------------------------------------------- |
| `deepbox/ndarray` | CSRMatrix (fromCOO, add, scale, multiply, matvec, matmul, transpose, toDense), tensor |

## Usage

```bash
npm run example:26
```

## Output

- Console output only: sparsity, element access, arithmetic, matrix-vector and matrix-matrix products, transpose and dense conversion.

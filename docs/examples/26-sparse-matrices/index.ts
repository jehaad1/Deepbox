/**
 * Example 26: Sparse Matrices
 *
 * CSR (Compressed Sparse Row) matrices store only the non-zero entries.
 * This example builds a CSRMatrix from COO triplets and runs the arithmetic,
 * products, transpose and dense conversion.
 */

import { CSRMatrix, tensor } from "deepbox/ndarray";

const expectFloat64Array = (value: unknown): Float64Array => {
  if (!(value instanceof Float64Array)) {
    throw new Error("Expected Float64Array");
  }
  return value;
};

console.log("=== Sparse Matrix Operations ===\n");

// Create a sparse matrix using COO (Coordinate) format
console.log("1. Creating sparse matrices");
// Matrix:
// [1, 0, 0, 2]
// [0, 3, 0, 0]
// [0, 0, 4, 0]
// [5, 0, 0, 6]
const sparse = CSRMatrix.fromCOO({
  rows: 4,
  cols: 4,
  rowIndices: new Int32Array([0, 0, 1, 2, 3, 3]),
  colIndices: new Int32Array([0, 3, 1, 2, 0, 3]),
  values: new Float64Array([1, 2, 3, 4, 5, 6]),
});
console.log("  Created 4x4 sparse matrix with 6 non-zero elements");
console.log(`  Sparsity: ${((1 - sparse.nnz / (sparse.rows * sparse.cols)) * 100).toFixed(1)}%`);

// Element access
console.log("\n2. Element access");
console.log(`  Element at (0,0): ${sparse.get(0, 0)}`);
console.log(`  Element at (0,3): ${sparse.get(0, 3)}`);
console.log(`  Element at (1,1): ${sparse.get(1, 1)}`);

// Arithmetic operations
console.log("\n3. Sparse matrix addition");
// Matrix:
// [0, 1, 0, 0]
// [2, 0, 0, 0]
// [0, 0, 0, 3]
// [0, 4, 0, 0]
const sparse2 = CSRMatrix.fromCOO({
  rows: 4,
  cols: 4,
  rowIndices: new Int32Array([0, 1, 2, 3]),
  colIndices: new Int32Array([1, 0, 3, 1]),
  values: new Float64Array([1, 2, 3, 4]),
});
const sum = sparse.add(sparse2);
console.log(`  Result has ${sum.nnz} non-zero elements`);

console.log("\n4. Scalar multiplication");
const scaled = sparse.scale(2);
console.log(`  Scaled matrix has ${scaled.nnz} non-zero elements`);
console.log(`  Element at (0,0): ${scaled.get(0, 0)} (was ${sparse.get(0, 0)})`);

console.log("\n5. Element-wise multiplication (Hadamard product)");
const product = sparse.multiply(sparse2);
console.log(`  Product has ${product.nnz} non-zero elements`);

// Matrix-vector multiplication
console.log("\n6. Matrix-vector multiplication");
const vec = tensor([1, 2, 3, 4]);
const result = sparse.matvec(vec);
const resultData = expectFloat64Array(result.data);
console.log(`  Result: [${Array.from(resultData).join(", ")}]`);

// Matrix-matrix multiplication
console.log("\n7. Matrix-matrix multiplication");
// Dense matrix (4x2):
// [1, 0]
// [0, 1]
// [1, 1]
// [0, 1]
const dense = tensor([
  [1, 0],
  [0, 1],
  [1, 1],
  [0, 1],
]);
const matmul = sparse.matmul(dense);
console.log(`  Result shape: [${matmul.shape.join(", ")}]`);
console.log("  The product of a sparse and a dense matrix is a dense tensor");

// Transpose
console.log("\n8. Transpose");
const transposed = sparse.transpose();
console.log(`  Original: ${sparse.rows}x${sparse.cols}`);
console.log(`  Transposed: ${transposed.rows}x${transposed.cols}`);
console.log(`  Non-zero elements preserved: ${transposed.nnz}`);

// Convert back to dense
console.log("\n9. Convert to dense");
const densified = sparse.toDense();
console.log("  Converted back to dense tensor");
console.log(`  Shape: [${densified.shape.join(", ")}]`);

console.log("\n=== When to use CSRMatrix ===");
console.log("- Memory use grows with the number of non-zeros, not with rows * cols");
console.log(
  "- Products touch only the stored entries, so they are cheaper when most entries are zero"
);
console.log(
  "- Typical uses: graph adjacency matrices, bag-of-words features, finite-difference operators"
);

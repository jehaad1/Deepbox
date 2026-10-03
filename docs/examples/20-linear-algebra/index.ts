/**
 * Example 20: Linear Algebra Operations
 *
 * Determinant, inverse, norms, matrix decompositions (SVD, QR, LU, eigen) and
 * linear systems. Each decomposition is checked by multiplying the factors
 * back together and comparing with the original matrix.
 */

import { det, eigh, inv, lu, matrixPower, norm, qr, solve, svd, trace } from "deepbox/linalg";
import { diag, type Tensor, tensor } from "deepbox/ndarray";

// Largest absolute difference between two tensors of the same shape.
const maxError = (a: Tensor, b: Tensor): string =>
  Number(a.sub(b).abs().max().item()).toExponential(2);

console.log("=== Linear Algebra Operations ===\n");

// Create a matrix
const A = tensor([
  [4, 2],
  [3, 1],
]);

console.log("Matrix A:");
console.log(`${A.toString()}\n`);

// Determinant (a plain number)
console.log(`Determinant: ${det(A).toFixed(4)}\n`);

// Trace (a tensor, read with item())
console.log(`Trace: ${Number(trace(A).item()).toFixed(4)}\n`);

// Matrix inverse, checked by multiplying with A
const invA = inv(A);
console.log("Inverse of A:");
console.log(`${invA.toString()}\n`);
console.log(`Largest error in A * inv(A) - I: ${maxError(A.matmul(invA), diag(tensor([1, 1])))}\n`);

// Matrix norms (plain numbers)
console.log(`Frobenius norm: ${norm(A, "fro").toFixed(4)}\n`);

// Integer matrix power: A^3 = A * A * A
console.log("A cubed with matrixPower(A, 3):");
console.log(`${matrixPower(A, 3).toString()}\n`);

// SVD Decomposition
console.log("SVD Decomposition:");
console.log("-".repeat(50));

const B = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

// The second argument false asks for the reduced SVD: U is [3, 2] instead of [3, 3].
const [U, S, Vt] = svd(B, false);
console.log("U (left singular vectors):");
console.log(U.toString());
console.log("\nS (singular values):");
console.log(S.toString());
console.log("\nVt (right singular vectors transposed):");
console.log(`${Vt.toString()}\n`);
console.log(
  `Largest error in U * diag(S) * Vt - B: ${maxError(U.matmul(diag(S)).matmul(Vt), B)}\n`
);

// QR Decomposition
console.log("QR Decomposition:");
console.log("-".repeat(50));

const [Q, R] = qr(B);
console.log("Q (orthonormal columns):");
console.log(Q.toString());
console.log("\nR (upper triangular):");
console.log(`${R.toString()}\n`);
console.log(`Largest error in Q * R - B: ${maxError(Q.matmul(R), B)}\n`);

// LU Decomposition
console.log("LU Decomposition:");
console.log("-".repeat(50));

const D = tensor([
  [1, 2],
  [3, 4],
]);

// P is a permutation matrix and D = P * L * U.
const [P, L, Ulu] = lu(D);
console.log("P (permutation):");
console.log(P.toString());
console.log("\nL (lower triangular):");
console.log(L.toString());
console.log("\nU (upper triangular):");
console.log(Ulu.toString());
console.log(`\nLargest error in P * L * U - D: ${maxError(P.matmul(L).matmul(Ulu), D)}\n`);

// Eigendecomposition of a symmetric matrix
console.log("Eigendecomposition (symmetric):");
console.log("-".repeat(50));

const M = tensor([
  [2, 1],
  [1, 2],
]);
// Eigenvalues come back in ascending order. Eigenvectors are the columns of V.
const [eigenvalues, V] = eigh(M);
console.log("Eigenvalues:", eigenvalues.toString());
console.log("Eigenvectors (columns):");
console.log(V.toString());
console.log(
  `Largest error in V * diag(w) * V^T - M: ${maxError(V.matmul(diag(eigenvalues)).matmul(V.T), M)}\n`
);

// Solving linear systems: Ax = b
console.log("Solving Linear System Ax = b:");
console.log("-".repeat(50));

const ASys = tensor([
  [3, 1],
  [1, 2],
]);
const b = tensor([9, 8]);

// dot() multiplies a matrix by a vector. matmul() needs two 2D tensors.
const x = solve(ASys, b);
console.log("Solution x:");
console.log(x.toString());
console.log(`Largest error in A * x - b: ${maxError(ASys.dot(x), b)}`);

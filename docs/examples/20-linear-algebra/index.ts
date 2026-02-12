/**
 * Example 20: Linear Algebra Operations
 *
 * Explore matrix decompositions and linear system solving.
 * Essential for understanding ML algorithms under the hood.
 */

import { det, inv, lu, norm, qr, solve, svd, trace } from "deepbox/linalg";
import { tensor } from "deepbox/ndarray";

console.log("=== Linear Algebra Operations ===\n");

// Create a matrix
const A = tensor([
  [4, 2],
  [3, 1],
]);

console.log("Matrix A:");
console.log(`${A.toString()}\n`);

// Determinant
const detA = det(A);
const detValue = Number(detA);
console.log(`Determinant: ${detValue.toFixed(4)}\n`);

// Trace
const traceA = trace(A);
const traceValue = typeof traceA === "number" ? traceA : Number(traceA);
console.log(`Trace: ${traceValue.toFixed(4)}\n`);

// Matrix inverse
const invA = inv(A);
console.log("Inverse of A:");
console.log(`${invA.toString()}\n`);

// Matrix norms
const frobNorm = norm(A, "fro");
console.log(`Frobenius norm: ${Number(frobNorm).toFixed(4)}\n`);

// SVD Decomposition
console.log("SVD Decomposition:");
console.log("-".repeat(50));

const B = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

const svdResult = svd(B);
const U = svdResult[0];
const S = svdResult[1];
const Vt = svdResult[2];
console.log("U (left singular vectors):");
console.log(U.toString());
console.log("\nS (singular values):");
console.log(S.toString());
console.log("\nVt (right singular vectors transposed):");
console.log(`${Vt.toString()}\n`);

// QR Decomposition
console.log("QR Decomposition:");
console.log("-".repeat(50));

const C = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

const qrResult = qr(C);
const Q = qrResult[0];
const R = qrResult[1];
console.log("Q (orthogonal matrix):");
console.log(Q.toString());
console.log("\nR (upper triangular):");
console.log(`${R.toString()}\n`);

// LU Decomposition
console.log("LU Decomposition:");
console.log("-".repeat(50));

const D = tensor([
  [2, 1],
  [1, 2],
]);

const luResult = lu(D);
const L = luResult[0];
const U_lu = luResult[1];
const P = luResult[2];
console.log("L (lower triangular):");
console.log(L.toString());
console.log("\nU (upper triangular):");
console.log(U_lu.toString());
console.log("\nP (permutation):");
console.log(`${P.toString()}\n`);

// Solving linear systems: Ax = b
console.log("Solving Linear System Ax = b:");
console.log("-".repeat(50));

const A_sys = tensor([
  [3, 1],
  [1, 2],
]);
const b = tensor([9, 8]);

const x = solve(A_sys, b);
console.log("Solution x:");
console.log(x.toString());

console.log("\n✓ Linear algebra operations complete!");

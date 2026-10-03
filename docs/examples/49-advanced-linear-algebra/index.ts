/**
 * Example 49: Advanced Linear Algebra Toolkit
 *
 * Decompositions and solvers beyond SVD, QR and LU: Hessenberg and Schur
 * decompositions, polar decomposition, matrix functions, banded and sparse
 * solvers, Sylvester and Lyapunov equations, and special matrices.
 * Reconstruction errors use the fluent Tensor methods (matmul, T, sub, square, sum).
 */

import {
  blockDiag,
  denseToCSR,
  expm,
  hadamard,
  hessenberg,
  logm,
  lyapunov,
  matrixPower,
  polar,
  schur,
  solveBanded,
  sparseSolve,
  sqrtm,
  sylvester,
  toeplitz,
} from "deepbox/linalg";
import { type Tensor, tensor } from "deepbox/ndarray";

// Frobenius norm of the difference between two tensors
const frobeniusDiff = (a: Tensor, b: Tensor): number =>
  Number(a.sub(b).square().sum().sqrt().item());

console.log("=".repeat(72));
console.log("Example 49: Advanced Linear Algebra Toolkit");
console.log("=".repeat(72));

// ============================================================================
// Part 1: Hessenberg and Schur decompositions
// ============================================================================
console.log("\nPart 1: Hessenberg + Schur");
console.log("-".repeat(72));

const systemMatrix = tensor([
  [4, 1, 0],
  [1, 3, 1],
  [0, 1, 2],
]);

const [hessenbergForm, hessenbergQ] = hessenberg(systemMatrix);
const [schurT, schurQ] = schur(systemMatrix);

// Both decompositions satisfy A = Q H Q^T
const hessenbergReconstruction = hessenbergQ.matmul(hessenbergForm).matmul(hessenbergQ.T);
const schurReconstruction = schurQ.matmul(schurT).matmul(schurQ.T);

console.log(
  `Hessenberg reconstruction error: ${frobeniusDiff(hessenbergReconstruction, systemMatrix).toExponential(3)}`
);
console.log(
  `Schur reconstruction error:      ${frobeniusDiff(schurReconstruction, systemMatrix).toExponential(3)}`
);
console.log("Upper Hessenberg form:");
console.log(hessenbergForm.toString());

// ============================================================================
// Part 2: Polar decomposition
// ============================================================================
console.log("\nPart 2: Polar Decomposition");
console.log("-".repeat(72));

const featureTransform = tensor([
  [1.2, 0.3],
  [-0.4, 0.9],
]);
const [orthogonalPart, positivePart] = polar(featureTransform);
// A = U P
const polarReconstruction = orthogonalPart.matmul(positivePart);

console.log(
  `Polar reconstruction error: ${frobeniusDiff(polarReconstruction, featureTransform).toExponential(3)}`
);
console.log("Orthogonal factor U:");
console.log(orthogonalPart.toString());
console.log("Positive-semidefinite factor P:");
console.log(positivePart.toString());

// ============================================================================
// Part 3: Matrix functions and powers
// ============================================================================
console.log("\nPart 3: Matrix Functions");
console.log("-".repeat(72));

const diagonalDynamics = tensor([
  [1.1, 0],
  [0, 0.85],
]);
const diagonalExp = expm(diagonalDynamics);
const diagonalLog = logm(diagonalExp);
const covariance = tensor([
  [4, 1],
  [1, 3],
]);
const covarianceSqrt = sqrtm(covariance);
const covarianceRecovered = covarianceSqrt.matmul(covarianceSqrt);
const transition = tensor([
  [0.92, 0.08],
  [0.05, 0.95],
]);
const fiveStepTransition = matrixPower(transition, 5);

console.log("expm(A) for diagonal dynamics:");
console.log(diagonalExp.toString());
console.log("logm(expm(A)) recovers:");
console.log(diagonalLog.toString());
console.log(
  `sqrtm(C) * sqrtm(C) reconstruction error: ${frobeniusDiff(covarianceRecovered, covariance).toExponential(3)}`
);
console.log("Five-step Markov transition:");
console.log(fiveStepTransition.toString());

// ============================================================================
// Part 4: Structured dense and sparse solvers
// ============================================================================
console.log("\nPart 4: Structured Solvers");
console.log("-".repeat(72));

// Banded storage: one row per diagonal (upper, main, lower), as in SciPy's solve_banded
const tridiagonalBands = tensor([
  [0, -1, -1, -1],
  [4, 4, 4, 4],
  [-1, -1, -1, 0],
]);
const tridiagonalRhs = tensor([15, 10, 10, 15]);
const tridiagonalSolution = solveBanded([1, 1], tridiagonalBands, tridiagonalRhs);

const sparseSystem = tensor([
  [4, -1, 0, 0],
  [-1, 4, -1, 0],
  [0, -1, 4, -1],
  [0, 0, -1, 3],
]);
const sparseRhs = tensor([15, 10, 10, 10]);
const sparseSolution = sparseSolve(denseToCSR(sparseSystem), sparseRhs);

console.log("Banded tridiagonal solution:");
console.log(tridiagonalSolution.toString());
console.log("Sparse CSR solution:");
console.log(sparseSolution.toString());

// Residual of the sparse solve: A x - b should be zero. Both vectors become 4x1 columns for matmul.
const n = sparseRhs.size;
const sparseResidual = sparseSystem
  .matmul(sparseSolution.reshape([n, 1]))
  .sub(sparseRhs.reshape([n, 1]));
console.log(
  `Sparse residual (max abs): ${Number(sparseResidual.abs().max().item()).toExponential(3)}`
);

// ============================================================================
// Part 5: Sylvester and Lyapunov equations
// ============================================================================
console.log("\nPart 5: Matrix Equations");
console.log("-".repeat(72));

const aSylvester = tensor([
  [1, 2],
  [0, 3],
]);
const bSylvester = tensor([
  [4, 1],
  [0, 5],
]);
const cSylvester = tensor([
  [10, 15],
  [12, 24],
]);
const sylvesterSolution = sylvester(aSylvester, bSylvester, cSylvester);

const stableA = tensor([
  [-1, 0.2],
  [0, -0.6],
]);
const lyapunovQ = tensor([
  [2, 0.5],
  [0.5, 1],
]);
const lyapunovSolution = lyapunov(stableA, lyapunovQ);

console.log("Sylvester solution X:");
console.log(sylvesterSolution.toString());
console.log("Lyapunov solution X:");
console.log(lyapunovSolution.toString());

// ============================================================================
// Part 6: Special matrix builders
// ============================================================================
console.log("\nPart 6: Special Matrices");
console.log("-".repeat(72));

const smoothingToeplitz = toeplitz([4, -1, 0]);
const hadamardBasis = hadamard(4);
const blockSystem = blockDiag(smoothingToeplitz, tensor([[2]]));

console.log("Toeplitz smoothing kernel:");
console.log(smoothingToeplitz.toString());
console.log("Hadamard basis (n=4):");
console.log(hadamardBasis.toString());
console.log(`Block-diagonal system shape: [${blockSystem.shape.join(", ")}]`);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(72));
console.log("• hessenberg, schur: the usual first steps of eigenvalue algorithms");
console.log(
  "• polar: splits a matrix into an orthogonal factor and a positive semidefinite factor"
);
console.log(
  "• expm, logm, sqrtm, matrixPower: matrix functions and powers, for example Markov chains"
);
console.log("• solveBanded and sparseSolve: solvers that use the structure of the matrix");
console.log("• sylvester, lyapunov: matrix equations from control and filtering");
console.log("• toeplitz, hadamard, blockDiag: builders for structured test matrices");

console.log("\nAdvanced Linear Algebra Toolkit Complete!");
console.log("=".repeat(72));

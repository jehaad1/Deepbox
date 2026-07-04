/**
 * Example 49: Advanced Linear Algebra Toolkit
 *
 * Documents the v1.0.0 linear algebra expansion beyond the earlier SVD/QR/LU
 * walkthrough: Hessenberg reduction, Schur/polar decompositions, matrix
 * functions, structured solvers, sparse CSR solving, and matrix equations.
 */

import {
  block_diag,
  denseToCSR,
  expm,
  hadamard,
  hessenberg,
  logm,
  lyapunov,
  matrix_power,
  polar,
  schur,
  solve_banded,
  sparseSolve,
  sqrtm,
  sylvester,
  toeplitz,
} from "deepbox/linalg";
import { type Tensor, tensor } from "deepbox/ndarray";

function toMatrix(t: Tensor): number[][] {
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => Number(t.at(i, j)))
  );
}

function matmul(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = b[0]?.length ?? 0;
  const inner = b.length;
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += (a[i]?.[k] ?? 0) * (b[k]?.[j] ?? 0);
      }
      return sum;
    })
  );
}

function transpose(a: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0]?.length ?? 0;
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => a[i]?.[j] ?? 0)
  );
}

function frobeniusDiff(a: number[][], b: number[][]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < (a[i]?.length ?? 0); j++) {
      const diff = (a[i]?.[j] ?? 0) - (b[i]?.[j] ?? 0);
      sum += diff * diff;
    }
  }
  return Math.sqrt(sum);
}

function matvec(a: number[][], x: number[]): number[] {
  return a.map((row) => row.reduce((sum, value, index) => sum + value * (x[index] ?? 0), 0));
}

console.log("=".repeat(72));
console.log("Example 49: Advanced Linear Algebra Toolkit");
console.log("=".repeat(72));

// ============================================================================
// Part 1: Hessenberg and Schur decompositions
// ============================================================================
console.log("\n🏗️  Part 1: Hessenberg + Schur");
console.log("-".repeat(72));

const systemMatrix = tensor([
  [4, 1, 0],
  [1, 3, 1],
  [0, 1, 2],
]);

const [hessenbergForm, hessenbergQ] = hessenberg(systemMatrix);
const [schurT, schurQ] = schur(systemMatrix);

const systemDense = toMatrix(systemMatrix);
const hessenbergReconstruction = matmul(
  matmul(toMatrix(hessenbergQ), toMatrix(hessenbergForm)),
  transpose(toMatrix(hessenbergQ))
);
const schurReconstruction = matmul(
  matmul(toMatrix(schurQ), toMatrix(schurT)),
  transpose(toMatrix(schurQ))
);

console.log(
  `Hessenberg reconstruction error: ${frobeniusDiff(hessenbergReconstruction, systemDense).toExponential(3)}`
);
console.log(
  `Schur reconstruction error:      ${frobeniusDiff(schurReconstruction, systemDense).toExponential(3)}`
);
console.log("Upper Hessenberg form:");
console.log(hessenbergForm.toString());

// ============================================================================
// Part 2: Polar decomposition
// ============================================================================
console.log("\n🧭 Part 2: Polar Decomposition");
console.log("-".repeat(72));

const featureTransform = tensor([
  [1.2, 0.3],
  [-0.4, 0.9],
]);
const [orthogonalPart, positivePart] = polar(featureTransform);
const polarReconstruction = matmul(toMatrix(orthogonalPart), toMatrix(positivePart));

console.log(
  `Polar reconstruction error: ${frobeniusDiff(polarReconstruction, toMatrix(featureTransform)).toExponential(3)}`
);
console.log("Orthogonal factor U:");
console.log(orthogonalPart.toString());
console.log("Positive-semidefinite factor P:");
console.log(positivePart.toString());

// ============================================================================
// Part 3: Matrix functions and powers
// ============================================================================
console.log("\n🧮 Part 3: Matrix Functions");
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
const covarianceRecovered = matmul(toMatrix(covarianceSqrt), toMatrix(covarianceSqrt));
const transition = tensor([
  [0.92, 0.08],
  [0.05, 0.95],
]);
const fiveStepTransition = matrix_power(transition, 5);

console.log("expm(A) for diagonal dynamics:");
console.log(diagonalExp.toString());
console.log("logm(expm(A)) recovers:");
console.log(diagonalLog.toString());
console.log(
  `sqrtm(C) * sqrtm(C) reconstruction error: ${frobeniusDiff(covarianceRecovered, toMatrix(covariance)).toExponential(3)}`
);
console.log("Five-step Markov transition:");
console.log(fiveStepTransition.toString());

// ============================================================================
// Part 4: Structured dense and sparse solvers
// ============================================================================
console.log("\n🪜 Part 4: Structured Solvers");
console.log("-".repeat(72));

const tridiagonalBands = tensor([
  [0, -1, -1, -1],
  [4, 4, 4, 4],
  [-1, -1, -1, 0],
]);
const tridiagonalRhs = tensor([15, 10, 10, 15]);
const tridiagonalSolution = solve_banded([1, 1], tridiagonalBands, tridiagonalRhs);

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

const denseSparseSystem = toMatrix(sparseSystem);
const sparseResidual = matvec(
  denseSparseSystem,
  Array.from({ length: sparseSolution.shape[0] ?? 0 }, (_, i) => Number(sparseSolution.at(i)))
);
console.log(
  `Sparse residual preview: ${sparseResidual.map((value, index) => (value - Number(sparseRhs.at(index))).toFixed(6)).join(", ")}`
);

// ============================================================================
// Part 5: Sylvester and Lyapunov equations
// ============================================================================
console.log("\n📐 Part 5: Matrix Equations");
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
console.log("\n🧱 Part 6: Special Matrices");
console.log("-".repeat(72));

const smoothingToeplitz = toeplitz([4, -1, 0]);
const hadamardBasis = hadamard(4);
const blockSystem = block_diag(smoothingToeplitz, tensor([[2]]));

console.log("Toeplitz smoothing kernel:");
console.log(smoothingToeplitz.toString());
console.log("Hadamard basis (n=4):");
console.log(hadamardBasis.toString());
console.log(`Block-diagonal system shape: [${blockSystem.shape.join(", ")}]`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(72));
console.log(
  "• Hessenberg and Schur factorizations are the backbone for many advanced eigensolver workflows."
);
console.log("• Polar decomposition splits a transform into rotation-like and scale-like factors.");
console.log(
  "• expm/logm/sqrtm/matrix_power let you move between discrete and continuous matrix dynamics."
);
console.log(
  "• solve_banded and sparseSolve are the practical path once dense solves become structured or sparse."
);
console.log(
  "• Sylvester and Lyapunov solvers are useful for control, filtering, and state-space tooling."
);
console.log(
  "• Special matrix constructors help build repeatable test systems and numerical examples quickly."
);

console.log("\n✅ Advanced Linear Algebra Toolkit Complete!");
console.log("=".repeat(72));

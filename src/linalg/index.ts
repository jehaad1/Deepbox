/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

export {
  type CholeskyOptions,
  cholesky,
  type EigOptions,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  hessenberg,
  lu,
  polar,
  qr,
  schur,
  svd,
  svdvals,
} from "./decomposition/index";
export { inv, pinv } from "./inverse";
export {
  block_diag,
  blockDiag,
  circulant,
  companion,
  expm,
  hadamard,
  hankel,
  hilbert,
  kron,
  logm,
  matrix_power,
  matrixPower,
  sqrtm,
  toeplitz,
  vandermonde,
} from "./matrix_ops";
export { cond, norm } from "./norms";
export { det, matrixRank, slogdet, trace } from "./properties";
export {
  type CSRMatrix,
  denseToCSR,
  lstsq,
  lyapunov,
  type SolveTriangularOptions,
  type SparseMatrixInput,
  solve,
  solve_banded,
  solveBanded,
  solveTriangular,
  sparseCholeskySolve,
  sparseSolve,
  sylvester,
} from "./solvers/index";

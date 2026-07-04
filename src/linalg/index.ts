/**
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox documentation}
 */

export {
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
  circulant,
  companion,
  expm,
  hadamard,
  hankel,
  hilbert,
  kron,
  logm,
  matrix_power,
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
  solve,
  solve_banded,
  solveTriangular,
  sparseCholeskySolve,
  sparseSolve,
  sylvester,
} from "./solvers/index";

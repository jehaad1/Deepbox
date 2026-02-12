export {
  cholesky,
  type EigOptions,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  lu,
  qr,
  svd,
} from "./decomposition/index";
export { inv, pinv } from "./inverse";
export { cond, norm } from "./norms";
export { det, matrixRank, slogdet, trace } from "./properties";
export { lstsq, solve, solveTriangular } from "./solvers/index";

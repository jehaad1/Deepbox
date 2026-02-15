import { DeepboxError } from "./base";

/**
 * Details for convergence failures.
 *
 * @property iterations - Number of iterations completed before failure
 * @property tolerance - Tolerance value used for convergence checks
 */
export type ConvergenceErrorDetails = {
  readonly iterations?: number;
  readonly tolerance?: number;
};

/**
 * Error thrown when an iterative algorithm fails to converge within
 * the maximum allowed iterations.
 *
 * This is a thrown exception that extends {@link DeepboxError}; callers
 * should handle it with standard try/catch.
 *
 * @example
 * ```ts
 * if (iterations >= maxIterations) {
 *   throw new ConvergenceError(
 *     `Algorithm did not converge after ${maxIterations} iterations`
 *   );
 * }
 * ```
 *
 * References:
 * - Deepbox convergence diagnostics:
 *   https://deepbox.dev/docs/core-errors
 */
export class ConvergenceError extends DeepboxError {
  override name = "ConvergenceError";

  /** Number of iterations completed before convergence failure */
  readonly iterations?: number;

  /** Tolerance value used for convergence check */
  readonly tolerance?: number;

  constructor(message: string, details?: ConvergenceErrorDetails) {
    super(message);
    if (details?.iterations !== undefined) {
      this.iterations = details.iterations;
    }
    if (details?.tolerance !== undefined) {
      this.tolerance = details.tolerance;
    }
  }
}

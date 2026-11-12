/**
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { DeepboxError } from "./base";

/**
 * Error thrown when data validation fails.
 *
 * Used for input data that doesn't meet expected format or constraints.
 *
 * @example
 * ```ts
 * import { DataValidationError } from 'deepbox/core';
 *
 * const data: unknown = 'not an array';
 * try {
 *   if (!Array.isArray(data)) {
 *     throw new DataValidationError('Expected array input');
 *   }
 * } catch (error) {
 *   console.log((error as Error).message); // Expected array input
 * }
 * ```
 */
export class DataValidationError extends DeepboxError {
  override name = "DataValidationError";
}

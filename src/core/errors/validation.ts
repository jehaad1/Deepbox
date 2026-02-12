import { DeepboxError } from "./base";

/**
 * Error thrown when data validation fails.
 *
 * Used for input data that doesn't meet expected format or constraints.
 *
 * @example
 * ```ts
 * if (!Array.isArray(data)) {
 *   throw new DataValidationError('Expected array input');
 * }
 * ```
 */
export class DataValidationError extends DeepboxError {
  override name = "DataValidationError";
}

import { DeepboxError } from "./base";

/**
 * Error thrown when invalid parameters are passed to a function or method.
 *
 * @example
 * ```ts
 * function kmeans(k: number, maxIter: number) {
 *   if (k <= 0) {
 *     throw new InvalidParameterError('k must be positive', 'k', k);
 *   }
 *   // ... algorithm logic
 * }
 * ```
 */
export class InvalidParameterError extends DeepboxError {
  override name = "InvalidParameterError";

  /** The name of the invalid parameter */
  readonly parameterName?: string;

  /** The invalid value that was provided */
  readonly value?: unknown;

  constructor(message: string, parameterName?: string, value?: unknown) {
    super(message);
    if (parameterName !== undefined) {
      this.parameterName = parameterName;
    }
    if (value !== undefined) {
      this.value = value;
    }
  }
}

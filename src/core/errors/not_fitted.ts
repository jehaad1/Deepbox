import { DeepboxError } from "./base";

/**
 * Error thrown when a model or estimator is used before being fitted.
 *
 * Thrown when a model is used before calling fit().
 *
 * @example
 * ```ts
 * class MyModel {
 *   private fitted = false;
 *
 *   fit(X: Tensor, y: Tensor) {
 *     // ... fitting logic
 *     this.fitted = true;
 *   }
 *
 *   predict(X: Tensor) {
 *     if (!this.fitted) {
 *       throw new NotFittedError('Model must be fitted before prediction');
 *     }
 *     // ... prediction logic
 *   }
 * }
 * ```
 *
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */
export class NotFittedError extends DeepboxError {
  override name = "NotFittedError";

  /** The name of the model or estimator that was not fitted */
  readonly modelName?: string;

  constructor(message: string, modelName?: string, options?: { readonly cause?: unknown }) {
    super(message, options);
    if (modelName !== undefined) {
      this.modelName = modelName;
    }
  }
}

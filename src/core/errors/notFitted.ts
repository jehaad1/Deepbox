import { DeepboxError } from "./base";

/**
 * Error thrown when a model or estimator is used before being fitted.
 *
 * This is analogous to sklearn's NotFittedError.
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
 * References:
 * - scikit-learn NotFittedError: https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html
 */
export class NotFittedError extends DeepboxError {
  override name = "NotFittedError";

  /** The name of the model or estimator that was not fitted */
  readonly modelName?: string;

  constructor(message: string, modelName?: string) {
    super(message);
    if (modelName !== undefined) {
      this.modelName = modelName;
    }
  }
}

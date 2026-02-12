/**
 * Base class for all Deepbox-specific errors.
 *
 * Provides a stable `instanceof DeepboxError` discriminator
 * and consistent `cause` chaining.
 */
export class DeepboxError extends Error {
  override name = "DeepboxError";
  override cause?: unknown;

  constructor(message?: string, options?: { readonly cause?: unknown }) {
    super(message);
    if (options?.cause !== undefined) {
      this.cause = options.cause;
    }
    // Required when extending built-in classes in TypeScript
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

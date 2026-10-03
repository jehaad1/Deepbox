/**
 * Base class for all Deepbox-specific errors.
 *
 * Provides a stable `instanceof DeepboxError` discriminator
 * and consistent `cause` chaining.
 *
 * The `cause` property is only defined on the instance when a cause was
 * passed, so `"cause" in error` reliably tells whether one was supplied.
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */
export class DeepboxError extends Error {
  override name = "DeepboxError";

  constructor(message?: string, options?: { readonly cause?: unknown }) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    // Required when extending built-in classes in TypeScript
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

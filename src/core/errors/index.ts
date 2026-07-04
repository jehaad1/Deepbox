/**
 * Error exports for Deepbox Core.
 *
 * This file is a barrel that re-exports all core error types.
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

export { DeepboxError } from "./base";
export { BroadcastError } from "./broadcast";
export { ConvergenceError, type ConvergenceErrorDetails } from "./convergence";
export { DeviceError } from "./device";
export { DTypeError } from "./dtype";
export { IndexError } from "./index_error";
export { InvalidParameterError } from "./invalid_parameter";
export { MemoryError } from "./memory";
export { NotFittedError } from "./not_fitted";
export { NotImplementedError } from "./not_implemented";
export { ShapeError, type ShapeErrorDetails } from "./shape";
export { DataValidationError } from "./validation";

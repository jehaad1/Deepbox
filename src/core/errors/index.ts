/**
 * Error exports for Deepbox Core.
 *
 * This file is a barrel that re-exports all core error types.
 */

export { DeepboxError } from "./base";
export { BroadcastError } from "./broadcast";
export { ConvergenceError, type ConvergenceErrorDetails } from "./convergence";
export { DeviceError } from "./device";
export { DTypeError } from "./dtype";
export { IndexError } from "./indexError";
export { InvalidParameterError } from "./invalidParameter";
export { MemoryError } from "./memory";
export { NotFittedError } from "./notFitted";
export { NotImplementedError } from "./notImplemented";
export { ShapeError, type ShapeErrorDetails } from "./shape";
export { DataValidationError } from "./validation";

/**
 * Deepbox Core
 *
 * This module is the stable, dependency-light foundation shared by all other Deepbox
 * subpackages. It exposes:
 * - configuration helpers
 * - error types
 * - foundational types/constants
 * - runtime validation and utility helpers
 */

// Config
export type { DeepboxConfig } from "./config/index";
export {
  getConfig,
  getDevice,
  getDtype,
  getSeed,
  resetConfig,
  setConfig,
  setDevice,
  setDtype,
  setSeed,
} from "./config/index";

// Errors
export {
  BroadcastError,
  ConvergenceError,
  type ConvergenceErrorDetails,
  DataValidationError,
  DeepboxError,
  DeviceError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
  type ShapeErrorDetails,
} from "./errors/index";

// Types
export type {
  Axis,
  Device,
  DType,
  ElementOf,
  ScalarDType,
  Shape,
  TensorLike,
  TensorStorage,
  TypedArray,
} from "./types/index";

// Constants
export { DEVICES, DTYPES, isDevice, isDType } from "./types/index";

// Utilities
export {
  asReadonlyArray,
  dtypeToTypedArrayCtor,
  ensureNumericDType,
  getArrayElement,
  getBigIntElement,
  getElementAsNumber,
  getNumericElement,
  getShapeDim,
  getStringElement,
  isBigInt64Array,
  isNumericTypedArray,
  isTypedArray,
  type NumericDType,
  type NumericTypedArray,
  normalizeAxes,
  normalizeAxis,
  shapeToSize,
  validateArray,
  validateDevice,
  validateDtype,
  validateInteger,
  validateNonNegative,
  validateOneOf,
  validatePositive,
  validateRange,
  validateShape,
} from "./utils/index";

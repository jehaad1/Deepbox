/**
 * Deepbox Core
 *
 * This module is the stable, dependency-light foundation shared by all other Deepbox
 * subpackages. It exposes:
 * - configuration helpers
 * - error types
 * - foundational types/constants
 * - runtime validation and utility helpers
 * @see {@link https://deepbox.dev/docs/core-types | Types & validation}
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

// Backend
export type {
  Backend,
  BackendCapability,
  BackendInfo,
  BinaryKernelOp,
  CompiledWasmModule,
  DeviceBuffer,
  DeviceDType,
  GpuPipelineInfo,
  HostAcceleratorBackend,
  HostBinaryOp,
  Im2ColParams,
  KernelBackend,
  KernelLayout,
  PoolKernelOp,
  ReduceKernelOp,
  ShaderName,
  TernaryKernelOp,
  UnaryKernelOp,
  WasmModuleName,
} from "./backend/index";
export {
  CpuBackend,
  getBackend,
  getHostAccelerator,
  getKernelBackend,
  isBackendAvailable,
  isHostAcceleratorBackend,
  isKernelBackend,
  listBackends,
  registerBackend,
  unregisterBackend,
  WASM_BINARIES,
  WAT_MODULES,
  WasmBackend,
  WebGpuBackend,
  WGSL_SHADERS,
} from "./backend/index";
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
// Logger
export type { LogEntry, VerboseLevel } from "./logger";
export { getLogHandler, Logger, setLogHandler } from "./logger";
export type {
  PoolStatus,
  TaskResult,
  WorkerPoolOptions,
} from "./parallel/index";
// Parallel
export { availableCores, createWorkerPool, WorkerPool } from "./parallel/index";
// Serialization
export type {
  SerializedEstimator,
  SerializedModuleState,
  SerializedPayload,
  SerializedTensor,
} from "./serialization";
export { fromJSON, load, save, toJSON } from "./serialization";
// Types
export type {
  Axis,
  Device,
  DType,
  ElementOf,
  ExtendedTypedArray,
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
  check_array,
  check_is_fitted,
  check_X_y,
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
  shapesEqual,
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

// Warnings
export {
  catchWarnings,
  type DeepboxWarning,
  filterWarnings,
  resetWarnings,
  setWarningHandler,
  type WarningAction,
  type WarningCategory,
  warn,
} from "./warnings";

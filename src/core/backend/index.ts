/**
 * Backend abstraction layer exports.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

export type { Backend, BackendCapability, BackendInfo } from "./Backend";
export { CpuBackend } from "./CpuBackend";
export {
  type BinaryKernelOp,
  type DeviceBuffer,
  type DeviceDType,
  type HostAcceleratorBackend,
  type HostBinaryOp,
  type Im2ColParams,
  isHostAcceleratorBackend,
  isKernelBackend,
  type KernelBackend,
  type KernelLayout,
  type PoolKernelOp,
  type ReduceKernelOp,
  type TernaryKernelOp,
  type UnaryKernelOp,
} from "./kernels";
export {
  getBackend,
  getHostAccelerator,
  getKernelBackend,
  isBackendAvailable,
  listBackends,
  registerBackend,
  unregisterBackend,
} from "./registry";
export {
  type CompiledWasmModule,
  WAT_MODULES,
  WasmBackend,
  type WasmModuleName,
} from "./WasmBackend";
export {
  type GpuPipelineInfo,
  type ShaderName,
  WebGpuBackend,
  WGSL_SHADERS,
} from "./WebGpuBackend";
export { WASM_BINARIES } from "./wasm_modules.generated";

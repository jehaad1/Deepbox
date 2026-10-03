/**
 * Logical compute devices for tensors and modules.
 *
 * - `cpu`: Default; CPU execution is always available for built-in ndarray ops.
 * - `webgpu`: GPU execution. Register `WebGpuBackend` from `deepbox/core`
 *   (after `await backend.init()`); tensors created on (or moved to) this
 *   device store their data in GPU memory and the accelerated op set
 *   (element-wise arithmetic, activations, matmul, full reductions)
 *   dispatches to WGSL compute kernels automatically. Read results back
 *   with `await t.cpu()`.
 * - `wasm`: WASM SIMD host accelerator. Register `WasmBackend`; tensors on
 *   this device keep zero-copy host storage, and eligible ops (contiguous
 *   float32 add, sub, mul and div) run through embedded SIMD kernels with
 *   bit-identical results, falling back to the CPU implementation otherwise.
 *   The `dotContiguous` and `sumContiguous` helpers of `WasmBackend` accumulate
 *   in four float32 lanes, so they can differ from the sequential CPU result
 *   in the last bits.
 *
 * @example
 * ```ts
 * import type { Device } from 'deepbox/core';
 * import { setDevice } from 'deepbox/core';
 *
 * const device: Device = 'cpu';
 * setDevice(device);
 * ```
 * @see {@link https://deepbox.dev/docs/core-types | Deepbox documentation}
 */
export type Device = "cpu" | "webgpu" | "wasm";

/**
 * Array of all supported device types.
 *
 * Use this constant for validation or UI selection.
 *
 * @example
 * ```ts
 * import { DEVICES } from 'deepbox/core';
 *
 * console.log(DEVICES); // ['cpu', 'webgpu', 'wasm']
 * ```
 */
export const DEVICES: readonly Device[] = ["cpu", "webgpu", "wasm"];

/**
 * Type guard to check if a value is a valid Device.
 *
 * @param value - The value to check
 * @returns True if value is a valid Device, false otherwise
 *
 * @example
 * ```ts
 * import { isDevice } from 'deepbox/core';
 *
 * if (isDevice('cpu')) {
 *   console.log('Valid device');
 * }
 *
 * isDevice('gpu');  // false
 * isDevice('cpu');  // true
 * ```
 */
export function isDevice(value: unknown): value is Device {
  if (typeof value !== "string") {
    return false;
  }
  for (const d of DEVICES) {
    if (d === value) {
      return true;
    }
  }
  return false;
}

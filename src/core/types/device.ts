/**
 * Supported compute devices for tensor operations.
 *
 * - `cpu`: Standard CPU execution (always available)
 * - `webgpu`: GPU acceleration via WebGPU API (when supported)
 * - `wasm`: WebAssembly acceleration for better CPU performance
 *
 * @example
 * ```ts
 * import type { Device } from 'deepbox/core';
 * import { setDevice } from 'deepbox/core';
 *
 * const device: Device = 'cpu';
 * setDevice(device);
 * ```
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

/**
 * Backend abstraction layer for Deepbox.
 *
 * Defines the contract that every execution backend must satisfy.
 * The CPU backend is always registered and powers built-in ndarray ops.
 * Registering `WebGpuBackend` routes ops on `webgpu` tensors to WGSL
 * compute kernels (see `KernelBackend`), and registering `WasmBackend`
 * accelerates eligible float32 arithmetic with WASM SIMD (see
 * `HostAcceleratorBackend`).
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

import type { Device } from "../types/device";

/**
 * Capability flags that a backend may support.
 *
 * Consumers can query these before dispatching work to avoid
 * runtime surprises on unsupported devices.
 */
export type BackendCapability =
  | "matmul"
  | "conv2d"
  | "fft"
  | "reduction"
  | "elementwise"
  | "blas"
  | "random";

/**
 * Information about a registered backend.
 */
export type BackendInfo = {
  /** The device this backend serves. */
  readonly device: Device;
  /** Human-readable name (e.g. "Deepbox CPU Backend"). */
  readonly name: string;
  /** Whether this backend is currently available in the runtime. */
  readonly available: boolean;
  /** Capabilities this backend supports. */
  readonly capabilities: readonly BackendCapability[];
};

/**
 * The interface every Deepbox execution backend must implement.
 *
 * Backends additionally implementing `KernelBackend` (device-memory
 * kernels, e.g. `WebGpuBackend`) or `HostAcceleratorBackend` (host-memory
 * SIMD kernels, e.g. `WasmBackend`) are picked up automatically by the
 * ndarray dispatch layer once registered. The CPU backend ships as the
 * always-available reference implementation.
 */
export interface Backend {
  /** Returns static information about this backend. */
  info(): BackendInfo;

  /**
   * Returns `true` if the backend supports the given capability.
   *
   * @param cap - Capability to check
   */
  supports(cap: BackendCapability): boolean;

  /**
   * Perform any async initialisation the backend requires.
   *
   * For the CPU backend this is a no-op that resolves immediately.
   * GPU/WASM backends may need to request adapters or compile modules.
   */
  init(): Promise<void>;

  /**
   * Release any resources held by the backend.
   *
   * Safe to call multiple times.
   */
  dispose(): void;
}

/**
 * CPU Backend — the reference execution backend for Deepbox.
 *
 * This backend is always available and provides the baseline
 * implementation for all tensor operations. It executes
 * synchronously on the host CPU using standard TypedArrays.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

import type { Backend, BackendCapability, BackendInfo } from "./Backend";

const CPU_CAPABILITIES: readonly BackendCapability[] = [
  "matmul",
  "conv2d",
  "fft",
  "reduction",
  "elementwise",
  "blas",
  "random",
];

/**
 * The built-in CPU execution backend.
 *
 * Supports all capabilities and requires no async initialisation.
 */
export class CpuBackend implements Backend {
  private disposed = false;

  info(): BackendInfo {
    return {
      device: "cpu",
      name: "Deepbox CPU Backend",
      available: !this.disposed,
      capabilities: CPU_CAPABILITIES,
    };
  }

  supports(cap: BackendCapability): boolean {
    for (const c of CPU_CAPABILITIES) {
      if (c === cap) return true;
    }
    return false;
  }

  async init(): Promise<void> {
    // CPU backend needs no async setup.
  }

  dispose(): void {
    this.disposed = true;
  }

  /** Returns `true` after {@link dispose} has been called. */
  get isDisposed(): boolean {
    return this.disposed;
  }
}

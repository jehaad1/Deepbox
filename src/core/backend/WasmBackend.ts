/**
 * WASM SIMD Backend — host-accelerator execution backend for Deepbox.
 *
 * Runs precompiled WebAssembly SIMD kernels (4-lane f32) over a shared
 * WebAssembly memory. Binaries are generated at build time from the WAT
 * sources by `scripts/build-wasm.mjs` and embedded in the package, so
 * `init()` compiles them directly — no WAT compiler needed at runtime.
 *
 * Unlike the WebGPU backend, `wasm` is a **host accelerator**: tensors on
 * the `wasm` device keep ordinary TypedArray storage (zero-copy with the
 * CPU), and eligible ops (contiguous float32 element-wise arithmetic) run
 * through the SIMD kernels while everything else falls back to the normal
 * CPU implementation with identical IEEE-754 results.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

import type { Backend, BackendCapability, BackendInfo } from "./Backend";
import type { HostBinaryOp } from "./kernels";
import { WASM_BINARIES, WAT_MODULES, type WasmModuleName } from "./wasm_modules.generated";

export { WAT_MODULES, type WasmModuleName };

/**
 * A compiled WASM SIMD module with its exports.
 */
export type CompiledWasmModule = {
  readonly name: WasmModuleName;
  readonly instance: WebAssembly.Instance;
  readonly memory: WebAssembly.Memory;
};

const BINARY_MODULE: Record<HostBinaryOp, WasmModuleName> = {
  add: "simdAdd",
  sub: "simdSub",
  mul: "simdMul",
  div: "simdDiv",
};

type BinaryRun = (a: number, b: number, out: number, count: number) => void;
type DotRun = (a: number, b: number, count: number) => number;
type SumRun = (a: number, count: number) => number;

function base64ToBytes(base64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    return new Uint8Array(Buffer.from(base64, "base64"));
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * WASM SIMD execution backend.
 *
 * `init()` feature-detects WASM SIMD and instantiates the embedded kernel
 * binaries over one shared memory; afterwards `info().available` is `true`
 * and registering the backend (`registerBackend('wasm', backend)`) lets
 * contiguous float32 arithmetic on `wasm`-device tensors run through the
 * SIMD kernels.
 *
 * @example
 * ```ts
 * import { WasmBackend, registerBackend } from 'deepbox/core';
 * import { tensor, add } from 'deepbox/ndarray';
 *
 * const wasm = new WasmBackend();
 * await wasm.init();
 * if (wasm.info().available) {
 *   registerBackend('wasm', wasm);
 *   const a = tensor([1, 2, 3, 4], { device: 'wasm' });
 *   const b = add(a, a); // eligible ops use WASM SIMD
 * }
 * ```
 */
export class WasmBackend implements Backend {
  private modules = new Map<WasmModuleName, CompiledWasmModule>();
  private memory: WebAssembly.Memory | null = null;
  private disposed = false;
  private initialized = false;
  private initPromise: Promise<void> | null = null;

  private static readonly CAPABILITIES: readonly BackendCapability[] = [
    "elementwise",
    "reduction",
    "blas",
  ];

  info(): BackendInfo {
    return {
      device: "wasm",
      name: "Deepbox WASM SIMD Backend",
      available: this.initialized && this.modules.size > 0 && !this.disposed,
      capabilities: WasmBackend.CAPABILITIES,
    };
  }

  supports(cap: BackendCapability): boolean {
    for (const c of WasmBackend.CAPABILITIES) {
      if (c === cap) return true;
    }
    return false;
  }

  /**
   * Compile and instantiate the embedded SIMD kernel modules.
   *
   * When the runtime does not support WASM SIMD this completes without
   * error and {@link WasmBackend.info} keeps reporting `available: false`.
   */
  async init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInit();
    return this.initPromise;
  }

  private async doInit(): Promise<void> {
    if (!this.isWasmSimdAvailable()) return;

    try {
      const memory = new WebAssembly.Memory({ initial: 16, maximum: 16384 });
      const imports = { env: { memory } };
      for (const name of Object.keys(WASM_BINARIES) as WasmModuleName[]) {
        const bytes = base64ToBytes(WASM_BINARIES[name]);
        const module = await WebAssembly.compile(bytes);
        const instance = await WebAssembly.instantiate(module, imports);
        this.modules.set(name, { name, instance, memory });
      }
      this.memory = memory;
      this.initialized = true;
    } catch {
      this.modules.clear();
      this.memory = null;
      this.initialized = false;
    }
  }

  // ─── SIMD kernel surface ───────────────────────────────────────────────────

  /**
   * Element-wise `a OP b` over contiguous float32 data using SIMD.
   *
   * Per-element IEEE-754 arithmetic — results are bit-identical to the
   * scalar CPU implementation. Returns `null` when the backend is not
   * initialized, letting callers fall back to the CPU path.
   *
   * @param op - One of `add`, `sub`, `mul`, `div`
   * @param a - Left operand (length defines the element count)
   * @param b - Right operand (same length)
   * @returns The result array, or `null` if the kernels are unavailable
   */
  binaryContiguous(op: HostBinaryOp, a: Float32Array, b: Float32Array): Float32Array | null {
    const moduleName = BINARY_MODULE[op];
    const mod = moduleName ? this.modules.get(moduleName) : undefined;
    const memory = this.memory;
    if (!mod || !memory || this.disposed || a.length !== b.length) return null;

    const n = a.length;
    const bytes = n * 4;
    this.ensureCapacity(3 * bytes + 48);

    const aPtr = 0;
    const bPtr = this.align16(bytes);
    const outPtr = this.align16(bPtr + bytes);

    const heap = new Float32Array(memory.buffer);
    heap.set(a, aPtr / 4);
    heap.set(b, bPtr / 4);

    (mod.instance.exports["run"] as BinaryRun)(aPtr, bPtr, outPtr, n);

    // memory.buffer may have been detached by growth inside ensureCapacity,
    // so re-view before reading out.
    return new Float32Array(memory.buffer, outPtr, n).slice();
  }

  /**
   * SIMD dot product over contiguous float32 data.
   *
   * Accumulates in four f32 lanes, so the result can differ from a
   * sequential sum in the last bits — callers that need exact
   * CPU-sequential semantics should not use this.
   *
   * @returns The dot product, or `null` if the kernels are unavailable
   */
  dotContiguous(a: Float32Array, b: Float32Array): number | null {
    const mod = this.modules.get("simdDot");
    const memory = this.memory;
    if (!mod || !memory || this.disposed || a.length !== b.length) return null;

    const bytes = a.length * 4;
    this.ensureCapacity(2 * bytes + 32);
    const aPtr = 0;
    const bPtr = this.align16(bytes);
    const heap = new Float32Array(memory.buffer);
    heap.set(a, aPtr / 4);
    heap.set(b, bPtr / 4);
    return (mod.instance.exports["run"] as DotRun)(aPtr, bPtr, a.length);
  }

  /**
   * SIMD sum over contiguous float32 data (same accumulation caveats as
   * {@link WasmBackend.dotContiguous}).
   *
   * @returns The sum, or `null` if the kernels are unavailable
   */
  sumContiguous(a: Float32Array): number | null {
    const mod = this.modules.get("simdSum");
    const memory = this.memory;
    if (!mod || !memory || this.disposed) return null;

    this.ensureCapacity(a.length * 4 + 16);
    new Float32Array(memory.buffer).set(a, 0);
    return (mod.instance.exports["run"] as SumRun)(0, a.length);
  }

  private align16(offset: number): number {
    return (offset + 15) & ~15;
  }

  private ensureCapacity(bytes: number): void {
    const memory = this.memory;
    if (!memory) return;
    const current = memory.buffer.byteLength;
    if (current >= bytes) return;
    const pagesNeeded = Math.ceil((bytes - current) / 65536);
    memory.grow(pagesNeeded);
  }

  // ─── Introspection ─────────────────────────────────────────────────────────

  /**
   * Check if a specific WASM module is compiled and ready.
   *
   * @param name - Module name
   * @returns True if the module is compiled
   */
  hasModule(name: WasmModuleName): boolean {
    return this.modules.has(name);
  }

  /**
   * Get a compiled WASM module by name.
   *
   * @param name - Module name
   * @returns The compiled module, or undefined if not available
   */
  getModule(name: WasmModuleName): CompiledWasmModule | undefined {
    return this.modules.get(name);
  }

  /**
   * List all available WAT module names.
   */
  listModules(): WasmModuleName[] {
    return Object.keys(WAT_MODULES) as WasmModuleName[];
  }

  /**
   * Check if the current runtime supports WASM with SIMD.
   */
  isSimdSupported(): boolean {
    return this.isWasmSimdAvailable();
  }

  /**
   * Whether the backend has been successfully initialized.
   */
  get isInitialized(): boolean {
    return this.initialized;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.modules.clear();
    this.memory = null;
    this.initialized = false;
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  private isWasmSimdAvailable(): boolean {
    if (typeof WebAssembly === "undefined") return false;
    try {
      // The most reliable feature probe: validate one of the real SIMD
      // kernels we are about to instantiate.
      return WebAssembly.validate(base64ToBytes(WASM_BINARIES.simdAdd));
    } catch {
      return false;
    }
  }
}

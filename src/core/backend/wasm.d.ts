/**
 * Minimal WebAssembly type declarations for the Deepbox WASM backend.
 *
 * The Deepbox tsconfig targets ES2024 without DOM lib, so WebAssembly
 * globals need to be declared explicitly.
 */

declare namespace WebAssembly {
  function validate(bytes: ArrayBuffer | Uint8Array): boolean;
  function compile(bytes: ArrayBuffer | Uint8Array): Promise<Module>;
  function instantiate(
    module: Module,
    importObject?: Record<string, Record<string, unknown>>
  ): Promise<Instance>;

  class Module {
    constructor(bytes: ArrayBuffer | Uint8Array);
  }

  class Instance {
    readonly exports: Record<string, unknown>;
    constructor(module: Module, importObject?: Record<string, Record<string, unknown>>);
  }

  class Memory {
    readonly buffer: ArrayBuffer;
    constructor(descriptor: { initial: number; maximum?: number; shared?: boolean });
    grow(pages: number): number;
  }
}

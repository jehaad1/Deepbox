/**
 * Minimal WebAssembly type declarations for the Deepbox WASM backend.
 *
 * These cover only the calls the backend makes, so it type-checks in builds
 * whose `lib` setting has no WebAssembly typings (for example a Node-only
 * configuration without the DOM lib). When the DOM lib is present its fuller
 * declarations merge with these.
 *
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
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

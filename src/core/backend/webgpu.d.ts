/**
 * Ambient declarations for the WebGPU runtime constants.
 *
 * The backend itself uses the module-level types and constants in
 * `webgpu_types.ts`, which are shipped with the package. This file only
 * declares the two runtime globals (`GPUBufferUsage`, `GPUMapMode`) that the
 * project's own tests read after installing a Node WebGPU binding. It is not
 * part of the published declarations.
 *
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly INDEX: number;
  readonly VERTEX: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
  readonly INDIRECT: number;
  readonly QUERY_RESOLVE: number;
};

declare const GPUMapMode: {
  readonly READ: number;
  readonly WRITE: number;
};

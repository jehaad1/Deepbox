/**
 * Minimal structural WebGPU types used by the Deepbox WebGPU backend.
 *
 * These interfaces describe only the part of the WebGPU API that
 * `WebGpuBackend` calls. They are ordinary module exports (not ambient
 * globals), so they ship in the published declaration files and never clash
 * with the DOM lib or `@webgpu/types`. Real `GPU` objects from a browser, or
 * from a Node binding such as Dawn, are accepted as {@link GpuLike}; the
 * adapter and device they produce are used through the interfaces below.
 *
 * @module core/backend/webgpu_types
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

/** Opaque handle to a GPU object that the backend only passes back to the device. */
export interface GpuObject {
  readonly label?: string;
}

/**
 * Entry point of a WebGPU implementation (`navigator.gpu` or a Node binding).
 *
 * The adapter is typed as `object` on purpose: the real `GPUAdapter` type
 * from `@webgpu/types` is nominal (its handles carry brands), so no
 * structural interface can accept every implementation. The backend narrows
 * the adapter to {@link GpuAdapter} once, at initialization.
 */
export interface GpuLike {
  requestAdapter(options?: GpuRequestAdapterOptions): Promise<object | null>;
}

export interface GpuRequestAdapterOptions {
  powerPreference?: "low-power" | "high-performance";
}

/** The adapter limits that the backend reads. */
export interface GpuSupportedLimits {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
}

export interface GpuAdapter {
  requestDevice(descriptor?: GpuDeviceDescriptor): Promise<GpuDevice>;
  readonly features: ReadonlySet<string>;
  readonly limits: GpuSupportedLimits;
}

export interface GpuDeviceDescriptor {
  label?: string;
  /** Only the half-precision feature is ever requested. */
  requiredFeatures?: "shader-f16"[];
  requiredLimits?: Record<string, number>;
}

export interface GpuDevice {
  createShaderModule(descriptor: GpuShaderModuleDescriptor): GpuShaderModule;
  createComputePipeline(descriptor: GpuComputePipelineDescriptor): GpuComputePipeline;
  createBuffer(descriptor: GpuBufferDescriptor): GpuBuffer;
  createBindGroup(descriptor: GpuBindGroupDescriptor): GpuBindGroup;
  createCommandEncoder(): GpuCommandEncoder;
  readonly queue: GpuQueue;
  readonly features: ReadonlySet<string>;
  /** Resolves when the device is lost (destroyed or failed). */
  readonly lost: Promise<GpuDeviceLostInfo>;
  destroy(): void;
}

export interface GpuDeviceLostInfo {
  readonly reason: "unknown" | "destroyed";
  readonly message: string;
}

export interface GpuShaderModuleDescriptor {
  code: string;
  label?: string;
}

export type GpuShaderModule = GpuObject;

export interface GpuComputePipelineDescriptor {
  layout: "auto" | GpuPipelineLayout;
  compute: GpuProgrammableStage;
}

export type GpuPipelineLayout = GpuObject;

export interface GpuProgrammableStage {
  module: GpuShaderModule;
  entryPoint: string;
}

export interface GpuComputePipeline {
  getBindGroupLayout(index: number): GpuBindGroupLayout;
}

export type GpuBindGroupLayout = GpuObject;

export interface GpuBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
  label?: string;
}

export interface GpuBuffer {
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  unmap(): void;
  destroy(): void;
  readonly size: number;
}

export interface GpuBindGroupDescriptor {
  layout: GpuBindGroupLayout;
  entries: GpuBindGroupEntry[];
}

export interface GpuBindGroupEntry {
  binding: number;
  resource: GpuBufferBinding;
}

export interface GpuBufferBinding {
  buffer: GpuBuffer;
  offset?: number;
  size?: number;
}

export type GpuBindGroup = GpuObject;

export interface GpuCommandEncoder {
  beginComputePass(): GpuComputePassEncoder;
  copyBufferToBuffer(
    source: GpuBuffer,
    sourceOffset: number,
    destination: GpuBuffer,
    destinationOffset: number,
    size: number
  ): void;
  finish(): GpuCommandBuffer;
}

export interface GpuComputePassEncoder {
  setPipeline(pipeline: GpuComputePipeline): void;
  setBindGroup(index: number, bindGroup: GpuBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export type GpuCommandBuffer = GpuObject;

export interface GpuQueue {
  submit(commandBuffers: GpuCommandBuffer[]): void;
  /**
   * Write host data into a buffer. For typed-array sources `dataOffset` and
   * `size` count elements; for an `ArrayBuffer` they count bytes.
   */
  writeBuffer(
    buffer: GpuBuffer,
    offset: number,
    data: ArrayBuffer | ArrayBufferView,
    dataOffset?: number,
    size?: number
  ): void;
  /** Resolves once all work submitted so far has completed on the GPU. */
  onSubmittedWorkDone(): Promise<void>;
}

/** WebGPU buffer usage flags (`GPUBufferUsage`), fixed by the specification. */
export const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

/** WebGPU buffer map modes (`GPUMapMode`), fixed by the specification. */
export const GPU_MAP_MODE = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;

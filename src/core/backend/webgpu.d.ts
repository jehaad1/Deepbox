/**
 * Minimal WebGPU type declarations for the Deepbox WebGPU backend.
 *
 * These ambient declarations allow the WebGPU backend to compile
 * without depending on `@webgpu/types` or the DOM lib. The types
 * are only resolved at runtime in environments where WebGPU is
 * actually available (modern browsers with WebGPU support).
 */

declare interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

declare interface GPURequestAdapterOptions {
  powerPreference?: "low-power" | "high-performance";
}

declare interface GPUAdapter {
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
  readonly features: ReadonlySet<string>;
  readonly limits: Record<string, number>;
}

declare interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

declare interface GPUDevice {
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  readonly queue: GPUQueue;
  destroy(): void;
}

declare interface GPUShaderModuleDescriptor {
  code: string;
  label?: string;
}

declare type GPUShaderModule = { readonly __brand: "GPUShaderModule" };

declare interface GPUComputePipelineDescriptor {
  layout: "auto" | GPUPipelineLayout;
  compute: GPUProgrammableStage;
}

declare type GPUPipelineLayout = { readonly __brand: "GPUPipelineLayout" };

declare interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint: string;
}

declare interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

declare type GPUBindGroupLayout = { readonly __brand: "GPUBindGroupLayout" };

declare interface GPUBufferDescriptor {
  size: number;
  usage: GPUBufferUsageFlags;
  mappedAtCreation?: boolean;
  label?: string;
}

declare type GPUBufferUsageFlags = number;

declare interface GPUBuffer {
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  unmap(): void;
  destroy(): void;
  readonly size: number;
}

declare type GPUMapModeFlags = number;

declare interface GPUBindGroupDescriptor {
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

declare interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

declare type GPUBindingResource = GPUBufferBinding;

declare interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

declare type GPUBindGroup = { readonly __brand: "GPUBindGroup" };

declare interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number
  ): void;
  finish(): GPUCommandBuffer;
}

declare interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

declare type GPUCommandBuffer = { readonly __brand: "GPUCommandBuffer" };

declare interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBuffer): void;
}

declare const GPUBufferUsage: {
  readonly MAP_READ: GPUBufferUsageFlags;
  readonly MAP_WRITE: GPUBufferUsageFlags;
  readonly COPY_SRC: GPUBufferUsageFlags;
  readonly COPY_DST: GPUBufferUsageFlags;
  readonly INDEX: GPUBufferUsageFlags;
  readonly VERTEX: GPUBufferUsageFlags;
  readonly UNIFORM: GPUBufferUsageFlags;
  readonly STORAGE: GPUBufferUsageFlags;
  readonly INDIRECT: GPUBufferUsageFlags;
  readonly QUERY_RESOLVE: GPUBufferUsageFlags;
};

declare const GPUMapMode: {
  readonly READ: GPUMapModeFlags;
  readonly WRITE: GPUMapModeFlags;
};

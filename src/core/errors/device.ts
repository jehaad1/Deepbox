import { DeepboxError } from "./base";

/**
 * Error thrown when an operation is requested on an unsupported or unavailable compute device.
 *
 * This is intended for device selection / dispatch layers (e.g. attempting to use `webgpu`
 * in an environment where WebGPU is not available).
 */
export class DeviceError extends DeepboxError {
  /**
   * Discriminator name for this error type.
   */
  override name = "DeviceError";
}

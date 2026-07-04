/**
 * Backend registry — manages the set of available execution backends.
 *
 * The CPU backend is registered automatically at module load time.
 * Future backends (WebGPU, WASM) can be registered via
 * {@link registerBackend}.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

import { DeviceError } from "../errors/index";
import type { Device } from "../types/device";
import type { Backend } from "./Backend";
import { CpuBackend } from "./CpuBackend";
import {
  type HostAcceleratorBackend,
  isHostAcceleratorBackend,
  isKernelBackend,
  type KernelBackend,
} from "./kernels";

const backends = new Map<Device, Backend>();

// Register the CPU backend eagerly — it is always available.
backends.set("cpu", new CpuBackend());

/**
 * Retrieve the backend for the given device.
 *
 * @param device - Target device
 * @returns The registered backend
 * @throws {DeviceError} If no backend is registered for the device
 *
 * @example
 * ```ts
 * import { getBackend } from 'deepbox/core';
 *
 * const cpu = getBackend('cpu');
 * console.log(cpu.info().name); // "Deepbox CPU Backend"
 * ```
 */
export function getBackend(device: Device): Backend {
  const backend = backends.get(device);
  if (!backend) {
    throw new DeviceError(
      `No backend registered for device "${device}". ` +
        `Only the following devices have backends: ${[...backends.keys()].join(", ")}`
    );
  }
  return backend;
}

/**
 * Register a new backend for a device.
 *
 * If a backend was previously registered for the same device it is
 * replaced (the old backend is **not** automatically disposed).
 *
 * @param device - Device the backend serves
 * @param backend - Backend implementation
 *
 * @example
 * ```ts
 * import { registerBackend } from 'deepbox/core';
 *
 * registerBackend('wasm', myWasmBackend);
 * ```
 */
export function registerBackend(device: Device, backend: Backend): void {
  backends.set(device, backend);
}

function formatRegisteredDevices(): string {
  const devices = [...backends.keys()];
  return devices.length > 0 ? devices.join(", ") : "none";
}

/**
 * Ensure a device has a registered backend that is currently usable.
 *
 * @param device - Device to validate
 * @param subject - Human-readable subject for the error message
 * @returns The validated device
 * @throws {DeviceError} If no backend is registered or the backend is unavailable
 *
 * @internal
 */
export function ensureBackendAvailable(device: Device, subject = "device"): Device {
  const backend = backends.get(device);
  if (!backend) {
    throw new DeviceError(
      `No backend registered for ${subject} "${device}". ` +
        `Registered backends: ${formatRegisteredDevices()}`
    );
  }

  const info = backend.info();
  if (!info.available) {
    throw new DeviceError(
      `Backend "${info.name}" for ${subject} "${device}" is registered but not currently available.`
    );
  }

  return device;
}

/**
 * Check whether a backend is registered and available for the device.
 *
 * @param device - Device to check
 * @returns `true` if a backend is registered and reports itself as available
 */
export function isBackendAvailable(device: Device): boolean {
  const backend = backends.get(device);
  if (!backend) return false;
  return backend.info().available;
}

/**
 * Remove a registered backend.
 *
 * The backend is **not** disposed — call `backend.dispose()` yourself if it
 * holds resources. The CPU backend is mandatory and cannot be unregistered.
 *
 * @param device - Device whose backend should be removed
 * @returns `true` if a backend was removed, `false` if none was registered
 * @throws {DeviceError} When attempting to unregister the CPU backend
 */
export function unregisterBackend(device: Device): boolean {
  if (device === "cpu") {
    throw new DeviceError("The CPU backend is mandatory and cannot be unregistered");
  }
  return backends.delete(device);
}

/**
 * List all currently registered devices with backends.
 *
 * @returns Array of device identifiers
 */
export function listBackends(): Device[] {
  return [...backends.keys()];
}

/**
 * Retrieve the kernel-capable backend for a device, or `null` when the
 * device has no registered backend, the backend is unavailable, or it does
 * not implement the {@link KernelBackend} execution surface.
 *
 * Used by the ndarray dispatch layer to route ops on non-CPU tensors.
 *
 * @param device - Target device
 * @returns The kernel backend, or `null`
 */
export function getKernelBackend(device: Device): KernelBackend | null {
  const backend = backends.get(device);
  if (!backend) return null;
  if (!isKernelBackend(backend)) return null;
  if (!backend.info().available) return null;
  return backend;
}

/**
 * Retrieve the host-accelerator backend for a device, or `null` when the
 * device has no registered, available backend implementing that surface.
 *
 * @param device - Target device
 * @returns The host-accelerator backend, or `null`
 */
export function getHostAccelerator(device: Device): HostAcceleratorBackend | null {
  const backend = backends.get(device);
  if (!backend) return null;
  if (!isHostAcceleratorBackend(backend)) return null;
  if (!backend.info().available) return null;
  return backend;
}

/**
 * Retrieve the kernel-capable backend for a device, throwing a descriptive
 * {@link DeviceError} when unavailable.
 *
 * @param device - Target device
 * @param subject - Human-readable subject for the error message
 * @returns The kernel backend
 * @throws {DeviceError} If no available kernel backend serves the device
 *
 * @internal
 */
export function requireKernelBackend(device: Device, subject = "operation"): KernelBackend {
  const backend = backends.get(device);
  if (!backend) {
    throw new DeviceError(
      `No backend registered for device "${device}" (required by ${subject}). ` +
        `Registered backends: ${formatRegisteredDevices()}`
    );
  }
  if (!isKernelBackend(backend)) {
    throw new DeviceError(
      `Backend "${backend.info().name}" for device "${device}" does not implement ` +
        `device kernels (required by ${subject}).`
    );
  }
  if (!backend.info().available) {
    throw new DeviceError(
      `Backend "${backend.info().name}" for device "${device}" is registered but not ` +
        `currently available (required by ${subject}). Did you await backend.init()?`
    );
  }
  return backend;
}

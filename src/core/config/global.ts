/**
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

import { __clearSeed, __getSeed, __setSeed } from "../../random/random";
import { ensureBackendAvailable } from "../backend/registry";
import { DataValidationError } from "../errors/validation";
import type { Device } from "../types/device";
import type { DType } from "../types/dtype";
import { validateDevice, validateDtype, validateInteger } from "../utils/validation";

/**
 * Global configuration for Deepbox.
 *
 * @property defaultDtype - Default data type for new tensors
 * @property defaultDevice - Default compute device
 * @property seed - Random seed for reproducibility (null = not set). Mirrors the
 *   state of the global random generator, so a seed set through `deepbox/random`
 *   is reported here too.
 */
export type DeepboxConfig = {
  readonly defaultDtype: DType;
  readonly defaultDevice: Device;
  readonly seed: number | null;
};

const DEFAULT_CONFIG: DeepboxConfig = {
  defaultDtype: "float32",
  defaultDevice: "cpu",
  seed: null,
};

type ConfigKey = keyof DeepboxConfig;

/**
 * Allowed config keys (used for defensive validation of user-provided config objects).
 *
 * Kept as a readonly array for stable iteration and clear error messaging.
 */
const CONFIG_KEYS: readonly ConfigKey[] = ["defaultDtype", "defaultDevice", "seed"];

let config: DeepboxConfig = { ...DEFAULT_CONFIG };

/**
 * The global random generator owns the seed. Reading it from there keeps
 * `getSeed()` / `getConfig().seed` correct when the seed was set or cleared
 * through `deepbox/random` instead of this module.
 */
function currentSeed(): number | null {
  return __getSeed() ?? null;
}

function isPlainConfigObject(value: object): boolean {
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasOwnConfigKey(source: Partial<DeepboxConfig>, key: ConfigKey): boolean {
  return Object.hasOwn(source, key);
}

function normalizeSeed(value: unknown, name: string, allowNull: true): number | null;
function normalizeSeed(value: unknown, name: string, allowNull: false): number;
function normalizeSeed(value: unknown, name: string, allowNull: boolean): number | null {
  // Accept explicit null only when allowed by the caller (seed in global config supports null).
  if (value === null) {
    if (allowNull) {
      return null;
    }
    throw new DataValidationError(`${name} must be a safe integer; received null`);
  }

  // Reject non-numeric values early with a descriptive message.
  if (typeof value !== "number") {
    throw new DataValidationError(
      `${name} must be a safe integer${allowNull ? " or null" : ""}; received ${String(value)}`
    );
  }

  // validateInteger enforces finite, integer, and safe-integer constraints.
  validateInteger(value, name);
  return value;
}

function normalizeConfiguredDevice(value: unknown, name: string): Device {
  const normalized = validateDevice(value, name);
  return ensureBackendAvailable(normalized, name);
}

/**
 * Get the current global configuration.
 *
 * Returns a copy of the configuration to prevent external mutation.
 *
 * @returns Current configuration object
 *
 * @example
 * ```ts
 * import { getConfig } from 'deepbox/core';
 *
 * const config = getConfig();
 * console.log(config.defaultDtype);  // 'float32'
 * ```
 */
export function getConfig(): Readonly<DeepboxConfig> {
  return { ...config, seed: currentSeed() };
}

/**
 * Update global configuration.
 *
 * Merges provided settings with current configuration.
 * Only specified fields are updated. All values are validated before any of
 * them is applied, so a failing call leaves the configuration unchanged. The
 * random generator is re-seeded only when `seed` is part of the update;
 * changing the dtype or device does not disturb a running random stream.
 *
 * @param next - Partial configuration to merge
 * @throws {DataValidationError} If config is invalid or contains unknown keys
 *
 * @example
 * ```ts
 * import { setConfig } from 'deepbox/core';
 *
 * setConfig({
 *   defaultDtype: 'float64',
 *   seed: 42
 * });
 * ```
 */
export function setConfig(next: Partial<DeepboxConfig>): void {
  // Validate the incoming value is an object (not null / array).
  if (next === null || typeof next !== "object" || Array.isArray(next)) {
    throw new DataValidationError(`config must be an object with keys [${CONFIG_KEYS.join(", ")}]`);
  }

  // Reject non-plain objects (e.g., class instances) to avoid prototype surprises.
  if (!isPlainConfigObject(next)) {
    throw new DataValidationError(
      `config must be a plain object with keys [${CONFIG_KEYS.join(", ")}]`
    );
  }

  // Find unsupported keys explicitly (defensive API design).
  const keys = Object.keys(next);
  const unknownKeys: string[] = [];
  for (const key of keys) {
    let isKnown = false;
    for (const allowed of CONFIG_KEYS) {
      if (allowed === key) {
        isKnown = true;
        break;
      }
    }
    if (!isKnown) {
      unknownKeys.push(key);
    }
  }

  // Fail fast with an actionable message listing allowed keys.
  if (unknownKeys.length > 0) {
    throw new DataValidationError(
      `config contains unsupported keys: ${unknownKeys.join(", ")}. Allowed keys are [${CONFIG_KEYS.join(
        ", "
      )}]`
    );
  }

  // Apply validated updates field-by-field (only if explicitly provided).
  const nextDefaultDtype = hasOwnConfigKey(next, "defaultDtype")
    ? validateDtype(next.defaultDtype, "defaultDtype")
    : config.defaultDtype;

  const nextDefaultDevice = hasOwnConfigKey(next, "defaultDevice")
    ? normalizeConfiguredDevice(next.defaultDevice, "defaultDevice")
    : config.defaultDevice;

  const seedProvided = hasOwnConfigKey(next, "seed");
  const nextSeed = seedProvided ? normalizeSeed(next.seed, "seed", true) : null;

  // Commit the new config snapshot.
  config = {
    defaultDtype: nextDefaultDtype,
    defaultDevice: nextDefaultDevice,
    seed: seedProvided ? nextSeed : currentSeed(),
  };

  // Touch the random generator only when the caller asked to change the seed.
  if (seedProvided) {
    if (nextSeed !== null) {
      __setSeed(nextSeed);
    } else {
      __clearSeed();
    }
  }
}

/**
 * Reset configuration to default values.
 *
 * Also clears the global random seed.
 *
 * @example
 * ```ts
 * import { resetConfig } from 'deepbox/core';
 *
 * resetConfig();  // Back to defaults
 * ```
 */
export function resetConfig(): void {
  config = { ...DEFAULT_CONFIG };
  __clearSeed();
}

/**
 * Set the global random seed for reproducibility.
 *
 * @param seed - Integer seed value (negative values and zero are allowed)
 * @throws {DataValidationError} If seed is not a safe integer
 *
 * @example
 * ```ts
 * import { setSeed } from 'deepbox/core';
 *
 * setSeed(42);  // All random operations now reproducible
 * ```
 */
export function setSeed(seed: number): void {
  const normalized = normalizeSeed(seed, "seed", false);
  config = { ...config, seed: normalized };
  __setSeed(normalized);
}

/**
 * Get the current random seed.
 *
 * @returns Current seed value or null if not set
 */
export function getSeed(): number | null {
  return currentSeed();
}

/**
 * Set the default compute device for new tensors and other device-aware APIs.
 *
 * @param device - Device to use ('cpu', 'webgpu', or 'wasm')
 * @throws {DataValidationError} If device is not a supported identifier
 * @throws {DeviceError} If no backend is registered for the device or it reports unavailable
 *
 * @example
 * ```ts
 * import { registerBackend, setDevice, WebGpuBackend } from 'deepbox/core';
 *
 * setDevice('cpu');  // Use CPU for all operations (default)
 *
 * // With a registered GPU backend, new tensors default to GPU memory:
 * const gpu = new WebGpuBackend();
 * await gpu.init();
 * if (gpu.info().available) {
 *   registerBackend('webgpu', gpu);
 *   setDevice('webgpu');
 * }
 * ```
 */
export function setDevice(device: Device): void {
  const normalized = normalizeConfiguredDevice(device, "device");
  config = { ...config, defaultDevice: normalized };
}

/**
 * Get the current default device.
 *
 * @returns Current default device
 */
export function getDevice(): Device {
  return config.defaultDevice;
}

/**
 * Set the default data type for new tensors.
 *
 * @param dtype - Data type to use as default
 * @throws {DataValidationError} If dtype is not supported
 *
 * @example
 * ```ts
 * import { setDtype } from 'deepbox/core';
 *
 * setDtype('float64');  // Use double precision by default
 * ```
 */
export function setDtype(dtype: DType): void {
  const normalized = validateDtype(dtype, "dtype");
  config = { ...config, defaultDtype: normalized };
}

/**
 * Get the current default data type.
 *
 * @returns Current default dtype
 */
export function getDtype(): DType {
  return config.defaultDtype;
}

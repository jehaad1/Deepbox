import { DataValidationError } from "../errors/validation";
import type { Device } from "../types/device";
import type { DType } from "../types/dtype";
import { validateDevice, validateDtype, validateInteger } from "../utils/validation";

/**
 * Global configuration for Deepbox.
 *
 * @property defaultDtype - Default data type for new tensors
 * @property defaultDevice - Default compute device
 * @property seed - Random seed for reproducibility (null = not set)
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
    throw new DataValidationError(`${name} must be a finite number; received ${String(value)}`);
  }

  // validateInteger enforces finite, integer, and safe-integer constraints.
  validateInteger(value, name);
  return value;
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
  return { ...config };
}

/**
 * Update global configuration.
 *
 * Merges provided settings with current configuration.
 * Only specified fields are updated.
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
    ? validateDevice(next.defaultDevice, "defaultDevice")
    : config.defaultDevice;

  const nextSeed = hasOwnConfigKey(next, "seed")
    ? normalizeSeed(next.seed, "seed", true)
    : config.seed;

  // Commit the new immutable config snapshot.
  config = {
    defaultDtype: nextDefaultDtype,
    defaultDevice: nextDefaultDevice,
    seed: nextSeed,
  };
}

/**
 * Reset configuration to default values.
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
}

/**
 * Set the global random seed for reproducibility.
 *
 * @param seed - Integer seed value
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
}

/**
 * Get the current random seed.
 *
 * @returns Current seed value or null if not set
 */
export function getSeed(): number | null {
  return config.seed;
}

/**
 * Set the default compute device.
 *
 * @param device - Device to use ('cpu', 'webgpu', or 'wasm')
 * @throws {DataValidationError} If device is not supported
 *
 * @example
 * ```ts
 * import { setDevice } from 'deepbox/core';
 *
 * setDevice('cpu');  // Use CPU for all operations
 * ```
 */
export function setDevice(device: Device): void {
  const normalized = validateDevice(device, "device");
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

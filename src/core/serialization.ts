/**
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { DataValidationError, DeepboxError, InvalidParameterError } from "./errors/index";

/**
 * Serialized representation of a tensor: flat data in row-major order plus metadata.
 */
export type SerializedTensor = {
  readonly __type: "Tensor";
  readonly data: ReadonlyArray<number | string | bigint>;
  readonly shape: readonly number[];
  readonly dtype: string;
};

/**
 * Serialized nn.Module state: parameters and buffers.
 */
export type SerializedModuleState = {
  readonly __type: "ModuleState";
  readonly parameters: Record<
    string,
    { data: Array<number | string | bigint>; dtype: string; shape: number[] }
  >;
  readonly buffers: Record<
    string,
    { data: Array<number | string | bigint>; dtype: string; shape: number[] }
  >;
};

/**
 * Serialized ML estimator: hyperparameters and fitted arrays.
 */
export type SerializedEstimator = {
  readonly __type: "Estimator";
  readonly className: string;
  readonly params: Record<string, unknown>;
  readonly state: Record<string, unknown>;
};

/**
 * Union of all serializable payloads.
 */
export type SerializedPayload = SerializedTensor | SerializedModuleState | SerializedEstimator;

// ─── JSON replacer/reviver for BigInt and non-finite numbers ──────
//
// JSON has no representation for bigint, NaN, Infinity or negative zero:
// JSON.stringify turns the non-finite values into `null` and -0 into `0`,
// which would silently corrupt tensor data. They are encoded as tagged
// objects instead and restored by the reviver.

function jsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") {
    return { __bigint: value.toString() };
  }
  if (typeof value === "number") {
    if (Number.isNaN(value)) return { __number: "NaN" };
    if (value === Number.POSITIVE_INFINITY) return { __number: "Infinity" };
    if (value === Number.NEGATIVE_INFINITY) return { __number: "-Infinity" };
    if (Object.is(value, -0)) return { __number: "-0" };
  }
  return value;
}

function jsonReviver(_key: string, value: unknown): unknown {
  if (value !== null && typeof value === "object") {
    if ("__bigint" in value) {
      const bi = (value as Record<string, unknown>)["__bigint"];
      if (typeof bi === "string") {
        return BigInt(bi);
      }
    }
    if ("__number" in value) {
      switch ((value as Record<string, unknown>)["__number"]) {
        case "NaN":
          return Number.NaN;
        case "Infinity":
          return Number.POSITIVE_INFINITY;
        case "-Infinity":
          return Number.NEGATIVE_INFINITY;
        case "-0":
          return -0;
      }
    }
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validateTensorRecord(rec: unknown, where: string): void {
  if (!isRecord(rec)) {
    throw new DataValidationError(`Invalid serialized payload: ${where} must be an object`);
  }
  if (!Array.isArray(rec["data"])) {
    throw new DataValidationError(`Invalid serialized payload: ${where}.data must be an array`);
  }
  const shape = rec["shape"];
  if (!Array.isArray(shape) || shape.some((d) => !Number.isSafeInteger(d) || d < 0)) {
    throw new DataValidationError(
      `Invalid serialized payload: ${where}.shape must be an array of non-negative integers`
    );
  }
  const dtype = rec["dtype"];
  if (typeof dtype !== "string") {
    throw new DataValidationError(`Invalid serialized payload: ${where}.dtype must be a string`);
  }
  // Complex data may be stored as interleaved pairs, so only real dtypes are length-checked.
  if (dtype !== "complex64" && dtype !== "complex128") {
    let size = 1;
    for (const d of shape as number[]) size *= d;
    if (rec["data"].length !== size) {
      throw new DataValidationError(
        `Invalid serialized payload: ${where}.data has ${rec["data"].length} elements but shape [${shape.join(", ")}] holds ${size}`
      );
    }
  }
}

function validatePayload(obj: unknown): asserts obj is SerializedPayload {
  if (!isRecord(obj)) {
    throw new DataValidationError("Invalid serialized payload: expected an object");
  }
  const typ = obj["__type"];
  if (typ === "Tensor") {
    validateTensorRecord(obj, "tensor");
  } else if (typ === "ModuleState") {
    for (const group of ["parameters", "buffers"] as const) {
      const entries = obj[group];
      if (!isRecord(entries)) {
        throw new DataValidationError(
          `Invalid serialized payload: ModuleState.${group} must be an object`
        );
      }
      for (const [name, entry] of Object.entries(entries)) {
        validateTensorRecord(entry, `${group}['${name}']`);
      }
    }
  } else if (typ === "Estimator") {
    if (typeof obj["className"] !== "string") {
      throw new DataValidationError(
        "Invalid serialized payload: Estimator.className must be a string"
      );
    }
    if (!isRecord(obj["params"]) || !isRecord(obj["state"])) {
      throw new DataValidationError(
        "Invalid serialized payload: Estimator.params and Estimator.state must be objects"
      );
    }
  } else {
    throw new DataValidationError(`Invalid serialized payload: unknown __type '${String(typ)}'`);
  }
}

/**
 * Serialize a payload to a JSON string.
 *
 * BigInt values, NaN, Infinity, -Infinity and negative zero are encoded as
 * tagged objects so that `fromJSON(toJSON(x))` restores them exactly.
 *
 * @param payload - The object to serialize
 * @returns JSON string
 * @throws {DataValidationError} If `payload` is not a valid serialized payload
 */
export function toJSON(payload: SerializedPayload): string {
  validatePayload(payload);
  return JSON.stringify(payload, jsonReplacer, 2);
}

/**
 * Deserialize a JSON string back to a payload object.
 *
 * @param json - JSON string produced by `toJSON()`
 * @returns Deserialized payload
 * @throws {DataValidationError} If the string is not valid JSON or does not describe a
 *   Tensor, ModuleState or Estimator payload (for tensors, the number of data
 *   elements must equal the product of the shape unless the dtype is complex)
 */
export function fromJSON(json: string): SerializedPayload {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json, jsonReviver);
  } catch (err) {
    throw new DataValidationError(
      `Invalid serialized payload: ${err instanceof Error ? err.message : String(err)}`,
      { cause: err }
    );
  }
  validatePayload(parsed);
  return parsed;
}

// ─── File I/O (Node.js only) ─────────────────────────────────────

/**
 * Save a serialized payload to a file (Node.js only).
 *
 * Uses dynamic import of `node:fs` so the library remains
 * browser-compatible when file I/O is not used.
 *
 * @param path - File path to write to
 * @param payload - The payload to serialize and save
 * @throws {InvalidParameterError} If `path` is empty
 * @throws {DataValidationError} If `payload` is not a valid serialized payload
 * @throws {DeepboxError} If the file cannot be written or `node:fs` is unavailable
 *
 * @example
 * ```ts
 * import { save } from 'deepbox/core';
 * // Save a tensor
 * await save('./model.json', { __type: 'Tensor', data: [1,2,3], shape: [3], dtype: 'float32' });
 * ```
 */
export async function save(path: string, payload: SerializedPayload): Promise<void> {
  if (typeof path !== "string" || path.length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  const json = toJSON(payload);
  try {
    const fs = await importNodeFs();
    await fs.promises.writeFile(path, json, "utf-8");
  } catch (err) {
    if (err instanceof DeepboxError) throw err;
    throw new DeepboxError(
      `Failed to save to '${path}': ${err instanceof Error ? err.message : String(err)}`,
      { cause: err }
    );
  }
}

/**
 * Load a serialized payload from a file (Node.js only).
 *
 * @param path - File path to read from
 * @returns Deserialized payload
 * @throws {InvalidParameterError} If `path` is empty
 * @throws {DataValidationError} If the file content is not a valid serialized payload
 * @throws {DeepboxError} If the file cannot be read or `node:fs` is unavailable
 *
 * @example
 * ```ts
 * import { load } from 'deepbox/core';
 * const payload = await load('./model.json');
 * ```
 */
export async function load(path: string): Promise<SerializedPayload> {
  if (typeof path !== "string" || path.length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  try {
    const fs = await importNodeFs();
    const json = await fs.promises.readFile(path, "utf-8");
    return fromJSON(json);
  } catch (err) {
    if (err instanceof DeepboxError) throw err;
    throw new DeepboxError(
      `Failed to load from '${path}': ${err instanceof Error ? err.message : String(err)}`,
      { cause: err }
    );
  }
}

// ─── Internal: dynamic fs import ─────────────────────────────────

type FsLike = {
  promises: {
    writeFile(path: string, data: string, encoding: string): Promise<void>;
    readFile(path: string, encoding: string): Promise<string>;
  };
};

async function importNodeFs(): Promise<FsLike> {
  try {
    // Dynamic import to avoid bundler issues in browser builds
    const mod = await import("node:fs");
    return mod as unknown as FsLike;
  } catch {
    throw new DeepboxError(
      "File I/O is only available in Node.js. Use toJSON()/fromJSON() for in-memory serialization."
    );
  }
}

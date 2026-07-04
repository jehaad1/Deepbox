/**
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { DataValidationError, DeepboxError, InvalidParameterError } from "./errors/index";

/**
 * Serialized representation of a tensor — flat data + metadata.
 */
export type SerializedTensor = {
  readonly __type: "Tensor";
  readonly data: ReadonlyArray<number | string | bigint>;
  readonly shape: readonly number[];
  readonly dtype: string;
};

/**
 * Serialized nn.Module state — parameters + buffers.
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
 * Serialized ML estimator — hyperparams + fitted arrays.
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

// ─── JSON replacer/reviver for BigInt ─────────────────────────────

function jsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") {
    return { __bigint: value.toString() };
  }
  return value;
}

function jsonReviver(_key: string, value: unknown): unknown {
  if (value !== null && typeof value === "object" && "__bigint" in value) {
    const rec = value as Record<string, unknown>;
    const bi = rec["__bigint"];
    if (typeof bi === "string") {
      return BigInt(bi);
    }
  }
  return value;
}

/**
 * Serialize a payload to a JSON string.
 *
 * Handles BigInt values transparently.
 *
 * @param payload - The object to serialize
 * @returns JSON string
 */
export function toJSON(payload: SerializedPayload): string {
  return JSON.stringify(payload, jsonReplacer, 2);
}

/**
 * Deserialize a JSON string back to a payload object.
 *
 * @param json - JSON string produced by `toJSON()`
 * @returns Deserialized payload
 */
export function fromJSON(json: string): SerializedPayload {
  const parsed: unknown = JSON.parse(json, jsonReviver);
  if (parsed === null || typeof parsed !== "object") {
    throw new DataValidationError("Invalid serialized payload: expected an object");
  }
  const obj = parsed as Record<string, unknown>;
  const typ = obj["__type"];
  if (typ !== "Tensor" && typ !== "ModuleState" && typ !== "Estimator") {
    throw new DataValidationError(`Invalid serialized payload: unknown __type '${String(typ)}'`);
  }
  return parsed as SerializedPayload;
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
 *
 * @example
 * ```ts
 * import { save } from 'deepbox/core';
 * // Save a tensor
 * save('./model.json', { __type: 'Tensor', data: [1,2,3], shape: [3], dtype: 'float32' });
 * ```
 */
export async function save(path: string, payload: SerializedPayload): Promise<void> {
  if (typeof path !== "string" || path.length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  const json = toJSON(payload);
  try {
    const fs = await importNodeFs();
    fs.writeFileSync(path, json, "utf-8");
  } catch (err) {
    if (err instanceof DeepboxError) throw err;
    throw new DeepboxError(
      `Failed to save to '${path}': ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

/**
 * Load a serialized payload from a file (Node.js only).
 *
 * @param path - File path to read from
 * @returns Deserialized payload
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
    const json = fs.readFileSync(path, "utf-8");
    return fromJSON(json);
  } catch (err) {
    if (err instanceof DeepboxError) throw err;
    throw new DeepboxError(
      `Failed to load from '${path}': ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// ─── Internal: dynamic fs import ─────────────────────────────────

type FsLike = {
  writeFileSync(path: string, data: string, encoding: string): void;
  readFileSync(path: string, encoding: string): string;
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

/**
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { getConfig } from "../core/config";
import { DeepboxError, InvalidParameterError } from "../core/errors";
import type { DType } from "../core/types/dtype";
import { Generator } from "../random/Generator";
import { __fillNormal, __fillUniform, __random, __randomBelow } from "../random/random";

/**
 * Assert that an input is a positive integer.
 *
 * @internal
 */
export function assertPositiveInt(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new InvalidParameterError(
      `${name} must be a positive safe integer; received ${value}`,
      name,
      value
    );
  }
}

/**
 * Assert that a value is a boolean.
 *
 * @internal
 */
export function assertBoolean(name: string, value: unknown): void {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(
      `${name} must be a boolean; received ${String(value)}`,
      name,
      value
    );
  }
}

const FLOAT_DTYPES: readonly DType[] = ["float16", "bfloat16", "float32", "float64"];

/**
 * Floating-point dtype for dataset values built without an explicit dtype.
 *
 * Follows the global default dtype when it is a floating-point dtype. When the
 * global default is an integer, boolean or other non-float dtype (for example
 * after `setDtype("int32")`), it returns `float32`, so dataset values are never
 * truncated.
 *
 * @internal
 */
export function defaultFloatDtype(): DType {
  const configured = getConfig().defaultDtype;
  return FLOAT_DTYPES.includes(configured) ? configured : "float32";
}

/**
 * Normalize and validate an optional seed value.
 *
 * @internal
 */
export function normalizeOptionalSeed(name: string, value: number | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isSafeInteger(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite safe integer; received ${value}`,
      name,
      value
    );
  }
  return value;
}

/**
 * A dataset RNG: callable for a uniform [0,1) draw, plus bulk fills that draw
 * from the SAME underlying stream. `fillNormal` runs the fast Ziggurat sampler
 * (about 10x faster than the Box-Muller {@link fillNormal01}); both fills stay consistent with
 * interspersed `rng()` calls because they share one generator instance.
 *
 * @internal
 */
export interface DatasetRng {
  (): number;
  fillNormal(out: Float64Array, off: number, count: number): void;
  fillUniform(out: Float64Array, off: number, count: number): void;
}

/**
 * Check that `out[off .. off+count)` lies inside `out`, so the unseeded and seeded
 * fills reject an out-of-range window the same way instead of truncating silently.
 */
function assertFillWindow(out: Float64Array, off: number, count: number): void {
  if (
    !Number.isSafeInteger(off) ||
    !Number.isSafeInteger(count) ||
    off < 0 ||
    count < 0 ||
    off + count > out.length
  ) {
    throw new InvalidParameterError(
      `fill window [${off}, ${off + count}) does not fit a buffer of length ${out.length}`,
      "count",
      count
    );
  }
}

/**
 * Create a pseudo-random number generator.
 *
 * Uses xoshiro128++ (same as the global PRNG) for high-quality randomness.
 * If seed is undefined, falls back to the global seeded PRNG.
 *
 * @internal
 */
export function createRng(seed?: number): DatasetRng {
  if (seed === undefined) {
    const f = (() => __random()) as DatasetRng;
    f.fillNormal = (out, off, count) => {
      assertFillWindow(out, off, count);
      __fillNormal(out.subarray(off, off + count), count);
    };
    f.fillUniform = (out, off, count) => {
      assertFillWindow(out, off, count);
      __fillUniform(out.subarray(off, off + count), count);
    };
    return f;
  }
  const generator = new Generator(seed);
  const f = (() => generator.random()) as DatasetRng;
  f.fillNormal = (out, off, count) => generator.fillNormalInto(out, off, count);
  f.fillUniform = (out, off, count) => generator.fillUniformInto(out, off, count);
  return f;
}

/**
 * Return a function that supplies the RNG for each pass over a dataset.
 *
 * By default every call builds a fresh generator from `seed`, so a seeded
 * dataset repeats the same order on every pass. With `persist` set and a seed
 * given, all calls share one generator, so the first pass matches the default
 * and later passes continue the stream and differ from each other, while the
 * whole sequence stays reproducible. Without a seed, the global generator is
 * used and `persist` has no effect.
 *
 * @internal
 */
export function createPassRng(seed: number | undefined, persist: boolean): () => DatasetRng {
  if (!persist || seed === undefined) return () => createRng(seed);
  let shared: DatasetRng | undefined;
  return () => {
    shared ??= createRng(seed);
    return shared;
  };
}

/**
 * Sample from standard normal distribution N(0, 1) using Box-Muller transform.
 *
 * @internal
 */
export function normal01(rng: () => number): number {
  const u1 = Math.max(rng(), Number.EPSILON);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Fill `out[off .. off+count)` with standard-normal samples, using BOTH
 * outputs of each Box-Muller pair (the scalar {@link normal01} discards the
 * sine twin, paying log+sqrt+cos and two rng draws per sample; this pays
 * them per two samples, roughly half the transcendental and RNG cost).
 *
 * Deterministic for a given rng stream. Draws `2*ceil(count/2)` uniforms.
 *
 * @internal
 */
export function fillNormal01(
  out: Float64Array,
  off: number,
  count: number,
  rng: () => number
): void {
  const twoPi = 2 * Math.PI;
  let i = 0;
  for (; i + 2 <= count; i += 2) {
    const u1 = Math.max(rng(), Number.EPSILON);
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = twoPi * u2;
    out[off + i] = r * Math.cos(theta);
    out[off + i + 1] = r * Math.sin(theta);
  }
  if (i < count) {
    const u1 = Math.max(rng(), Number.EPSILON);
    const u2 = rng();
    out[off + i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(twoPi * u2);
  }
}

/**
 * Shuffle an array in-place using Fisher-Yates.
 *
 * Works for arrays of any element type, including arrays that contain
 * `undefined`. Consumes exactly one `rng()` draw per swap position.
 *
 * @internal
 */
export function shuffleInPlace<T>(array: T[], rng: () => number): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = __randomBelow(rng, i + 1);
    const tmp = array[i] as T;
    array[i] = array[j] as T;
    array[j] = tmp;
  }
}

/**
 * Shuffle two aligned arrays in-place using Fisher-Yates.
 *
 * Consumes the RNG exactly like {@link shuffleInPlace}, so both arrays end up
 * permuted identically.
 *
 * @internal
 */
export function shufflePairedInPlace<T, U>(left: T[], right: U[], rng: () => number): void {
  if (left.length !== right.length) {
    throw new DeepboxError(
      `Internal error: array length mismatch during shuffle (${left.length} vs ${right.length})`
    );
  }

  for (let i = left.length - 1; i > 0; i--) {
    const j = __randomBelow(rng, i + 1);
    const leftTmp = left[i] as T;
    left[i] = left[j] as T;
    left[j] = leftTmp;

    const rightTmp = right[i] as U;
    right[i] = right[j] as U;
    right[j] = rightTmp;
  }
}

/**
 * Fisher-Yates shuffle of the rows of a row-major flat matrix, paired with a
 * label array. Consumes the RNG in exactly the same order as
 * {@link shufflePairedInPlace} on nested arrays, so seeded output is
 * bit-identical, it just swaps `nCols`-wide row slabs in a typed buffer
 * instead of reordering an array of row references.
 *
 * @internal
 */
export function shuffleRowsPairedInPlace(
  x: Float64Array,
  y: Int32Array | Float64Array,
  nRows: number,
  nCols: number,
  rng: () => number
): void {
  if (nRows <= 1) return;
  for (let i = nRows - 1; i > 0; i--) {
    const j = __randomBelow(rng, i + 1);
    if (j !== i) {
      // Direct element swap of the two rows. For the narrow rows these
      // datasets produce (2-3 columns), this beats subarray/copyWithin/set,
      // which allocate a view and run three typed-array ops per swap.
      let iOff = i * nCols;
      let jOff = j * nCols;
      for (let k = 0; k < nCols; k++) {
        const t = x[iOff] as number;
        x[iOff] = x[jOff] as number;
        x[jOff] = t;
        iOff++;
        jOff++;
      }
      const ty = y[i] as number;
      y[i] = y[j] as number;
      y[j] = ty;
    }
  }
}

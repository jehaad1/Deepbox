/**
 * Software Float16 (IEEE 754 half-precision) and BFloat16 typed arrays.
 *
 * JavaScript has no native Float16Array, so these classes wrap a Uint16Array
 * for storage and convert to/from float64 on access. This provides memory
 * efficiency (2 bytes per element) while maintaining correct IEEE 754
 * half-precision semantics.
 *
 * @module ndarray/tensor/float16
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { IndexError, InvalidParameterError } from "../../core/errors/index";

/**
 * Resolve a TypedArray-style relative bound: truncated, negative values count
 * from the end, and the result is clamped to `[0, length]`.
 */
function resolveBound(value: number, length: number): number {
  const v = Math.trunc(value);
  if (Number.isNaN(v)) return 0;
  return v < 0 ? Math.max(0, length + v) : Math.min(v, length);
}

// ─── IEEE 754 Half-Precision Conversion ──────────────────────────────────────

const FLOAT16_EXPONENT_BIAS = 15;
const FLOAT16_MANTISSA_BITS = 10;

const float32View = new Float32Array(1);
const int32View = new Int32Array(float32View.buffer);

/** Round to nearest integer, ties to even (IEEE 754 default rounding). */
function roundTiesToEven(x: number): number {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (diff > 0.5) return fl + 1;
  if (diff < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/**
 * Convert a JavaScript number (float64) to IEEE 754 half-precision (uint16).
 *
 * Handles:
 * - NaN, ±Infinity
 * - Denormalized numbers
 * - Rounding to nearest even
 * - Overflow to Infinity
 * - Underflow to zero
 */
function float64ToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00; // NaN
  if (value === 0) return Object.is(value, -0) ? 0x8000 : 0;
  if (!Number.isFinite(value)) return value > 0 ? 0x7c00 : 0xfc00;

  // Convert directly from float64, because rounding through a float32 intermediate
  // double-rounds (e.g. 65519.999 → f32 65520 → Infinity instead of 65504)
  // and drops sticky bits for denormals.
  const sign = value < 0 ? 0x8000 : 0;
  const abs = Math.abs(value);

  // 65520 is the midpoint between the max half (65504) and 2^16; values at or
  // above it round to Infinity under round-to-nearest-even.
  if (abs >= 65520) return sign | 0x7c00;

  if (abs < 2 ** -14) {
    // Subnormal: value = m * 2^-24 with m in [0, 1024); round m with full
    // float64 precision so shifted-out (sticky) bits still influence rounding.
    const m = roundTiesToEven(abs * 2 ** 24);
    // m === 1024 means it rounded up to the smallest normal (2^-14)
    return sign | m;
  }

  // Normal: find the base-2 exponent (guarding against log2 rounding at
  // power-of-two boundaries), then round the 10-bit mantissa.
  let e = Math.floor(Math.log2(abs));
  if (abs / 2 ** e < 1) e--;
  else if (abs / 2 ** e >= 2) e++;
  let m = roundTiesToEven((abs / 2 ** e - 1) * 1024);
  if (m === 1024) {
    m = 0;
    e++;
    if (e > 15) return sign | 0x7c00;
  }
  return sign | ((e + FLOAT16_EXPONENT_BIAS) << FLOAT16_MANTISSA_BITS) | m;
}

/**
 * Convert IEEE 754 half-precision (uint16) to a JavaScript number (float64).
 */
function float16BitsToFloat64(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >>> FLOAT16_MANTISSA_BITS) & 0x1f;
  const mantissa = bits & 0x03ff;

  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : NaN;
  }

  if (exponent === 0) {
    // Denormalized
    return sign * 2 ** -14 * (mantissa / 1024);
  }

  return sign * 2 ** (exponent - FLOAT16_EXPONENT_BIAS) * (1 + mantissa / 1024);
}

// ─── BFloat16 Conversion ─────────────────────────────────────────────────────

/**
 * Convert a JavaScript number to bfloat16 (truncated float32 upper 16 bits).
 *
 * BFloat16 has the same exponent range as float32 (8 bits) but only 7
 * mantissa bits, making it ideal for deep learning where range matters
 * more than precision.
 */
function float64ToBFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7fc0; // NaN
  float32View[0] = value;
  const f32bits = int32View[0]!;
  // Round to nearest even (add 0x7fff + LSB of result)
  const roundingBias = ((f32bits >> 16) & 1) + 0x7fff;
  return (f32bits + roundingBias) >>> 16;
}

/**
 * Convert bfloat16 bits to a JavaScript number.
 */
function bfloat16BitsToFloat64(bits: number): number {
  // BFloat16 is the upper 16 bits of float32
  int32View[0] = bits << 16;
  return float32View[0]!;
}

// ─── Float16Array ───────────────────────────────────────────────────────────────

/**
 * Software IEEE 754 half-precision floating-point array.
 *
 * Stores elements in 2 bytes each (Uint16Array backing) and converts
 * to/from float64 on access. Provides the same interface as native
 * TypedArrays so they can be used with Deepbox tensors.
 *
 * @example
 * ```ts
 * import { Float16Array } from 'deepbox/ndarray';
 *
 * const arr = Float16Array.from([1.5, 2.5, 3.5]);
 * console.log(arr[0]); // 1.5 (after float16 rounding)
 * console.log(arr.length); // 3
 * console.log(arr.BYTES_PER_ELEMENT); // 2
 * ```
 */
export class Float16Array {
  /** Bytes per element (always 2). */
  static readonly BYTES_PER_ELEMENT = 2;

  /** Numeric index access (implemented via a Proxy in the constructor). */
  [index: number]: number;
  /** Underlying uint16 storage. */
  readonly _storage: Uint16Array;
  /** Number of elements. */
  readonly length: number;
  /** Bytes per element (always 2). */
  readonly BYTES_PER_ELEMENT = 2;
  /** ArrayBuffer backing the storage. */
  readonly buffer: ArrayBuffer;
  /** Byte offset into the buffer. */
  readonly byteOffset: number;
  /** Byte length of the storage. */
  readonly byteLength: number;

  /**
   * @param lengthOrData - Element count, an array-like of numbers (each is
   *   rounded to float16), or an `ArrayBuffer` holding raw 16-bit patterns.
   * @param byteOffset - Byte offset into the buffer (buffer form only).
   * @param length - Element count to view (buffer form only).
   * @throws {InvalidParameterError} If a numeric length is not a non-negative integer.
   */
  constructor(
    lengthOrData: number | ArrayLike<number> | ArrayBuffer,
    byteOffset?: number,
    length?: number
  ) {
    if (typeof lengthOrData === "number") {
      if (!Number.isInteger(lengthOrData) || lengthOrData < 0) {
        throw new InvalidParameterError(
          `Float16Array length must be a non-negative integer; received ${String(lengthOrData)}`,
          "length",
          lengthOrData
        );
      }
      this._storage = new Uint16Array(lengthOrData);
      this.length = lengthOrData;
    } else if (lengthOrData instanceof ArrayBuffer) {
      this._storage = new Uint16Array(lengthOrData, byteOffset, length);
      this.length = this._storage.length;
    } else {
      this._storage = new Uint16Array(lengthOrData.length);
      this.length = lengthOrData.length;
      for (let i = 0; i < lengthOrData.length; i++) {
        this._storage[i] = float64ToFloat16Bits(Number(lengthOrData[i]));
      }
    }
    this.buffer = this._storage.buffer as ArrayBuffer;
    this.byteOffset = this._storage.byteOffset;
    this.byteLength = this._storage.byteLength;

    // biome-ignore lint/correctness/noConstructorReturn: Proxy is required for indexed element access with automatic float16 conversion
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            return float16BitsToFloat64(target._storage[idx]!);
          }
          return undefined;
        }
        return Reflect.get(target, prop, receiver);
      },
      set(target, prop, value) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            target._storage[idx] = float64ToFloat16Bits(Number(value));
            return true;
          }
          return true;
        }
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Create a Float16Array from an iterable of numbers.
   */
  static from(source: ArrayLike<number> | Iterable<number>): Float16Array {
    const arr = Array.isArray(source) ? source : Array.from(source as Iterable<number>);
    return new Float16Array(arr);
  }

  /**
   * Create a Float16Array with the given values.
   */
  static of(...values: number[]): Float16Array {
    return new Float16Array(values);
  }

  /**
   * Get the element at `index` (negative counts from the end), or
   * `undefined` when out of range.
   */
  at(index: number): number | undefined {
    const i = Math.trunc(index) || 0;
    const idx = i < 0 ? this.length + i : i;
    if (!(idx >= 0 && idx < this.length)) return undefined;
    return float16BitsToFloat64(this._storage[idx]!);
  }

  /**
   * Copy numbers from `source` into this array starting at `offset`, rounding
   * each to float16.
   *
   * @throws {IndexError} If the values do not fit starting at `offset`
   *   (same condition as `TypedArray.prototype.set`).
   */
  set(source: ArrayLike<number>, offset = 0): void {
    if (!Number.isInteger(offset) || offset < 0 || offset + source.length > this.length) {
      throw new IndexError(
        `cannot set ${source.length} values at offset ${String(offset)} in an array of length ${this.length}`,
        { index: offset, validRange: [0, this.length] }
      );
    }
    for (let i = 0; i < source.length; i++) {
      this._storage[offset + i] = float64ToFloat16Bits(Number(source[i]));
    }
  }

  /**
   * Create a copy of a portion of the array.
   */
  slice(start = 0, end = this.length): Float16Array {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    const result = new Float16Array(Math.max(0, e - s));
    result._storage.set(this._storage.subarray(s, Math.max(s, e)));
    return result;
  }

  /**
   * Create a view over a portion of the array that shares memory with it.
   */
  subarray(begin = 0, end = this.length): Float16Array {
    const sub = this._storage.subarray(begin, end);
    return new Float16Array(sub.buffer as ArrayBuffer, sub.byteOffset, sub.length);
  }

  /**
   * Fill the range `[start, end)` with a value rounded to float16.
   */
  fill(value: number, start = 0, end = this.length): this {
    const bits = float64ToFloat16Bits(value);
    this._storage.fill(bits, start, end);
    return this;
  }

  /**
   * Copy elements within the array.
   */
  copyWithin(target: number, start: number, end?: number): this {
    this._storage.copyWithin(target, start, end);
    return this;
  }

  /** Iterator over the element values. */
  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield float16BitsToFloat64(this._storage[i]!);
    }
  }

  /** String tag. */
  get [Symbol.toStringTag](): string {
    return "Float16Array";
  }

  /** Convert to a regular Array. */
  toArray(): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(float16BitsToFloat64(this._storage[i]!));
    }
    return result;
  }
}

// ─── BFloat16Array ───────────────────────────────────────────────────────────────

/**
 * Software BFloat16 (Brain Floating Point) array.
 *
 * BFloat16 uses the same exponent range as float32 (8 bits) but only
 * 7 mantissa bits, making it ideal for deep learning workloads where
 * dynamic range matters more than precision.
 *
 * @example
 * ```ts
 * import { BFloat16Array } from 'deepbox/ndarray';
 *
 * const arr = BFloat16Array.from([1.0, 2.0, 3.0]);
 * console.log(arr[0]); // 1.0
 * console.log(arr.BYTES_PER_ELEMENT); // 2
 * ```
 */
export class BFloat16Array {
  /** Bytes per element (always 2). */
  static readonly BYTES_PER_ELEMENT = 2;

  /** Numeric index access (implemented via a Proxy in the constructor). */
  [index: number]: number;
  /** Underlying uint16 storage. */
  readonly _storage: Uint16Array;
  /** Number of elements. */
  readonly length: number;
  /** Bytes per element (always 2). */
  readonly BYTES_PER_ELEMENT = 2;
  /** ArrayBuffer backing the storage. */
  readonly buffer: ArrayBuffer;
  /** Byte offset into the buffer. */
  readonly byteOffset: number;
  /** Byte length of the storage. */
  readonly byteLength: number;

  /**
   * @param lengthOrData - Element count, an array-like of numbers (each is
   *   rounded to bfloat16), or an `ArrayBuffer` holding raw 16-bit patterns.
   * @param byteOffset - Byte offset into the buffer (buffer form only).
   * @param length - Element count to view (buffer form only).
   * @throws {InvalidParameterError} If a numeric length is not a non-negative integer.
   */
  constructor(
    lengthOrData: number | ArrayLike<number> | ArrayBuffer,
    byteOffset?: number,
    length?: number
  ) {
    if (typeof lengthOrData === "number") {
      if (!Number.isInteger(lengthOrData) || lengthOrData < 0) {
        throw new InvalidParameterError(
          `BFloat16Array length must be a non-negative integer; received ${String(lengthOrData)}`,
          "length",
          lengthOrData
        );
      }
      this._storage = new Uint16Array(lengthOrData);
      this.length = lengthOrData;
    } else if (lengthOrData instanceof ArrayBuffer) {
      this._storage = new Uint16Array(lengthOrData, byteOffset, length);
      this.length = this._storage.length;
    } else {
      this._storage = new Uint16Array(lengthOrData.length);
      this.length = lengthOrData.length;
      for (let i = 0; i < lengthOrData.length; i++) {
        this._storage[i] = float64ToBFloat16Bits(Number(lengthOrData[i]));
      }
    }
    this.buffer = this._storage.buffer as ArrayBuffer;
    this.byteOffset = this._storage.byteOffset;
    this.byteLength = this._storage.byteLength;

    // biome-ignore lint/correctness/noConstructorReturn: Proxy is required for indexed element access with automatic bfloat16 conversion
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            return bfloat16BitsToFloat64(target._storage[idx]!);
          }
          return undefined;
        }
        return Reflect.get(target, prop, receiver);
      },
      set(target, prop, value) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            target._storage[idx] = float64ToBFloat16Bits(Number(value));
            return true;
          }
          return true;
        }
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Create a BFloat16Array from an iterable of numbers.
   */
  static from(source: ArrayLike<number> | Iterable<number>): BFloat16Array {
    const arr = Array.isArray(source) ? source : Array.from(source as Iterable<number>);
    return new BFloat16Array(arr);
  }

  /**
   * Create a BFloat16Array with the given values.
   */
  static of(...values: number[]): BFloat16Array {
    return new BFloat16Array(values);
  }

  /**
   * Get the element at `index` (negative counts from the end), or
   * `undefined` when out of range.
   */
  at(index: number): number | undefined {
    const i = Math.trunc(index) || 0;
    const idx = i < 0 ? this.length + i : i;
    if (!(idx >= 0 && idx < this.length)) return undefined;
    return bfloat16BitsToFloat64(this._storage[idx]!);
  }

  /**
   * Copy numbers from `source` into this array starting at `offset`, rounding
   * each to bfloat16.
   *
   * @throws {IndexError} If the values do not fit starting at `offset`
   *   (same condition as `TypedArray.prototype.set`).
   */
  set(source: ArrayLike<number>, offset = 0): void {
    if (!Number.isInteger(offset) || offset < 0 || offset + source.length > this.length) {
      throw new IndexError(
        `cannot set ${source.length} values at offset ${String(offset)} in an array of length ${this.length}`,
        { index: offset, validRange: [0, this.length] }
      );
    }
    for (let i = 0; i < source.length; i++) {
      this._storage[offset + i] = float64ToBFloat16Bits(Number(source[i]));
    }
  }

  /**
   * Create a copy of a portion of the array.
   */
  slice(start = 0, end = this.length): BFloat16Array {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    const result = new BFloat16Array(Math.max(0, e - s));
    result._storage.set(this._storage.subarray(s, Math.max(s, e)));
    return result;
  }

  /**
   * Create a view over a portion of the array that shares memory with it.
   */
  subarray(begin = 0, end = this.length): BFloat16Array {
    const sub = this._storage.subarray(begin, end);
    return new BFloat16Array(sub.buffer as ArrayBuffer, sub.byteOffset, sub.length);
  }

  /**
   * Fill the range `[start, end)` with a value rounded to bfloat16.
   */
  fill(value: number, start = 0, end = this.length): this {
    const bits = float64ToBFloat16Bits(value);
    this._storage.fill(bits, start, end);
    return this;
  }

  /**
   * Copy elements within the array.
   */
  copyWithin(target: number, start: number, end?: number): this {
    this._storage.copyWithin(target, start, end);
    return this;
  }

  /** Iterator over the element values. */
  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield bfloat16BitsToFloat64(this._storage[i]!);
    }
  }

  /** String tag. */
  get [Symbol.toStringTag](): string {
    return "BFloat16Array";
  }

  /** Convert to a regular Array. */
  toArray(): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(bfloat16BitsToFloat64(this._storage[i]!));
    }
    return result;
  }
}

// ─── Conversion Utilities ────────────────────────────────────────────────────

/**
 * Round a number to the nearest representable IEEE 754 binary16 value
 * (ties to even, overflow to ±Infinity).
 */
function roundToFloat16(value: number): number {
  return float16BitsToFloat64(float64ToFloat16Bits(value));
}

/** Round a number to the nearest representable bfloat16 value (ties to even). */
function roundToBFloat16(value: number): number {
  return bfloat16BitsToFloat64(float64ToBFloat16Bits(value));
}

/**
 * Convert float16 bits to float64, and the reverse (`float64ToFloat16Bits`).
 * The bfloat16 pair works the same way. Useful for serialization.
 * `roundToFloat16` / `roundToBFloat16` snap a number onto the half-precision grid.
 */
export {
  bfloat16BitsToFloat64,
  float16BitsToFloat64,
  float64ToBFloat16Bits,
  float64ToFloat16Bits,
  roundToBFloat16,
  roundToFloat16,
};

/**
 * Complex number types and typed arrays for complex64/complex128 support.
 *
 * Complex64 stores each element as two float32 values (real, imaginary).
 * Complex128 stores each element as two float64 values (real, imaginary).
 *
 * @module ndarray/tensor/complex
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { IndexError, InvalidParameterError } from "../../core/errors/index";

// ─── Shared helpers ──────────────────────────────────────────────────────────

function assertValidLength(length: number, who: string): void {
  if (!Number.isInteger(length) || length < 0) {
    throw new InvalidParameterError(
      `${who} length must be a non-negative integer; received ${String(length)}`,
      "length",
      length
    );
  }
}

function assertValidIndex(index: number, length: number, who: string): void {
  if (!Number.isInteger(index) || index < 0 || index >= length) {
    throw new IndexError(`${who} index ${String(index)} is out of bounds for length ${length}`, {
      index,
      validRange: [0, Math.max(0, length - 1)],
    });
  }
}

/**
 * Resolve a TypedArray-style relative bound (negative counts from the end,
 * values are truncated and clamped to `[0, length]`).
 */
function resolveBound(value: number, length: number): number {
  const v = Math.trunc(value);
  if (Number.isNaN(v)) return 0;
  return v < 0 ? Math.max(0, length + v) : Math.min(v, length);
}

// ─── Complex Number ──────────────────────────────────────────────────────────

/**
 * An immutable complex number with real and imaginary parts.
 *
 * @example
 * ```ts
 * import { Complex } from 'deepbox/ndarray';
 *
 * const z = new Complex(3, 4);
 * console.log(z.abs());   // 5
 * console.log(z.phase()); // 0.9273...
 * console.log(z.conj());  // Complex(3, -4)
 * ```
 */
export class Complex {
  readonly re: number;
  readonly im: number;

  constructor(re: number, im = 0) {
    this.re = re;
    this.im = im;
  }

  /** Absolute value (modulus). Uses hypot to avoid overflow/underflow of re²+im². */
  abs(): number {
    return Math.hypot(this.re, this.im);
  }

  /** Phase angle (argument) in radians. */
  phase(): number {
    return Math.atan2(this.im, this.re);
  }

  /** Complex conjugate. */
  conj(): Complex {
    return new Complex(this.re, -this.im);
  }

  /** Addition. */
  add(other: Complex): Complex {
    return new Complex(this.re + other.re, this.im + other.im);
  }

  /** Subtraction. */
  sub(other: Complex): Complex {
    return new Complex(this.re - other.re, this.im - other.im);
  }

  /** Multiplication. */
  mul(other: Complex): Complex {
    return new Complex(
      this.re * other.re - this.im * other.im,
      this.re * other.im + this.im * other.re
    );
  }

  /**
   * Division using Smith's algorithm. The naive formula computes
   * re²+im² of the denominator, which under/overflows for well-scaled
   * quotients (e.g. (1e-300+1e-300j) / (1e-300+1e-300j) would give NaN).
   */
  div(other: Complex): Complex {
    const { re: a, im: b } = this;
    const { re: c, im: d } = other;
    if (Math.abs(c) >= Math.abs(d)) {
      if (c === 0 && d === 0) {
        return new Complex(a / 0, b / 0);
      }
      const r = d / c;
      const denom = c + d * r;
      return new Complex((a + b * r) / denom, (b - a * r) / denom);
    }
    const r = c / d;
    const denom = c * r + d;
    return new Complex((a * r + b) / denom, (b * r - a) / denom);
  }

  /** Negation. */
  neg(): Complex {
    return new Complex(-this.re, -this.im);
  }

  /** Complex exponential. */
  exp(): Complex {
    const er = Math.exp(this.re);
    // A zero imaginary part keeps the result exactly real (exp(inf+0j) = inf+0j).
    if (this.im === 0) return new Complex(er, this.im);
    return new Complex(er * Math.cos(this.im), er * Math.sin(this.im));
  }

  /**
   * Complex natural logarithm (principal value).
   *
   * Near the unit circle the real part is computed with `log1p` on
   * `|z|² - 1`, so `log(1 + 1e-10j)` keeps its `5e-21` real part instead of
   * rounding `|z|` to exactly 1.
   */
  log(): Complex {
    const ax = Math.abs(this.re);
    const ay = Math.abs(this.im);
    const hi = Math.max(ax, ay);
    const lo = Math.min(ax, ay);
    const logAbs =
      hi >= 0.5 && hi <= 2
        ? 0.5 * Math.log1p((hi - 1) * (hi + 1) + lo * lo)
        : Math.log(Math.hypot(this.re, this.im));
    return new Complex(logAbs, this.phase());
  }

  /**
   * Complex square root (principal value, branch cut along the negative real
   * axis). Uses the half-angle-free formulation so that `sqrt(-4+0j)` is
   * exactly `2j` and signed zeros select the side of the branch cut.
   */
  sqrt(): Complex {
    const { re, im } = this;
    if (re === 0 && im === 0) return new Complex(0, im);
    if (im === Number.POSITIVE_INFINITY || im === Number.NEGATIVE_INFINITY) {
      return new Complex(Number.POSITIVE_INFINITY, im);
    }
    const negativeSide = im < 0 || Object.is(im, -0);
    if (re === Number.POSITIVE_INFINITY) {
      return new Complex(re, Number.isNaN(im) ? Number.NaN : negativeSide ? -0 : 0);
    }
    if (re === Number.NEGATIVE_INFINITY) {
      if (Number.isNaN(im)) return new Complex(Number.NaN, Number.POSITIVE_INFINITY);
      return new Complex(0, negativeSide ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY);
    }
    if (Number.isNaN(re) || Number.isNaN(im)) return new Complex(Number.NaN, Number.NaN);

    // Scale extreme magnitudes so hypot and the halving below neither overflow
    // nor underflow; the result is rescaled by the square root of the factor.
    const big = Math.max(Math.abs(re), Math.abs(im));
    let scale = 1;
    let result = 1;
    if (big >= 2 ** 1000) {
      scale = 0.25;
      result = 2;
    } else if (big < 2 ** -900) {
      scale = 2 ** 600;
      result = 2 ** -300;
    }
    const a = re * scale;
    const b = im * scale;
    const m = Math.hypot(a, b);
    let x: number;
    let y: number;
    if (a >= 0) {
      x = Math.sqrt((m + a) / 2);
      y = b / (2 * x);
    } else {
      y = Math.sqrt((m - a) / 2);
      x = Math.abs(b) / (2 * y);
      if (negativeSide) y = -y;
    }
    return new Complex(x * result, y * result);
  }

  /**
   * Complex power.
   *
   * Real integer exponents with magnitude below 100 use repeated
   * multiplication (exact for Gaussian integers, same as NumPy); every other
   * exponent goes through `exp(n * log(z))` on the principal branch.
   */
  pow(n: Complex | number): Complex {
    const exp = typeof n === "number" ? new Complex(n, 0) : n;
    if (exp.re === 0 && exp.im === 0) {
      // z ** 0 = 1 for every z, including 0 (NumPy convention)
      return new Complex(1, 0);
    }
    if (this.re === 0 && this.im === 0) {
      return exp.re > 0 ? new Complex(0, 0) : new Complex(NaN, NaN);
    }
    if (exp.im === 0 && Number.isInteger(exp.re) && Math.abs(exp.re) < 100) {
      let k = Math.abs(exp.re);
      let result: Complex = Complex.ONE;
      let base: Complex = this;
      while (k > 0) {
        if (k % 2 === 1) result = result.mul(base);
        k = Math.floor(k / 2);
        if (k > 0) base = base.mul(base);
      }
      return exp.re < 0 ? Complex.ONE.div(result) : result;
    }
    return this.log().mul(exp).exp();
  }

  /**
   * Check equality within an absolute tolerance on both parts.
   * Identical parts (including matching infinities) always compare equal;
   * NaN never does.
   */
  equals(other: Complex, tol = 1e-10): boolean {
    const close = (a: number, b: number): boolean => a === b || Math.abs(a - b) <= tol;
    return close(this.re, other.re) && close(this.im, other.im);
  }

  /** String representation. */
  toString(): string {
    if (this.im === 0) return `${this.re}`;
    if (this.re === 0) return `${this.im}j`;
    const sign = this.im >= 0 || Number.isNaN(this.im) ? "+" : "";
    return `(${this.re}${sign}${this.im}j)`;
  }

  /** Create from polar coordinates. */
  static fromPolar(r: number, theta: number): Complex {
    return new Complex(r * Math.cos(theta), r * Math.sin(theta));
  }

  /** Zero constant. */
  static readonly ZERO = new Complex(0, 0);
  /** One constant. */
  static readonly ONE = new Complex(1, 0);
  /** Imaginary unit. */
  static readonly I = new Complex(0, 1);
}

// ─── Complex64Array ──────────────────────────────────────────────────────────

/**
 * Complex64 typed array: each element is two float32 values (real, imag).
 *
 * The `length` property reports the number of complex elements.
 * The underlying Float32Array has `2 * length` entries.
 *
 * Index access returns the real part as a number. Use `getComplex(i)` to
 * get the full Complex value, or `getReal(i)`/`getImag(i)` for components.
 *
 * @example
 * ```ts
 * import { Complex64Array, Complex } from 'deepbox/ndarray';
 *
 * const arr = Complex64Array.fromComplexArray([
 *   new Complex(1, 2),
 *   new Complex(3, 4),
 * ]);
 * console.log(arr.getComplex(0)); // Complex(1, 2)
 * console.log(arr.length);        // 2
 * ```
 */
export class Complex64Array {
  /**
   * Numeric index access (via the constructor Proxy): reading returns the
   * real part; writing stores the value as the real part and resets the
   * imaginary part to 0. Out-of-range reads return `undefined` and
   * out-of-range writes are ignored, like a native TypedArray.
   */
  [index: number]: number;
  /** Underlying interleaved float32 storage: [re0, im0, re1, im1, ...] */
  readonly _storage: Float32Array;
  /** Number of complex elements. */
  readonly length: number;
  /** Bytes per complex element (8 = 2 × 4). */
  readonly BYTES_PER_ELEMENT = 8;
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;

  /**
   * @param length - Number of complex elements (a non-negative integer).
   * @throws {InvalidParameterError} If `length` is not a non-negative integer.
   */
  constructor(length: number) {
    assertValidLength(length, "Complex64Array");
    this._storage = new Float32Array(length * 2);
    this.length = length;
    this.buffer = this._storage.buffer as ArrayBuffer;
    this.byteOffset = this._storage.byteOffset;
    this.byteLength = this._storage.byteLength;

    // biome-ignore lint/correctness/noConstructorReturn: Proxy is required for indexed element access returning real part of complex numbers
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            return target._storage[idx * 2]!;
          }
          return undefined;
        }
        return Reflect.get(target, prop, receiver);
      },
      set(target, prop, value) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            target._storage[idx * 2] = value as number;
            target._storage[idx * 2 + 1] = 0;
            return true;
          }
          return true;
        }
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Get the real part at `index`.
   * @throws {IndexError} If `index` is not an integer inside `[0, length)`.
   */
  getReal(index: number): number {
    assertValidIndex(index, this.length, "Complex64Array");
    return this._storage[index * 2] as number;
  }

  /**
   * Get the imaginary part at `index`.
   * @throws {IndexError} If `index` is not an integer inside `[0, length)`.
   */
  getImag(index: number): number {
    assertValidIndex(index, this.length, "Complex64Array");
    return this._storage[index * 2 + 1] as number;
  }

  /**
   * Get the element at `index` as a {@link Complex}.
   * @throws {IndexError} If `index` is out of range.
   */
  getComplex(index: number): Complex {
    return new Complex(this.getReal(index), this.getImag(index));
  }

  /**
   * Store a complex value at `index`.
   * @throws {IndexError} If `index` is out of range.
   */
  setComplex(index: number, value: Complex): void {
    assertValidIndex(index, this.length, "Complex64Array");
    this._storage[index * 2] = value.re;
    this._storage[index * 2 + 1] = value.im;
  }

  /**
   * Store real and imaginary parts at `index`.
   * @throws {IndexError} If `index` is out of range.
   */
  setRI(index: number, re: number, im: number): void {
    assertValidIndex(index, this.length, "Complex64Array");
    this._storage[index * 2] = re;
    this._storage[index * 2 + 1] = im;
  }

  /** Create from an array of {@link Complex} values. */
  static fromComplexArray(values: readonly Complex[]): Complex64Array {
    const arr = new Complex64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      arr._storage[i * 2] = v.re;
      arr._storage[i * 2 + 1] = v.im;
    }
    return arr;
  }

  /**
   * Create from interleaved real/imaginary pairs `[re0, im0, re1, im1, ...]`.
   * @throws {InvalidParameterError} If `data` has an odd number of entries.
   */
  static fromInterleaved(data: ArrayLike<number>): Complex64Array {
    if (data.length % 2 !== 0) {
      throw new InvalidParameterError(
        `interleaved data must hold real/imaginary pairs; received ${data.length} values`,
        "data",
        data.length
      );
    }
    const arr = new Complex64Array(data.length / 2);
    arr._storage.set(data);
    return arr;
  }

  /**
   * Fill the range `[start, end)` with a complex value. Negative bounds count
   * from the end and out-of-range bounds are clamped, as in `TypedArray.fill`.
   */
  fill(value: Complex, start = 0, end = this.length): this {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    for (let i = s; i < e; i++) {
      this._storage[i * 2] = value.re;
      this._storage[i * 2 + 1] = value.im;
    }
    return this;
  }

  /**
   * Copy of the range `[start, end)`. Negative bounds count from the end.
   */
  slice(start = 0, end = this.length): Complex64Array {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    const result = new Complex64Array(Math.max(0, e - s));
    result._storage.set(this._storage.subarray(s * 2, Math.max(s, e) * 2));
    return result;
  }

  /**
   * Copy values into this array starting at complex index `offset`.
   *
   * `source` is either another complex array (copied element-wise) or a flat
   * list of interleaved real/imaginary pairs `[re0, im0, re1, im1, ...]`.
   *
   * @throws {InvalidParameterError} If a flat `source` has an odd length.
   * @throws {IndexError} If the values do not fit starting at `offset`.
   */
  set(source: ArrayLike<number> | Complex64Array | Complex128Array, offset = 0): void {
    const flat =
      source instanceof Complex64Array || source instanceof Complex128Array
        ? source._storage
        : source;
    if (flat.length % 2 !== 0) {
      throw new InvalidParameterError(
        `interleaved source must hold real/imaginary pairs; received ${flat.length} values`,
        "source",
        flat.length
      );
    }
    if (!Number.isInteger(offset) || offset < 0 || offset + flat.length / 2 > this.length) {
      throw new IndexError(
        `cannot set ${flat.length / 2} complex values at offset ${String(offset)} in an array of length ${this.length}`,
        { index: offset, validRange: [0, this.length] }
      );
    }
    this._storage.set(flat, offset * 2);
  }

  /** Iterator over real parts (for TypedArray compatibility). */
  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield this._storage[i * 2]!;
    }
  }

  get [Symbol.toStringTag](): string {
    return "Complex64Array";
  }

  /** Convert to an array of {@link Complex} values. */
  toComplexArray(): Complex[] {
    const result: Complex[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(this.getComplex(i));
    }
    return result;
  }
}

// ─── Complex128Array ──────────────────────────────────────────────────────────

/**
 * Complex128 typed array: each element is two float64 values (real, imag).
 *
 * Same API as {@link Complex64Array} but with double precision.
 *
 * @example
 * ```ts
 * import { Complex128Array, Complex } from 'deepbox/ndarray';
 *
 * const arr = Complex128Array.fromComplexArray([
 *   new Complex(1.5, 2.5),
 *   new Complex(3.14, -1.0),
 * ]);
 * console.log(arr.getComplex(1)); // Complex(3.14, -1)
 * ```
 */
export class Complex128Array {
  /**
   * Numeric index access (via the constructor Proxy): reading returns the
   * real part; writing stores the value as the real part and resets the
   * imaginary part to 0. Out-of-range reads return `undefined` and
   * out-of-range writes are ignored, like a native TypedArray.
   */
  [index: number]: number;
  /** Underlying interleaved float64 storage: [re0, im0, re1, im1, ...] */
  readonly _storage: Float64Array;
  /** Number of complex elements. */
  readonly length: number;
  /** Bytes per complex element (16 = 2 × 8). */
  readonly BYTES_PER_ELEMENT = 16;
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;

  /**
   * @param length - Number of complex elements (a non-negative integer).
   * @throws {InvalidParameterError} If `length` is not a non-negative integer.
   */
  constructor(length: number) {
    assertValidLength(length, "Complex128Array");
    this._storage = new Float64Array(length * 2);
    this.length = length;
    this.buffer = this._storage.buffer as ArrayBuffer;
    this.byteOffset = this._storage.byteOffset;
    this.byteLength = this._storage.byteLength;

    // biome-ignore lint/correctness/noConstructorReturn: Proxy is required for indexed element access returning real part of complex numbers
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            return target._storage[idx * 2]!;
          }
          return undefined;
        }
        return Reflect.get(target, prop, receiver);
      },
      set(target, prop, value) {
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          const idx = Number(prop);
          if (idx >= 0 && idx < target.length) {
            target._storage[idx * 2] = value as number;
            target._storage[idx * 2 + 1] = 0;
            return true;
          }
          return true;
        }
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Get the real part at `index`.
   * @throws {IndexError} If `index` is not an integer inside `[0, length)`.
   */
  getReal(index: number): number {
    assertValidIndex(index, this.length, "Complex128Array");
    return this._storage[index * 2] as number;
  }

  /**
   * Get the imaginary part at `index`.
   * @throws {IndexError} If `index` is not an integer inside `[0, length)`.
   */
  getImag(index: number): number {
    assertValidIndex(index, this.length, "Complex128Array");
    return this._storage[index * 2 + 1] as number;
  }

  /**
   * Get the element at `index` as a {@link Complex}.
   * @throws {IndexError} If `index` is out of range.
   */
  getComplex(index: number): Complex {
    return new Complex(this.getReal(index), this.getImag(index));
  }

  /**
   * Store a complex value at `index`.
   * @throws {IndexError} If `index` is out of range.
   */
  setComplex(index: number, value: Complex): void {
    assertValidIndex(index, this.length, "Complex128Array");
    this._storage[index * 2] = value.re;
    this._storage[index * 2 + 1] = value.im;
  }

  /**
   * Store real and imaginary parts at `index`.
   * @throws {IndexError} If `index` is out of range.
   */
  setRI(index: number, re: number, im: number): void {
    assertValidIndex(index, this.length, "Complex128Array");
    this._storage[index * 2] = re;
    this._storage[index * 2 + 1] = im;
  }

  /** Create from an array of {@link Complex} values. */
  static fromComplexArray(values: readonly Complex[]): Complex128Array {
    const arr = new Complex128Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      arr._storage[i * 2] = v.re;
      arr._storage[i * 2 + 1] = v.im;
    }
    return arr;
  }

  /**
   * Create from interleaved real/imaginary pairs `[re0, im0, re1, im1, ...]`.
   * @throws {InvalidParameterError} If `data` has an odd number of entries.
   */
  static fromInterleaved(data: ArrayLike<number>): Complex128Array {
    if (data.length % 2 !== 0) {
      throw new InvalidParameterError(
        `interleaved data must hold real/imaginary pairs; received ${data.length} values`,
        "data",
        data.length
      );
    }
    const arr = new Complex128Array(data.length / 2);
    arr._storage.set(data);
    return arr;
  }

  /**
   * Fill the range `[start, end)` with a complex value. Negative bounds count
   * from the end and out-of-range bounds are clamped, as in `TypedArray.fill`.
   */
  fill(value: Complex, start = 0, end = this.length): this {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    for (let i = s; i < e; i++) {
      this._storage[i * 2] = value.re;
      this._storage[i * 2 + 1] = value.im;
    }
    return this;
  }

  /**
   * Copy of the range `[start, end)`. Negative bounds count from the end.
   */
  slice(start = 0, end = this.length): Complex128Array {
    const s = resolveBound(start, this.length);
    const e = resolveBound(end, this.length);
    const result = new Complex128Array(Math.max(0, e - s));
    result._storage.set(this._storage.subarray(s * 2, Math.max(s, e) * 2));
    return result;
  }

  /**
   * Copy values into this array starting at complex index `offset`.
   *
   * `source` is either another complex array (copied element-wise) or a flat
   * list of interleaved real/imaginary pairs `[re0, im0, re1, im1, ...]`.
   *
   * @throws {InvalidParameterError} If a flat `source` has an odd length.
   * @throws {IndexError} If the values do not fit starting at `offset`.
   */
  set(source: ArrayLike<number> | Complex64Array | Complex128Array, offset = 0): void {
    const flat =
      source instanceof Complex64Array || source instanceof Complex128Array
        ? source._storage
        : source;
    if (flat.length % 2 !== 0) {
      throw new InvalidParameterError(
        `interleaved source must hold real/imaginary pairs; received ${flat.length} values`,
        "source",
        flat.length
      );
    }
    if (!Number.isInteger(offset) || offset < 0 || offset + flat.length / 2 > this.length) {
      throw new IndexError(
        `cannot set ${flat.length / 2} complex values at offset ${String(offset)} in an array of length ${this.length}`,
        { index: offset, validRange: [0, this.length] }
      );
    }
    this._storage.set(flat, offset * 2);
  }

  /** Iterator over real parts (for TypedArray compatibility). */
  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield this._storage[i * 2]!;
    }
  }

  get [Symbol.toStringTag](): string {
    return "Complex128Array";
  }

  /** Convert to an array of {@link Complex} values. */
  toComplexArray(): Complex[] {
    const result: Complex[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(this.getComplex(i));
    }
    return result;
  }
}

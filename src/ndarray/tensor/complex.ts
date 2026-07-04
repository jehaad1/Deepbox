/**
 * Complex number types and typed arrays for complex64/complex128 support.
 *
 * Complex64 stores each element as two float32 values (real, imaginary).
 * Complex128 stores each element as two float64 values (real, imaginary).
 *
 * @module ndarray/tensor/complex
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

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
    return new Complex(er * Math.cos(this.im), er * Math.sin(this.im));
  }

  /** Complex natural logarithm (principal value). */
  log(): Complex {
    return new Complex(Math.log(this.abs()), this.phase());
  }

  /** Complex square root (principal value). */
  sqrt(): Complex {
    const r = this.abs();
    const t = this.phase();
    const sr = Math.sqrt(r);
    return new Complex(sr * Math.cos(t / 2), sr * Math.sin(t / 2));
  }

  /** Complex power. */
  pow(n: Complex | number): Complex {
    const exp = typeof n === "number" ? new Complex(n, 0) : n;
    if (exp.re === 0 && exp.im === 0) {
      // z ** 0 = 1 for every z, including 0 (NumPy convention)
      return new Complex(1, 0);
    }
    if (this.re === 0 && this.im === 0) {
      return exp.re > 0 ? new Complex(0, 0) : new Complex(NaN, NaN);
    }
    return this.log().mul(exp).exp();
  }

  /** Check equality with tolerance. */
  equals(other: Complex, tol = 1e-10): boolean {
    return Math.abs(this.re - other.re) < tol && Math.abs(this.im - other.im) < tol;
  }

  /** String representation. */
  toString(): string {
    if (this.im === 0) return `${this.re}`;
    if (this.re === 0) return `${this.im}j`;
    const sign = this.im >= 0 ? "+" : "";
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
 * Complex64 typed array — each element is two float32 values (real, imag).
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
  /** Numeric index access returns the real part (via the constructor Proxy). */
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

  constructor(length: number) {
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

  /** Get the real part at index. */
  getReal(index: number): number {
    return this._storage[index * 2] ?? 0;
  }

  /** Get the imaginary part at index. */
  getImag(index: number): number {
    return this._storage[index * 2 + 1] ?? 0;
  }

  /** Get a Complex value at index. */
  getComplex(index: number): Complex {
    return new Complex(this.getReal(index), this.getImag(index));
  }

  /** Set a complex value at index. */
  setComplex(index: number, value: Complex): void {
    this._storage[index * 2] = value.re;
    this._storage[index * 2 + 1] = value.im;
  }

  /** Set real and imaginary parts at index. */
  setRI(index: number, re: number, im: number): void {
    this._storage[index * 2] = re;
    this._storage[index * 2 + 1] = im;
  }

  /** Create from an array of Complex values. */
  static fromComplexArray(values: readonly Complex[]): Complex64Array {
    const arr = new Complex64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      arr._storage[i * 2] = v.re;
      arr._storage[i * 2 + 1] = v.im;
    }
    return arr;
  }

  /** Create from interleaved real/imaginary pairs. */
  static fromInterleaved(data: ArrayLike<number>): Complex64Array {
    const len = Math.floor(data.length / 2);
    const arr = new Complex64Array(len);
    for (let i = 0; i < data.length; i++) {
      arr._storage[i] = data[i]!;
    }
    return arr;
  }

  /** Fill all elements with a complex value. */
  fill(value: Complex, start = 0, end = this.length): this {
    for (let i = start; i < end; i++) {
      this._storage[i * 2] = value.re;
      this._storage[i * 2 + 1] = value.im;
    }
    return this;
  }

  /** Create a slice copy. */
  slice(start = 0, end = this.length): Complex64Array {
    const s = start < 0 ? Math.max(0, this.length + start) : Math.min(start, this.length);
    const e = end < 0 ? Math.max(0, this.length + end) : Math.min(end, this.length);
    const len = Math.max(0, e - s);
    const result = new Complex64Array(len);
    result._storage.set(this._storage.subarray(s * 2, e * 2));
    return result;
  }

  /** Set values from another source. */
  set(source: ArrayLike<number>, offset = 0): void {
    this._storage.set(source, offset * 2);
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

  /** Convert to array of Complex values. */
  toComplexArray(): Complex[] {
    const result: Complex[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(this.getComplex(i));
    }
    return result;
  }
}

// ─── Complex128Array ─────────────────────────────────────────────────────────

/**
 * Complex128 typed array — each element is two float64 values (real, imag).
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
  /** Numeric index access returns the real part (via the constructor Proxy). */
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

  constructor(length: number) {
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
   * Get the real part of the complex number at the given index.
   */
  getReal(index: number): number {
    return this._storage[index * 2]!;
  }

  /**
   * Get the imaginary part of the complex number at the given index.
   */
  getImag(index: number): number {
    return this._storage[index * 2 + 1]!;
  }

  getComplex(index: number): Complex {
    return new Complex(this.getReal(index), this.getImag(index));
  }

  setComplex(index: number, value: Complex): void {
    this._storage[index * 2] = value.re;
    this._storage[index * 2 + 1] = value.im;
  }

  setRI(index: number, re: number, im: number): void {
    this._storage[index * 2] = re;
    this._storage[index * 2 + 1] = im;
  }

  static fromComplexArray(values: readonly Complex[]): Complex128Array {
    const arr = new Complex128Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      arr._storage[i * 2] = v.re;
      arr._storage[i * 2 + 1] = v.im;
    }
    return arr;
  }

  static fromInterleaved(data: ArrayLike<number>): Complex128Array {
    const len = Math.floor(data.length / 2);
    const arr = new Complex128Array(len);
    for (let i = 0; i < data.length; i++) {
      arr._storage[i] = data[i]!;
    }
    return arr;
  }

  fill(value: Complex, start = 0, end = this.length): this {
    for (let i = start; i < end; i++) {
      this._storage[i * 2] = value.re;
      this._storage[i * 2 + 1] = value.im;
    }
    return this;
  }

  slice(start = 0, end = this.length): Complex128Array {
    const s = start < 0 ? Math.max(0, this.length + start) : Math.min(start, this.length);
    const e = end < 0 ? Math.max(0, this.length + end) : Math.min(end, this.length);
    const len = Math.max(0, e - s);
    const result = new Complex128Array(len);
    result._storage.set(this._storage.subarray(s * 2, e * 2));
    return result;
  }

  set(source: ArrayLike<number>, offset = 0): void {
    this._storage.set(source, offset * 2);
  }

  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield this._storage[i * 2]!;
    }
  }

  get [Symbol.toStringTag](): string {
    return "Complex128Array";
  }

  toComplexArray(): Complex[] {
    const result: Complex[] = [];
    for (let i = 0; i < this.length; i++) {
      result.push(this.getComplex(i));
    }
    return result;
  }
}

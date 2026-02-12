export function expectNumber(value: unknown, context: string): number {
  if (typeof value !== "number") {
    throw new TypeError(`${context} expected a number`);
  }
  return value;
}

export function expectNumberArray(value: unknown, context: string): number[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "number")) {
    throw new TypeError(`${context} expected number[]`);
  }
  return value;
}

export function expectNumberArray2D(value: unknown, context: string): number[][] {
  if (
    !Array.isArray(value) ||
    !value.every((row) => Array.isArray(row) && row.every((item) => typeof item === "number"))
  ) {
    throw new TypeError(`${context} expected number[][]`);
  }
  return value;
}

export function expectNumberArray3D(value: unknown, context: string): number[][][] {
  if (
    !Array.isArray(value) ||
    !value.every(
      (plane) =>
        Array.isArray(plane) &&
        plane.every((row) => Array.isArray(row) && row.every((item) => typeof item === "number"))
    )
  ) {
    throw new TypeError(`${context} expected number[][][]`);
  }
  return value;
}

export function expectFloatTypedArray(
  value: unknown,
  context: string
): Float32Array | Float64Array {
  if (value instanceof Float32Array || value instanceof Float64Array) {
    return value;
  }
  throw new TypeError(`${context} expected Float32Array or Float64Array`);
}

export function expectNumericTypedArray(
  value: unknown,
  context: string
): Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array {
  if (
    value instanceof Float32Array ||
    value instanceof Float64Array ||
    value instanceof Int32Array ||
    value instanceof Uint8Array ||
    value instanceof BigInt64Array
  ) {
    return value;
  }
  throw new TypeError(`${context} expected a numeric typed array`);
}

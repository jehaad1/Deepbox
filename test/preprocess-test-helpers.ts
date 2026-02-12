import { DeepboxError } from "../src/core/errors";
import type { Tensor } from "../src/ndarray";

function assertArray(value: unknown, label: string): asserts value is unknown[] {
  if (!Array.isArray(value)) {
    throw new DeepboxError(`${label} must be an array`);
  }
}

export function assertNumberArray(value: unknown, label: string): asserts value is number[] {
  assertArray(value, label);
  for (const item of value) {
    if (typeof item !== "number") {
      throw new DeepboxError(`${label} must contain numbers`);
    }
  }
}

export function assertNumberMatrix(value: unknown, label: string): asserts value is number[][] {
  assertArray(value, label);
  for (const row of value) {
    assertNumberArray(row, `${label} row`);
  }
}

export function assertStringArray(value: unknown, label: string): asserts value is string[] {
  assertArray(value, label);
  for (const item of value) {
    if (typeof item !== "string") {
      throw new DeepboxError(`${label} must contain strings`);
    }
  }
}

export function assertStringMatrix(value: unknown, label: string): asserts value is string[][] {
  assertArray(value, label);
  for (const row of value) {
    assertStringArray(row, `${label} row`);
  }
}

export function getFloat64Data(t: Tensor): Float64Array {
  if (t.dtype !== "float64") {
    throw new DeepboxError(`Expected float64 tensor, got ${t.dtype}`);
  }
  if (!(t.data instanceof Float64Array)) {
    throw new DeepboxError("Expected Float64Array storage");
  }
  return t.data;
}

export function getStringData(t: Tensor): string[] {
  if (t.dtype !== "string") {
    throw new DeepboxError(`Expected string tensor, got ${t.dtype}`);
  }
  if (!Array.isArray(t.data)) {
    throw new DeepboxError("Expected string array storage");
  }
  for (const item of t.data) {
    if (typeof item !== "string") {
      throw new DeepboxError("Expected string array storage");
    }
  }
  return t.data;
}

export function toNumberArray(t: Tensor, label: string): number[] {
  const value = t.toArray();
  assertNumberArray(value, label);
  return value;
}

export function toNumberMatrix(t: Tensor, label: string): number[][] {
  const value = t.toArray();
  assertNumberMatrix(value, label);
  return value;
}

export function toStringArray(t: Tensor, label: string): string[] {
  const value = t.toArray();
  assertStringArray(value, label);
  return value;
}

export function toStringMatrix(t: Tensor, label: string): string[][] {
  const value = t.toArray();
  assertStringMatrix(value, label);
  return value;
}

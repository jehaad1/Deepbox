/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { IndexError, InvalidParameterError } from "../../core";

/** A single index or a start/end/step range for tensor slicing. */
export type SliceRange =
  | number
  | {
      readonly start?: number;
      readonly end?: number;
      readonly step?: number;
    };

/**
 * Resolve a single (possibly negative) index against a dimension size.
 *
 * @param index - Integer index; negative values count from the end
 * @param dim - Size of the dimension being indexed
 * @throws {InvalidParameterError} If `index` is not an integer
 * @throws {IndexError} If the index is out of bounds
 */
export function normalizeIndex(index: number, dim: number): number {
  if (!Number.isInteger(index)) {
    throw new InvalidParameterError(
      `slice index must be an integer; received ${index}`,
      "index",
      index
    );
  }
  const idx = index < 0 ? dim + index : index;
  if (idx < 0 || idx >= dim) {
    throw new IndexError(`index ${index} is out of bounds for dimension of size ${dim}`);
  }
  return idx;
}

/**
 * Validate a slice bound. Integers and +/-Infinity are accepted (infinite
 * bounds are clamped like any other out-of-range bound); NaN and fractional
 * values are rejected instead of silently producing a garbage view.
 */
function assertSliceBound(value: number | undefined, name: "start" | "end"): void {
  if (value === undefined) return;
  if (Number.isNaN(value) || (Number.isFinite(value) && !Number.isInteger(value))) {
    throw new InvalidParameterError(
      `slice ${name} must be an integer; received ${value}`,
      name,
      value
    );
  }
}

/**
 * Resolve a {@link SliceRange} against a dimension size using NumPy slicing
 * semantics: negative bounds count from the end, out-of-range bounds are
 * clamped, and a negative step walks backwards (`end` is exclusive and an
 * omitted `end` means "through index 0", returned as the internal bound -1).
 *
 * The returned `start`/`end` describe a half-open range; the number of
 * selected elements is `max(0, ceil((end - start) / step))`.
 *
 * @param range - A single index, or a `{ start, end, step }` range
 * @param dim - Size of the dimension being sliced
 * @throws {InvalidParameterError} If `step` is zero or any bound is not an integer
 * @throws {IndexError} If a single-index range is out of bounds
 */
export function normalizeRange(
  range: SliceRange,
  dim: number
): { start: number; end: number; step: number } {
  if (typeof range === "number") {
    const idx = normalizeIndex(range, dim);
    return { start: idx, end: idx + 1, step: 1 };
  }

  assertSliceBound(range.start, "start");
  assertSliceBound(range.end, "end");

  const step = range.step ?? 1;
  if (!Number.isInteger(step) || step === 0) {
    throw new InvalidParameterError(
      `slice step must be a non-zero integer; received ${step}`,
      "step",
      step
    );
  }

  if (step > 0) {
    const startRaw = range.start ?? 0;
    const endRaw = range.end ?? dim;

    const start = startRaw < 0 ? dim + startRaw : startRaw;
    const end = endRaw < 0 ? dim + endRaw : endRaw;

    const clampedStart = Math.min(Math.max(start, 0), dim);
    const clampedEnd = Math.min(Math.max(end, 0), dim);

    return { start: clampedStart, end: clampedEnd, step };
  }

  const startRaw = range.start ?? dim - 1;

  let start = startRaw < 0 ? dim + startRaw : startRaw;
  if (start >= dim) start = dim - 1;
  if (start < -1) start = -1;

  let end: number;
  if (range.end === undefined) {
    // Omitted end with a negative step means "through the beginning";
    // -1 is the internal exclusive bound one before index 0.
    end = -1;
  } else if (range.end < 0) {
    // An explicit negative end is dim-relative (NumPy semantics), including
    // end === -1 which means dim - 1, NOT the internal sentinel above.
    end = dim + range.end;
    if (end < -1) end = -1;
  } else {
    end = range.end;
    if (end >= dim) end = dim - 1;
  }

  return { start, end, step };
}

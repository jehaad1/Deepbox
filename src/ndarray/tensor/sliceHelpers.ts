import { IndexError, InvalidParameterError } from "../../core";

export type SliceRange =
  | number
  | {
      readonly start?: number;
      readonly end?: number;
      readonly step?: number;
    };

export function normalizeIndex(index: number, dim: number): number {
  const idx = index < 0 ? dim + index : index;
  if (idx < 0 || idx >= dim) {
    throw new IndexError(`index ${index} is out of bounds for dimension of size ${dim}`);
  }
  return idx;
}

export function normalizeRange(
  range: SliceRange,
  dim: number
): { start: number; end: number; step: number } {
  if (typeof range === "number") {
    const idx = normalizeIndex(range, dim);
    return { start: idx, end: idx + 1, step: 1 };
  }

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
  const endRaw = range.end ?? -1;

  let start = startRaw < 0 ? dim + startRaw : startRaw;
  if (start >= dim) start = dim - 1;
  if (start < -1) start = -1;

  let end: number;
  if (endRaw === -1) {
    end = -1;
  } else if (endRaw < 0) {
    end = dim + endRaw;
    if (end < -1) end = -1;
  } else {
    end = endRaw;
    if (end >= dim) end = dim - 1;
  }

  return { start, end, step };
}

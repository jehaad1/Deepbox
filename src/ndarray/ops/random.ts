/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  type Device,
  type DType,
  dtypeToTypedArrayCtor,
  InvalidParameterError,
  type Shape,
  shapeToSize,
  validateShape,
} from "../../core";
import { __fillUniform } from "../../random/random";
import { Tensor } from "../tensor/Tensor";

type NumericDType = Exclude<DType, "string">;

/** Uniform samples are drawn this many at a time to bound the scratch buffer. */
const DRAW_CHUNK = 1 << 14;

/**
 * Generate a tensor of Bernoulli samples scaled by a constant.
 *
 * Each element is independently drawn: it equals `scale` with probability
 * `(1 - p)` and `0` with probability `p`. This is the mask used by inverted
 * dropout: non-dropped elements are pre-scaled by `1 / (1 - p)`.
 *
 * Draws come from the library's shared random stream, so `setSeed` makes the
 * mask reproducible. For integer dtypes `scale` is converted like `astype`
 * does (truncated toward zero); `bool` masks hold 1 where `scale` is non-zero.
 *
 * @param shape  - Output tensor shape
 * @param p      - Probability of an element being zero (drop probability), in `[0, 1)`
 * @param scale  - Value assigned to kept elements (typically `1 / (1 - p)`), must be finite
 * @param dtype  - Numeric dtype for the output tensor
 * @param device - Target device
 * @returns Tensor of the given shape filled with `0` or `scale`
 * @throws {InvalidParameterError} If `p` is outside `[0, 1)` or `scale` is not finite
 *
 * @internal
 */
export function dropoutMask(
  shape: Shape,
  p: number,
  scale: number,
  dtype: NumericDType,
  device: Device
): Tensor {
  if (!Number.isFinite(p) || p < 0 || p >= 1) {
    throw new InvalidParameterError("p must be in [0, 1)", "p", p);
  }
  if (!Number.isFinite(scale)) {
    throw new InvalidParameterError("scale must be finite", "scale", scale);
  }
  validateShape(shape);

  const size = shapeToSize(shape);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(size);
  const keptNumber = dtype === "bool" ? (scale !== 0 ? 1 : 0) : scale;
  const keptBig = BigInt(Math.trunc(scale));

  const draws = new Float64Array(Math.min(size, DRAW_CHUNK));
  for (let start = 0; start < size; start += DRAW_CHUNK) {
    const count = Math.min(DRAW_CHUNK, size - start);
    __fillUniform(draws, count);
    // keep with probability 1 - p: P(u >= p) = 1 - p for u uniform in [0, 1)
    if (data instanceof BigInt64Array) {
      for (let i = 0; i < count; i++) {
        data[start + i] = (draws[i] as number) >= p ? keptBig : 0n;
      }
    } else {
      for (let i = 0; i < count; i++) {
        data[start + i] = (draws[i] as number) >= p ? keptNumber : 0;
      }
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape,
    dtype,
    device,
  });
}

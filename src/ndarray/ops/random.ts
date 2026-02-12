import {
  type Device,
  type DType,
  dtypeToTypedArrayCtor,
  InvalidParameterError,
  type Shape,
  shapeToSize,
} from "../../core";
import { Tensor } from "../tensor/Tensor";

type NumericDType = Exclude<DType, "string">;

/**
 * Generate a tensor of Bernoulli samples scaled by a constant.
 *
 * Each element is independently drawn: it equals `scale` with probability
 * `(1 - p)` and `0` with probability `p`. This is the mask used by inverted
 * dropout: non-dropped elements are pre-scaled by `1 / (1 - p)`.
 *
 * @param shape  - Output tensor shape
 * @param p      - Probability of an element being zero (drop probability)
 * @param scale  - Value assigned to kept elements (typically `1 / (1 - p)`)
 * @param dtype  - Numeric dtype for the output tensor
 * @param device - Target device
 * @returns Tensor of the given shape filled with `0` or `scale`
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

  const size = shapeToSize(shape);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(size);

  if (data instanceof BigInt64Array) {
    const scaleBig = BigInt(Math.round(scale));
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() > p ? scaleBig : 0n;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() > p ? scale : 0;
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape,
    dtype,
    device,
  });
}

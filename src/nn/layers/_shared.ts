/**
 * Internal helpers shared by the layers: the automatic gradient tracking rule,
 * parameter dtype resolution and PyTorch-style uniform weight initialization.
 *
 * Nothing in this file is part of the public API.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox documentation}
 */

import { type Device, getConfig, InvalidParameterError } from "../../core";
import { type AnyTensor, GradTensor, parameter, zeros } from "../../ndarray";
import { Tensor } from "../../ndarray/tensor/Tensor";
import { __fillUniform } from "../../random/random";

/** Floating point dtypes a layer can store its parameters in. */
export type LayerDType = "float32" | "float64";

/** Leaf that requires grad, created once and only read by {@link isGradEnabled}. */
let probe: GradTensor | undefined;

/**
 * Whether gradient tracking is on (it is off inside `noGrad`).
 *
 * An operation on a leaf that requires grad records a graph node only while tracking is
 * on, so the result of a probe operation tells the current mode without touching the
 * autograd module.
 */
export function isGradEnabled(): boolean {
  probe ??= parameter(zeros([], { dtype: "float32" }));
  return probe.requiresGrad && probe.neg().requiresGrad;
}

/**
 * Wrap a plain tensor as a leaf `GradTensor` that does not require gradients, so a layer
 * can run one `GradTensor` code path. A `GradTensor` is returned unchanged.
 */
export function toGradInput(x: AnyTensor): GradTensor {
  return GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x, { requiresGrad: false });
}

/** True when none of `inputs` is a `GradTensor`. */
export function allPlain(...inputs: readonly AnyTensor[]): boolean {
  for (const input of inputs) {
    if (GradTensor.isGradTensor(input)) return false;
  }
  return true;
}

/**
 * Finish a layer's forward pass. The result is returned as a `GradTensor` when it tracks
 * gradients (a weight that requires grad was used and tracking is on) or when the caller
 * passed a `GradTensor`. Otherwise a plain input gets a plain `Tensor` back.
 *
 * @param out - Result computed on the `GradTensor` path
 * @param plainInput - Whether every input of the layer was a plain tensor
 */
export function settle(out: GradTensor, plainInput: boolean): AnyTensor {
  return plainInput && !out.requiresGrad ? out.tensor : out;
}

/**
 * Resolve the dtype option of a layer: `float32` or `float64`, or the global default
 * dtype (`float32` unless changed with `configure`) when it is not given.
 *
 * @throws {InvalidParameterError} If `dtype` is neither `float32` nor `float64`
 */
export function resolveLayerDtype(dtype: LayerDType | undefined): LayerDType {
  if (dtype === undefined) {
    return getConfig().defaultDtype === "float64" ? "float64" : "float32";
  }
  const resolved: string = dtype;
  if (resolved !== "float32" && resolved !== "float64") {
    throw new InvalidParameterError("dtype must be 'float32' or 'float64'", "dtype", dtype);
  }
  return resolved;
}

/**
 * Tensor of the given shape drawn from the uniform distribution `U(-bound, bound)`, using
 * the global generator (`manualSeed` makes it reproducible). This is PyTorch's default
 * initialization of Linear, Conv and recurrent layers with `bound = 1 / sqrt(fan)`.
 */
export function uniformTensor(
  shape: readonly number[],
  bound: number,
  options: { readonly dtype: LayerDType; readonly device?: Device }
): Tensor {
  let size = 1;
  for (const dim of shape) size *= dim;
  const draws = new Float64Array(size);
  __fillUniform(draws, size);
  const data = options.dtype === "float64" ? new Float64Array(size) : new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = (2 * (draws[i] as number) - 1) * bound;
  }
  return Tensor.fromTypedArray({
    data,
    shape: [...shape],
    dtype: options.dtype,
    device: options.device ?? getConfig().defaultDevice,
  });
}

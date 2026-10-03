/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import {
  type DType,
  DTypeError,
  getBigIntElement,
  getElementAsNumber,
  getNumericElement,
  InvalidParameterError,
  ShapeError,
  shapesEqual,
} from "../../core";
import { type AnyTensor, customOp, GradTensor, logSoftmaxGrad, Tensor } from "../../ndarray";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";

/** How per-sample losses are combined: average, total, or kept per sample. */
export type LossReduction = "mean" | "sum" | "none";

/**
 * Options for {@link crossEntropyLoss}.
 */
export interface CrossEntropyLossOptions {
  /** `"mean"` (default), `"sum"`, or `"none"` to return one loss per sample. */
  readonly reduction?: LossReduction;
  /**
   * Per-class rescaling weights, length `n_classes`. With `"mean"` reduction and
   * class-index targets the loss is divided by the sum of the target classes'
   * weights, as in PyTorch.
   */
  readonly weight?: Tensor | readonly number[];
  /**
   * Class index to skip (class-index targets only). Samples with this target
   * contribute no loss and no gradient and are not counted by `"mean"`.
   * Ignored for probability targets. By default no index is ignored.
   */
  readonly ignoreIndex?: number;
  /**
   * Label smoothing in [0, 1]. The target distribution becomes
   * `(1 - labelSmoothing) * target + labelSmoothing / n_classes`.
   */
  readonly labelSmoothing?: number;
}

/**
 * Options for {@link binaryCrossEntropyWithLogitsLoss}.
 */
export interface BinaryCrossEntropyWithLogitsOptions {
  /** `"mean"` (default), `"sum"`, or `"none"` to return the element-wise loss. */
  readonly reduction?: LossReduction;
  /**
   * Weight of the positive class. A number, or a tensor that broadcasts against
   * the logits (for example one entry per label in a multi-label problem).
   */
  readonly posWeight?: number | Tensor;
  /** Element-wise rescaling weight that broadcasts against the logits. */
  readonly weight?: Tensor;
}

/** @internal */
export function validateReduction(reduction: LossReduction, context: string): void {
  if (reduction !== "mean" && reduction !== "sum" && reduction !== "none") {
    throw new InvalidParameterError(
      `${context} reduction must be 'mean', 'sum', or 'none'`,
      "reduction",
      reduction
    );
  }
}

function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/** @internal */
export type NumericDType = Exclude<DType, "string">;

/**
 * Numeric dtype of `t`; string tensors are rejected.
 * @internal
 */
export function numericDtype(t: GradTensor, context: string): NumericDType {
  const dtype = t.dtype;
  if (dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  return dtype;
}

/**
 * Cast `t` to `dtype` when they differ (differentiable).
 * @internal
 */
export function asDtype(t: GradTensor, dtype: NumericDType): GradTensor {
  return t.dtype === dtype ? t : t.astype(dtype);
}

/**
 * Elements of `t` in row-major order as numbers. Contiguous float/int tensors are
 * returned as a view of their storage (no copy); strided views and int64 tensors
 * are gathered into a float64 array.
 * @internal
 */
export function numbersOf(t: Tensor, context: string): ArrayLike<number> {
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  const size = t.size;
  if (!(data instanceof BigInt64Array) && isContiguous(t.shape, t.strides)) {
    return data.subarray(t.offset, t.offset + size);
  }
  const out = new Float64Array(size);
  const logical = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);
  for (let i = 0; i < size; i++) {
    const offset = contiguous ? t.offset + i : offsetFromFlatIndex(i, logical, t.strides, t.offset);
    out[i] = getElementAsNumber(data, offset);
  }
  return out;
}

/**
 * Read the value of a single-element tensor as a number.
 * @internal
 */
export function scalarValue(t: Tensor, context: string): number {
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  if (data instanceof BigInt64Array) {
    return Number(getBigIntElement(data, t.offset));
  }
  return getNumericElement(data, t.offset);
}

/**
 * Read class indices from a 1D target tensor. Entries equal to `ignoreIndex`
 * are returned as -1; every other entry must be an integer in `[0, numClasses)`.
 * @internal
 */
export function readClassIndices(
  indices: Tensor,
  numClasses: number,
  ignoreIndex: number | undefined
): Int32Array {
  const data = indices.data;
  if (Array.isArray(data)) {
    throw new DTypeError("crossEntropyLoss target indices must be numeric");
  }

  const n = indices.size;
  const out = new Int32Array(n);
  const stride0 = indices.strides[0] ?? 0;
  const base = indices.offset;

  for (let i = 0; i < n; i++) {
    const offset = base + i * stride0;
    let idx: number;
    if (data instanceof BigInt64Array) {
      const raw = getBigIntElement(data, offset);
      const asNumber = Number(raw);
      if (!Number.isSafeInteger(asNumber)) {
        throw new InvalidParameterError(
          `Class index ${raw.toString()} exceeds safe integer range`,
          "target",
          raw.toString()
        );
      }
      idx = asNumber;
    } else {
      idx = Number(getNumericElement(data, offset));
    }

    if (ignoreIndex !== undefined && idx === ignoreIndex) {
      out[i] = -1;
      continue;
    }

    if (!Number.isFinite(idx) || !Number.isInteger(idx)) {
      throw new InvalidParameterError(`Class index ${idx} is not a valid integer`, "target", idx);
    }

    if (idx < 0 || idx >= numClasses) {
      throw new InvalidParameterError(
        `Class index ${idx} out of range [0, ${numClasses})`,
        "target",
        idx
      );
    }
    out[i] = idx;
  }
  return out;
}

/**
 * Validate and read the `weight` option as a float64 vector of length `numClasses`.
 * @internal
 */
export function readClassWeights(
  weight: Tensor | readonly number[],
  numClasses: number
): Float64Array {
  let values: Float64Array;
  if (weight instanceof Tensor) {
    if (weight.ndim !== 1) {
      throw new ShapeError(`weight must be 1-dimensional; got ${weight.ndim}D`);
    }
    values = Float64Array.from(numbersOf(weight, "crossEntropyLoss"));
  } else {
    values = Float64Array.from(weight);
  }
  if (values.length !== numClasses) {
    throw new ShapeError(
      `weight must have one entry per class (${numClasses}); got ${values.length}`
    );
  }
  for (const v of values) {
    if (Number.isNaN(v)) {
      throw new InvalidParameterError("weight must not contain NaN", "weight", Array.from(values));
    }
  }
  return values;
}

/**
 * How log-probability `[i, c]` enters sample `i`'s loss for class-index targets:
 * `coefficient(i, c) = smoothing / C * classWeight[c] + [c == indices[i]] * (1 - smoothing) * targetWeight[i]`,
 * and 0 for ignored rows (`indices[i] < 0`).
 */
type RowSpec = {
  readonly indices: Int32Array;
  readonly targetWeight: Float64Array;
  readonly classWeights: Float64Array | null;
  readonly labelSmoothing: number;
};

function rowCoefficient(spec: RowSpec, i: number, c: number, nClasses: number): number {
  const y = spec.indices[i] ?? -1;
  if (y < 0) return 0;
  const target = c === y ? (1 - spec.labelSmoothing) * (spec.targetWeight[i] ?? 1) : 0;
  if (spec.labelSmoothing === 0) return target;
  const cw = spec.classWeights ? (spec.classWeights[c] ?? 1) : 1;
  return (spec.labelSmoothing / nClasses) * cw + target;
}

/**
 * Per-row weighted sum of log-probabilities: `out[i] = -sum_c coefficient(i, c) * logProbs[i, c]`.
 *
 * Entries with a zero coefficient are skipped, so `-Infinity` log-probabilities of
 * classes that carry no weight (masked logits) do not turn the loss into NaN.
 * The backward pass returns `-coefficient(i, c) * grad[i]` for the log-probabilities.
 */
function weightedNegRowSum(
  logProbs: GradTensor,
  spec: RowSpec,
  nSamples: number,
  nClasses: number
): GradTensor {
  const lp = numbersOf(logProbs.tensor, "crossEntropyLoss");
  const out = new Float64Array(nSamples);
  const sparse = spec.labelSmoothing === 0;
  for (let i = 0; i < nSamples; i++) {
    const y = spec.indices[i] ?? -1;
    if (y < 0) continue;
    const row = i * nClasses;
    let acc = 0;
    if (sparse) {
      const w = spec.targetWeight[i] ?? 1;
      if (w !== 0) acc = w * (lp[row + y] ?? 0);
    } else {
      for (let c = 0; c < nClasses; c++) {
        const w = rowCoefficient(spec, i, c, nClasses);
        if (w !== 0) acc += w * (lp[row + c] ?? 0);
      }
    }
    out[i] = acc === 0 ? 0 : -acc;
  }

  const dtype: DType = isFloatDType(logProbs.dtype) ? logProbs.dtype : "float64";
  const device = logProbs.tensor.device;
  const toDtype = (t: Tensor): Tensor => (t.dtype === dtype ? t : t.astype(dtype));
  const outTensor = toDtype(
    Tensor.fromTypedArray({ data: out, shape: [nSamples], dtype: "float64", device })
  );

  return customOp(outTensor, [
    [
      logProbs,
      (g: Tensor): Tensor => {
        const go = numbersOf(g, "crossEntropyLoss");
        const gi = new Float64Array(nSamples * nClasses);
        for (let i = 0; i < nSamples; i++) {
          const y = spec.indices[i] ?? -1;
          if (y < 0) continue;
          const row = i * nClasses;
          const gv = go[i] ?? 0;
          if (sparse) {
            const w = spec.targetWeight[i] ?? 1;
            if (w !== 0) gi[row + y] = -w * gv;
          } else {
            for (let c = 0; c < nClasses; c++) {
              const w = rowCoefficient(spec, i, c, nClasses);
              if (w !== 0) gi[row + c] = -w * gv;
            }
          }
        }
        return toDtype(
          Tensor.fromTypedArray({
            data: gi,
            shape: [nSamples, nClasses],
            dtype: "float64",
            device,
          })
        );
      },
    ],
  ]);
}

/**
 * Combine per-sample losses according to `reduction`. For `"mean"` the sum is divided
 * by `denominator`; a zero denominator (for example every sample ignored) gives NaN.
 * @internal
 */
export function reduceSampleLoss(
  sampleLoss: GradTensor,
  reduction: LossReduction,
  denominator: number
): GradTensor {
  if (reduction === "none") return sampleLoss;
  if (reduction === "sum") return sampleLoss.sum();
  return sampleLoss
    .sum()
    .div(GradTensor.scalar(denominator, { dtype: numericDtype(sampleLoss, "loss reduction") }));
}

/**
 * Negative log-likelihood of class-index targets given log-probabilities of shape
 * (n_samples, n_classes), with optional class weights, ignored index and label
 * smoothing. Differentiable with respect to `logProbs`.
 * @internal
 */
export function classIndexLoss(
  logProbs: GradTensor,
  targets: Tensor,
  options: {
    readonly reduction: LossReduction;
    readonly weight: Float64Array | null;
    readonly ignoreIndex: number | undefined;
    readonly labelSmoothing: number;
  }
): GradTensor {
  const nSamples = logProbs.shape[0] ?? 0;
  const nClasses = logProbs.shape[1] ?? 0;
  const { weight: classWeights, labelSmoothing } = options;
  const indices = readClassIndices(targets, nClasses, options.ignoreIndex);

  const targetWeight = new Float64Array(nSamples).fill(1);
  let denominator = 0;
  for (let i = 0; i < nSamples; i++) {
    const y = indices[i] ?? -1;
    if (y < 0) continue;
    const wy = classWeights ? (classWeights[y] ?? 1) : 1;
    targetWeight[i] = wy;
    denominator += wy;
  }
  const sampleLoss = weightedNegRowSum(
    logProbs,
    { indices, targetWeight, classWeights, labelSmoothing },
    nSamples,
    nClasses
  );
  return reduceSampleLoss(sampleLoss, options.reduction, denominator);
}

/**
 * Cross Entropy Loss.
 *
 * Computes the cross entropy between logits and targets. Commonly used for
 * multi-class classification. The softmax is folded into the loss through a
 * log-sum-exp, so large logits do not overflow.
 *
 * Targets are either integer class indices of shape `(n_samples,)` or class
 * probabilities (for example one-hot) of shape `(n_samples, n_classes)`.
 *
 * **Formula** (probability targets): `L_i = -sum_c target[i, c] * log_softmax(input)[i, c]`,
 * averaged over the batch by default. For class-index targets this is
 * `-log_softmax(input)[i, target[i]]`.
 *
 * Passing plain tensors returns a number (or a tensor for `reduction: "none"`);
 * passing a GradTensor input or target returns a GradTensor that supports `.backward()`.
 *
 * @param input - Logits of shape (n_samples, n_classes)
 * @param target - Class indices of shape (n_samples,), or probabilities of shape (n_samples, n_classes)
 * @param options - `reduction`, per-class `weight`, `ignoreIndex` and `labelSmoothing`
 * @returns The reduced loss, or the per-sample losses of shape (n_samples,) for `reduction: "none"`
 * @throws {ShapeError} If shapes are inconsistent
 * @throws {InvalidParameterError} If a class index is not an integer in range or an option is invalid
 *
 * @example
 * ```ts
 * import { crossEntropyLoss } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const logits = tensor([[2.0, 0.5, 0.1], [0.1, 1.8, 0.1]]);
 * const targets = tensor([0, 1]);
 * const loss = crossEntropyLoss(logits, targets); // number
 * const perSample = crossEntropyLoss(logits, targets, { reduction: 'none' }); // Tensor (2,)
 * ```
 */
export function crossEntropyLoss(
  input: Tensor,
  target: Tensor,
  options: CrossEntropyLossOptions & { readonly reduction: "none" }
): Tensor;
export function crossEntropyLoss(
  input: Tensor,
  target: Tensor,
  options?: CrossEntropyLossOptions
): number;
export function crossEntropyLoss(
  input: GradTensor,
  target: AnyTensor,
  options?: CrossEntropyLossOptions
): GradTensor;
export function crossEntropyLoss(
  input: AnyTensor,
  target: AnyTensor,
  options?: CrossEntropyLossOptions
): number | Tensor | GradTensor;
export function crossEntropyLoss(
  input: AnyTensor,
  target: AnyTensor,
  options: CrossEntropyLossOptions = {}
): number | Tensor | GradTensor {
  const reduction = options.reduction ?? "mean";
  validateReduction(reduction, "crossEntropyLoss");
  const labelSmoothing = options.labelSmoothing ?? 0;
  if (
    typeof labelSmoothing !== "number" ||
    !Number.isFinite(labelSmoothing) ||
    labelSmoothing < 0 ||
    labelSmoothing > 1
  ) {
    throw new InvalidParameterError(
      `labelSmoothing must be in [0, 1]; got ${String(labelSmoothing)}`,
      "labelSmoothing",
      labelSmoothing
    );
  }
  const ignoreIndex = options.ignoreIndex;
  if (ignoreIndex !== undefined && !Number.isInteger(ignoreIndex)) {
    throw new InvalidParameterError(
      `ignoreIndex must be an integer; got ${String(ignoreIndex)}`,
      "ignoreIndex",
      ignoreIndex
    );
  }

  const yPred = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
  const targetIsGrad = GradTensor.isGradTensor(target);
  // Target usually doesn't require grad, but if it's soft labels (distillation), it might.
  const yTrue = GradTensor.isGradTensor(target)
    ? target
    : GradTensor.fromTensor(target, { requiresGrad: false });

  if (yPred.ndim !== 2) {
    throw new ShapeError(`Input must be 2-dimensional (batch, classes); got ${yPred.ndim}`);
  }

  const nSamples = yPred.shape[0] ?? 0;
  const nClasses = yPred.shape[1] ?? 0;
  if (nClasses === 0) {
    throw new ShapeError("Input must have at least one class; got 0 columns");
  }

  const classWeights =
    options.weight === undefined ? null : readClassWeights(options.weight, nClasses);

  // Compute Log Softmax
  const logProbs = logSoftmaxGrad(yPred, 1);

  let loss: GradTensor;

  if (yTrue.ndim === 1) {
    // Class indices are discrete labels, so no gradient flows through them.
    if (yTrue.shape[0] !== nSamples) {
      throw new ShapeError(
        `Target must have same number of samples as input; got ${yTrue.shape[0]} and ${nSamples}`
      );
    }
    loss = classIndexLoss(logProbs, yTrue.tensor, {
      reduction,
      weight: classWeights,
      ignoreIndex,
      labelSmoothing,
    });
  } else if (yTrue.ndim === 2) {
    if (yTrue.shape[0] !== nSamples || yTrue.shape[1] !== nClasses) {
      throw new ShapeError(
        "Target must be 1-dimensional class indices or have the same shape as input"
      );
    }
    const dtype = numericDtype(logProbs, "crossEntropyLoss");
    let probs = asDtype(yTrue, dtype);
    if (labelSmoothing > 0) {
      probs = probs
        .mul(GradTensor.scalar(1 - labelSmoothing, { dtype }))
        .add(GradTensor.scalar(labelSmoothing / nClasses, { dtype }));
    }
    let weighted = logProbs.mul(probs);
    if (classWeights) {
      const wTensor = Tensor.fromTypedArray({
        data: dtype === "float64" ? classWeights : Float32Array.from(classWeights),
        shape: [nClasses],
        dtype: dtype === "float64" ? "float64" : "float32",
        device: yPred.tensor.device,
      });
      weighted = weighted.mul(asDtype(GradTensor.fromTensor(wTensor), dtype));
    }
    // Sum over classes (dim 1) -> (N,), negated so a good fit has a small loss.
    loss = reduceSampleLoss(weighted.sum(1).neg(), reduction, nSamples);
  } else {
    throw new ShapeError(`Target must be 1D (indices) or 2D (probs); got ${yTrue.ndim}D`);
  }

  if (!GradTensor.isGradTensor(input) && !targetIsGrad) {
    return reduction === "none" ? loss.tensor : scalarValue(loss.tensor, "crossEntropyLoss");
  }
  return loss;
}

/**
 * Binary Cross Entropy Loss with logits.
 *
 * Combines a sigmoid and binary cross entropy in one numerically stable
 * expression, `max(x, 0) - x * z + log(1 + exp(-|x|))`, which neither overflows
 * for large positive logits nor loses the loss for large negative ones.
 *
 * `input` and `target` must have the same shape (any number of dimensions, for
 * example `(n_samples, n_labels)` for multi-label problems). As a convenience a
 * `(n_samples,)` tensor may be paired with a `(n_samples, 1)` tensor. A target of
 * another float or integer dtype is converted to the dtype of `input`.
 *
 * Passing plain tensors returns a number (or a tensor for `reduction: "none"`);
 * passing a GradTensor input or target returns a GradTensor.
 *
 * @param input - Predicted logits
 * @param target - True binary labels (0 or 1, or probabilities in [0, 1]) of the same shape as input
 * @param options - `reduction`, `posWeight` and element-wise `weight`
 * @returns The reduced loss, or the element-wise loss for `reduction: "none"`
 * @throws {ShapeError} If input and target shapes do not match
 *
 * @example
 * ```ts
 * import { binaryCrossEntropyWithLogitsLoss } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const loss = binaryCrossEntropyWithLogitsLoss(tensor([0.5, -1.2, 2.0]), tensor([1, 0, 1]));
 * ```
 */
export function binaryCrossEntropyWithLogitsLoss(
  input: Tensor,
  target: Tensor,
  options: BinaryCrossEntropyWithLogitsOptions & { readonly reduction: "none" }
): Tensor;
export function binaryCrossEntropyWithLogitsLoss(
  input: Tensor,
  target: Tensor,
  options?: BinaryCrossEntropyWithLogitsOptions
): number;
export function binaryCrossEntropyWithLogitsLoss(
  input: GradTensor,
  target: AnyTensor,
  options?: BinaryCrossEntropyWithLogitsOptions
): GradTensor;
export function binaryCrossEntropyWithLogitsLoss(
  input: AnyTensor,
  target: AnyTensor,
  options?: BinaryCrossEntropyWithLogitsOptions
): number | Tensor | GradTensor;
export function binaryCrossEntropyWithLogitsLoss(
  input: AnyTensor,
  target: AnyTensor,
  options: BinaryCrossEntropyWithLogitsOptions = {}
): number | Tensor | GradTensor {
  const reduction = options.reduction ?? "mean";
  validateReduction(reduction, "binaryCrossEntropyWithLogitsLoss");

  const yPred = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
  const yTrue = GradTensor.isGradTensor(target)
    ? target
    : GradTensor.fromTensor(target, { requiresGrad: false });

  let pred = yPred;
  let truth = yTrue;

  if (!shapesEqual(pred.shape, truth.shape)) {
    // Support (N,) paired with (N, 1)
    if (pred.ndim !== 1 && pred.ndim !== 2) {
      throw new ShapeError("Input must be 1 or 2-dimensional");
    }
    if (truth.ndim !== 1 && truth.ndim !== 2) {
      throw new ShapeError("Target must be 1 or 2-dimensional");
    }

    if (pred.ndim === 1) {
      pred = pred.reshape([pred.shape[0] ?? 0, 1]);
    }
    if (truth.ndim === 1) {
      truth = truth.reshape([truth.shape[0] ?? 0, 1]);
    }

    if (pred.ndim !== 2 || pred.shape[1] !== 1) {
      throw new ShapeError("Input must have shape (N,) or (N, 1), or the same shape as the target");
    }
    if (truth.ndim !== 2 || truth.shape[1] !== 1) {
      throw new ShapeError(`Target must be 1-dimensional or have shape (N, 1)`);
    }
    if ((pred.shape[0] ?? 0) !== (truth.shape[0] ?? 0)) {
      throw new ShapeError(
        `Batch size mismatch: input has ${pred.shape[0] ?? 0} rows, target has ${truth.shape[0] ?? 0}`
      );
    }
  }

  const predDtype = pred.dtype;
  if (predDtype === "string") {
    throw new DTypeError("Binary cross entropy does not support string dtype");
  }
  truth = asDtype(truth, predDtype);

  // Numerically stable formulation of binary cross entropy with logits:
  //   loss = max(x, 0) - x * z + log(1 + exp(-|x|))
  // where x is the logit and z is the target. This avoids overflow in exp(x)
  // for large positive logits and underflow in log for large negative ones.
  //
  // |x| is relu(x) + relu(-x), whose gradient is 0 at x = 0. max(x, 0) is written as
  // (x + |x|) / 2 so that the gradient at x = 0 is 1/2 and the total gradient
  // sigmoid(x) - z is exact there too (zero-initialised logits are common).
  const reluNeg = pred.neg().relu();
  const absPred = pred.relu().add(reluNeg);
  const expNegAbs = absPred.neg().exp();
  const one = GradTensor.scalar(1, { dtype: predDtype });
  const half = GradTensor.scalar(0.5, { dtype: predDtype });
  const logTerm = one.add(expNegAbs).log();

  let loss: GradTensor;
  if (options.posWeight === undefined) {
    // max(x, 0) - x * z + log(1 + exp(-|x|))
    loss = pred.add(absPred).mul(half).sub(pred.mul(truth)).add(logTerm);
  } else {
    // (1 - z) * x + (1 + (pw - 1) * z) * softplus(-x), with
    // softplus(-x) = max(-x, 0) + log(1 + exp(-|x|)) = (|x| - x) / 2 + log(1 + exp(-|x|))
    const pw =
      typeof options.posWeight === "number"
        ? GradTensor.scalar(options.posWeight, { dtype: predDtype })
        : asDtype(GradTensor.fromTensor(options.posWeight), predDtype);
    const logWeight = pw.sub(one).mul(truth).add(one);
    const softplusNeg = absPred.sub(pred).mul(half).add(logTerm);
    loss = pred.sub(pred.mul(truth)).add(logWeight.mul(softplusNeg));
  }
  if (options.weight !== undefined) {
    loss = loss.mul(asDtype(GradTensor.fromTensor(options.weight), predDtype));
  }

  if (reduction === "sum") {
    loss = loss.sum();
  } else if (reduction === "mean") {
    loss = loss.mean();
  }

  if (!GradTensor.isGradTensor(input) && !GradTensor.isGradTensor(target)) {
    return reduction === "none"
      ? loss.tensor
      : scalarValue(loss.tensor, "binaryCrossEntropyWithLogitsLoss");
  }
  return loss;
}

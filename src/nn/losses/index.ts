/**
 * Neural network loss functions (`deepbox/nn` losses barrel).
 *
 * @see {@link https://deepbox.dev/docs/nn-losses | Deepbox documentation}
 */
import { DTypeError, InvalidParameterError, ShapeError, shapesEqual } from "../../core";
import type { AnyTensor } from "../../ndarray";
import {
  abs,
  customOp,
  div,
  GradTensor,
  mean,
  pow,
  reshape,
  sqrt,
  sub,
  sum,
  Tensor,
  tensor,
} from "../../ndarray";
import {
  asDtype,
  classIndexLoss,
  DEFAULT_IGNORE_INDEX,
  flattenClassLayout,
  type LossReduction,
  numbersOf,
  numericDtype,
  readClassWeights,
  scalarValue,
  validateReduction,
} from "./cross_entropy";

export type {
  BinaryCrossEntropyWithLogitsOptions,
  CrossEntropyLossOptions,
  LossReduction,
} from "./cross_entropy";
export {
  binaryCrossEntropyWithLogitsLoss,
  crossEntropyLoss,
} from "./cross_entropy";

/** Options for {@link nllLoss}. */
export interface NllLossOptions {
  /** `"mean"` (default), `"sum"`, or `"none"` to return one loss per sample. */
  readonly reduction?: LossReduction;
  /**
   * Per-class rescaling weights, length `n_classes`. With `"mean"` reduction the loss is
   * divided by the sum of the target classes' weights, as in PyTorch.
   */
  readonly weight?: Tensor | readonly number[];
  /**
   * Class index to skip: such samples add no loss and are not counted by `"mean"`
   * (default: -100, as in PyTorch).
   */
  readonly ignoreIndex?: number;
}

/** Options for {@link gaussianNLLLoss}. */
export interface GaussianNLLLossOptions {
  /** Include the constant `0.5 * log(2 * pi)` term (default: false). */
  readonly full?: boolean;
  /** Variances below this value are clamped to it (default: 1e-6). */
  readonly eps?: number;
  /** `"mean"` (default), `"sum"`, or `"none"` for the element-wise loss. */
  readonly reduction?: LossReduction;
}

/** Options for {@link poissonNLLLoss}. */
export interface PoissonNLLLossOptions {
  /** Treat `input` as `log(rate)` (default: true) or as the rate itself. */
  readonly logInput?: boolean;
  /** Add the Stirling approximation term for targets above 1 (default: false). */
  readonly full?: boolean;
  /** Small value added inside the logarithm when `logInput` is false (default: 1e-8). */
  readonly eps?: number;
  /** `"mean"` (default), `"sum"`, or `"none"` for the element-wise loss. */
  readonly reduction?: LossReduction;
}

/** Options for {@link ctcLoss}. */
export interface CtcLossOptions {
  /** Index of the blank label (default: 0). */
  readonly blank?: number;
  /**
   * `"mean"` (default) divides each sample's loss by its target length (at least 1) and
   * averages over the batch, as PyTorch does. `"sum"` adds the raw losses; `"none"`
   * returns one raw loss per sample.
   */
  readonly reduction?: LossReduction;
  /** Replace infinite losses (inputs too short for the target) with zero (default: false). */
  readonly zeroInfinity?: boolean;
}

function ensureSameShape(a: Tensor, b: Tensor, context: string): void {
  if (!shapesEqual(a.shape, b.shape)) {
    throw new ShapeError(`Shape mismatch in ${context}: [${a.shape}] vs [${b.shape}]`);
  }
}

function alignShapes(a: Tensor, b: Tensor): [Tensor, Tensor] {
  if (shapesEqual(a.shape, b.shape)) return [a, b];
  if (a.size === b.size) {
    if (a.ndim > b.ndim) return [reshape(a, b.shape), b];
    if (b.ndim > a.ndim) return [a, reshape(b, a.shape)];
  }
  return [a, b];
}

function ensureNumeric(t: Tensor, context: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
}

function toTensor(t: AnyTensor): Tensor {
  return GradTensor.isGradTensor(t) ? t.tensor : t;
}

/**
 * Prepare the operands of an autograd loss: targets become a non-differentiable
 * GradTensor of the prediction dtype, and operands of equal size but different rank
 * (for example `(N, 1)` and `(N,)`) are reshaped to the lower rank instead of being
 * broadcast into an `(N, N)` matrix.
 */
function gradOperands(
  pred: GradTensor,
  targets: AnyTensor,
  context: string
): [GradTensor, GradTensor] {
  const dtype = numericDtype(pred, context);
  let tgt = GradTensor.isGradTensor(targets)
    ? targets
    : GradTensor.fromTensor(targets, { requiresGrad: false });
  numericDtype(tgt, context);
  tgt = asDtype(tgt, dtype);
  let p = pred;
  if (!shapesEqual(p.shape, tgt.shape) && p.size === tgt.size) {
    if (p.ndim > tgt.ndim) p = p.reshape([...tgt.shape]);
    else if (tgt.ndim > p.ndim) tgt = tgt.reshape([...p.shape]);
  }
  return [p, tgt];
}

/** Float dtype of a loss computed from `t`: float64 stays float64, everything else is float32. */
function lossDtypeOf(t: Tensor): "float32" | "float64" {
  return t.dtype === "float64" ? "float64" : "float32";
}

function newLossBuffer(dtype: "float32" | "float64", size: number): Float32Array | Float64Array {
  return dtype === "float64" ? new Float64Array(size) : new Float32Array(size);
}

function reduceLoss(loss: Tensor, reduction: LossReduction): Tensor {
  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
}

function reduceGradLoss(loss: GradTensor, reduction: LossReduction): GradTensor {
  if (reduction === "none") return loss;
  if (reduction === "sum") return loss.sum();
  return loss.mean();
}

/** True when any operand is a GradTensor, so the loss must be built from autograd ops. */
function anyGrad(...operands: readonly AnyTensor[]): boolean {
  return operands.some((operand) => GradTensor.isGradTensor(operand));
}

/** Lower bound on the denominator of the binary cross-entropy gradient, as in PyTorch. */
const BCE_GRAD_EPSILON = 1e-12;

/** Natural logarithm clamped at -100, so a hard 0 or 1 prediction gives a finite loss. */
function bceLog(x: number): number {
  return Math.max(Math.log(x), -100);
}

/** Reject probabilities outside [0, 1] (NaN included). */
function checkProbabilities(p: ArrayLike<number>): void {
  for (let i = 0; i < p.length; i++) {
    const v = p[i] ?? Number.NaN;
    if (!(v >= 0 && v <= 1)) {
      throw new InvalidParameterError(
        `binaryCrossEntropyLoss predictions must be probabilities in [0, 1]; got ${v} at index ${i}. ` +
          "Apply a sigmoid first, or use binaryCrossEntropyWithLogitsLoss for logits.",
        "predictions",
        v
      );
    }
  }
}

/** Element-wise `-(t * log(p) + (1 - t) * log(1 - p))` with both logarithms clamped at -100. */
function bceElementwise(
  p: ArrayLike<number>,
  t: ArrayLike<number>,
  dtype: "float32" | "float64"
): Float32Array | Float64Array {
  const out = newLossBuffer(dtype, p.length);
  for (let i = 0; i < p.length; i++) {
    const pi = p[i] ?? 0;
    const ti = t[i] ?? 0;
    out[i] = -(ti * bceLog(pi) + (1 - ti) * bceLog(1 - pi));
  }
  return out;
}

/** Wrap `t` as a GradTensor (a constant when it is a plain tensor) cast to the float `dtype`. */
function gradOperand(t: AnyTensor, dtype: "float32" | "float64"): GradTensor {
  const g = GradTensor.isGradTensor(t) ? t : GradTensor.fromTensor(t, { requiresGrad: false });
  return asDtype(g, dtype);
}

/**
 * Mean Squared Error (MSE) loss function.
 *
 * **Mathematical Formula:**
 * ```
 * MSE = mean((y_pred - y_true)^2)
 * ```
 *
 * **Use Cases:**
 * - Regression tasks
 * - Continuous value prediction
 * - Measuring distance between predictions and targets
 *
 * **Properties:**
 * - Always non-negative
 * - Penalizes large errors more heavily (quadratic)
 * - Differentiable everywhere
 *
 * Passing a GradTensor as `predictions` returns a GradTensor that supports `.backward()`.
 * Operands with the same number of elements but different rank (for example `(N, 1)`
 * and `(N,)`) are matched element for element rather than broadcast.
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 * import { mseLoss } from 'deepbox/nn';
 *
 * const predictions = tensor([2.5, 0.0, 2.1, 7.8]);
 * const targets = tensor([3.0, -0.5, 2.0, 8.0]);
 * const loss = mseLoss(predictions, targets); // Scalar tensor
 * ```
 *
 * @category Loss Functions
 */
export function mseLoss(predictions: Tensor, targets: Tensor, reduction?: LossReduction): Tensor;
export function mseLoss(
  predictions: GradTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): GradTensor;
export function mseLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): AnyTensor;
export function mseLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "mseLoss");

  // GradTensor path: preserves computation graph for .backward()
  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "mseLoss");
    const diff = pred.sub(tgt);
    return reduceGradLoss(diff.mul(diff), reduction);
  }

  // Plain Tensor path
  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "mseLoss");
  ensureNumeric(tgts, "mseLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "mseLoss");

  const diff = sub(preds, tgts);
  const squaredDiff = pow(diff, tensor(2, { dtype: diff.dtype, device: diff.device }));
  return reduceLoss(squaredDiff, reduction);
}

/**
 * Mean Absolute Error (MAE) loss function, also known as L1 loss.
 *
 * **Mathematical Formula:**
 * ```
 * MAE = mean(|y_pred - y_true|)
 * ```
 *
 * **Use Cases:**
 * - Regression tasks where outliers should have less influence
 * - Less sensitive to outliers than MSE
 *
 * **Properties:**
 * - Always non-negative
 * - Linear penalty for errors
 * - Less sensitive to outliers than MSE
 *
 * Passing a GradTensor as `predictions` returns a GradTensor that supports `.backward()`.
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @category Loss Functions
 */
export function maeLoss(predictions: Tensor, targets: Tensor, reduction?: LossReduction): Tensor;
export function maeLoss(
  predictions: GradTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): GradTensor;
export function maeLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): AnyTensor;
export function maeLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "maeLoss");

  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "maeLoss");
    return reduceGradLoss(pred.sub(tgt).abs(), reduction);
  }

  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "maeLoss");
  ensureNumeric(tgts, "maeLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "maeLoss");

  return reduceLoss(abs(sub(preds, tgts)), reduction);
}

/**
 * Binary Cross-Entropy (BCE) loss function.
 *
 * **Mathematical Formula:**
 * ```
 * BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
 * ```
 *
 * **Use Cases:**
 * - Binary classification tasks
 * - Multi-label classification (independent binary decisions)
 * - Predictions should be probabilities in (0, 1)
 *
 * **Properties:**
 * - Requires predictions in range [0, 1] - use sigmoid activation; other values (and NaN)
 *   throw an `InvalidParameterError`
 * - Targets should be 0 or 1
 * - Like PyTorch, each logarithm is clamped at -100, so the loss is finite (at most 100 per
 *   element) when a prediction is exactly 0 or 1. The gradient with respect to the prediction
 *   is `(p - t) / max(p * (1 - p), 1e-12)`.
 *   When you have logits, prefer `binaryCrossEntropyWithLogitsLoss`.
 *
 * @param predictions - Predicted probabilities (0 to 1)
 * @param targets - True binary labels (0 or 1)
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 * @throws {InvalidParameterError} If a prediction is outside [0, 1] or NaN
 *
 * @category Loss Functions
 */
export function binaryCrossEntropyLoss(
  predictions: Tensor,
  targets: Tensor,
  reduction?: LossReduction
): Tensor;
export function binaryCrossEntropyLoss(
  predictions: GradTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): GradTensor;
export function binaryCrossEntropyLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction?: LossReduction
): AnyTensor;
export function binaryCrossEntropyLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "binaryCrossEntropyLoss");

  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "binaryCrossEntropyLoss");
    const p = numbersOf(pred.tensor, "binaryCrossEntropyLoss");
    const t = numbersOf(tgt.tensor, "binaryCrossEntropyLoss");
    checkProbabilities(p);
    const dtype = lossDtypeOf(pred.tensor);
    const loss = Tensor.fromTypedArray({
      data: bceElementwise(p, t, dtype),
      shape: pred.shape.slice(),
      dtype,
      device: pred.device,
    });
    const shape = pred.shape.slice();
    const fromElements = (fn: (i: number) => number): Tensor => {
      const out = newLossBuffer(dtype, pred.size);
      for (let i = 0; i < pred.size; i++) out[i] = fn(i);
      return Tensor.fromTypedArray({ data: out, shape, dtype, device: pred.device });
    };
    const elementwise = customOp(loss, [
      [
        pred,
        (g: Tensor): Tensor => {
          // d loss / d p = (p - t) / max(p * (1 - p), 1e-12), the PyTorch backward formula.
          const go = numbersOf(g, "binaryCrossEntropyLoss");
          return fromElements((i) => {
            const pi = p[i] ?? 0;
            const denominator = Math.max(pi * (1 - pi), BCE_GRAD_EPSILON);
            return ((go[i] ?? 0) * (pi - (t[i] ?? 0))) / denominator;
          });
        },
      ],
      [
        tgt,
        (g: Tensor): Tensor => {
          const go = numbersOf(g, "binaryCrossEntropyLoss");
          return fromElements((i) => {
            const pi = p[i] ?? 0;
            return (go[i] ?? 0) * (bceLog(1 - pi) - bceLog(pi));
          });
        },
      ],
    ]);
    return reduceGradLoss(elementwise, reduction);
  }

  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "binaryCrossEntropyLoss");
  ensureNumeric(tgts, "binaryCrossEntropyLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "binaryCrossEntropyLoss");

  const p = numbersOf(preds, "binaryCrossEntropyLoss");
  checkProbabilities(p);
  const dtype = lossDtypeOf(preds);
  return reduceLoss(
    Tensor.fromTypedArray({
      data: bceElementwise(p, numbersOf(tgts, "binaryCrossEntropyLoss"), dtype),
      shape: preds.shape.slice(),
      dtype,
      device: preds.device,
    }),
    reduction
  );
}

/**
 * Root Mean Squared Error (RMSE) loss function.
 *
 * **Mathematical Formula:**
 * ```
 * RMSE = sqrt(mean((y_pred - y_true)^2))
 * ```
 *
 * **Use Cases:**
 * - Regression tasks
 * - When you want error in same units as target
 * - More interpretable than MSE
 *
 * The gradient is undefined (NaN) at a perfect fit, where the square root is taken at 0.
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @returns Scalar loss value
 *
 * @category Loss Functions
 */
export function rmseLoss(predictions: Tensor, targets: Tensor): Tensor;
export function rmseLoss(predictions: GradTensor, targets: AnyTensor): GradTensor;
export function rmseLoss(predictions: AnyTensor, targets: AnyTensor): AnyTensor;
export function rmseLoss(predictions: AnyTensor, targets: AnyTensor): AnyTensor {
  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "rmseLoss");
    const diff = pred.sub(tgt);
    return diff.mul(diff).mean().sqrt();
  }

  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "rmseLoss");
  ensureNumeric(tgts, "rmseLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "rmseLoss");

  const mse = mseLoss(preds, tgts, "mean");
  return sqrt(mse);
}

/**
 * Huber loss function - combines MSE and MAE.
 *
 * **Mathematical Formula:**
 * ```
 * Huber(a) = 0.5 * a^2           if |a| <= delta
 *          = delta * (|a| - 0.5 * delta)  otherwise
 * where a = y_pred - y_true
 * ```
 *
 * **Use Cases:**
 * - Regression with outliers
 * - Limits the influence of outliers while keeping MSE behavior for small errors
 *
 * **Properties:**
 * - Quadratic for small errors (like MSE)
 * - Linear for large errors (like MAE)
 * - Controlled by delta parameter
 *
 * Passing a GradTensor as `predictions` returns a GradTensor that supports `.backward()`.
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param delta - Threshold where loss transitions from quadratic to linear (must be positive)
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @category Loss Functions
 */
export function huberLoss(
  predictions: Tensor,
  targets: Tensor,
  delta?: number,
  reduction?: LossReduction
): Tensor;
export function huberLoss(
  predictions: GradTensor,
  targets: AnyTensor,
  delta?: number,
  reduction?: LossReduction
): GradTensor;
export function huberLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  delta?: number,
  reduction?: LossReduction
): AnyTensor;
export function huberLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  delta = 1.0,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "huberLoss");
  if (!Number.isFinite(delta) || delta <= 0) {
    throw new InvalidParameterError(`delta must be positive; got ${delta}`, "delta", delta);
  }

  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "huberLoss");
    // With c = min(|a|, delta): huber(a) = 0.5 * c^2 + delta * (|a| - c).
    const absDiff = pred.sub(tgt).abs();
    const c = absDiff.clip(0, delta);
    const dtype = numericDtype(pred, "huberLoss");
    const half = GradTensor.scalar(0.5, { dtype });
    const deltaT = GradTensor.scalar(delta, { dtype });
    const loss = half.mul(c.mul(c)).add(deltaT.mul(absDiff.sub(c)));
    return reduceGradLoss(loss, reduction);
  }

  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "huberLoss");
  ensureNumeric(tgts, "huberLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "huberLoss");

  const p = numbersOf(preds, "huberLoss");
  const t = numbersOf(tgts, "huberLoss");
  const dtype = lossDtypeOf(preds);
  const lossData = newLossBuffer(dtype, preds.size);
  for (let i = 0; i < preds.size; i++) {
    const absVal = Math.abs((p[i] ?? 0) - (t[i] ?? 0));
    lossData[i] = absVal <= delta ? 0.5 * absVal * absVal : delta * (absVal - 0.5 * delta);
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: preds.shape, dtype, device: preds.device }),
    reduction
  );
}

/**
 * Negative Log Likelihood (NLL) loss.
 *
 * Expects log-probabilities as input (use LogSoftmax before this loss).
 *
 * **Formula**: NLL = -sum(log_probs[i][target[i]]) / N
 *
 * With class weights `w` and `"mean"` reduction the sum is divided by `sum(w[target[i]])`
 * instead of N, as in PyTorch. Passing a GradTensor as `logProbs` returns a GradTensor
 * that supports `.backward()`.
 *
 * Besides `(N, C)` log-probabilities with `(N,)` targets, the input may be `(N, C, d1, ...)` with
 * `(N, d1, ...)` targets, or an unbatched `(C)` vector with a scalar target, as in PyTorch.
 * Targets equal to `ignoreIndex` (default -100) are skipped.
 *
 * @param logProbs - Log-probabilities of shape (N, C), (N, C, d1, ...) or (C), where C = number of classes
 * @param targets - Class indices: integers in [0, C) of shape (N,), (N, d1, ...) or a scalar
 * @param reductionOrOptions - Reduction (`"mean"` default) or an options object with
 *   `reduction`, per-class `weight` and `ignoreIndex`
 * @throws {InvalidParameterError} If a target is not an integer in [0, C)
 *
 * @category Loss Functions
 */
export function nllLoss(
  logProbs: Tensor,
  targets: Tensor,
  reductionOrOptions?: LossReduction | NllLossOptions
): Tensor;
export function nllLoss(
  logProbs: GradTensor,
  targets: AnyTensor,
  reductionOrOptions?: LossReduction | NllLossOptions
): GradTensor;
export function nllLoss(
  logProbs: AnyTensor,
  targets: AnyTensor,
  reductionOrOptions?: LossReduction | NllLossOptions
): AnyTensor;
export function nllLoss(
  logProbs: AnyTensor,
  targets: AnyTensor,
  reductionOrOptions: LossReduction | NllLossOptions = "mean"
): AnyTensor {
  const options: NllLossOptions =
    typeof reductionOrOptions === "string" ? { reduction: reductionOrOptions } : reductionOrOptions;
  const reduction = options.reduction ?? "mean";
  validateReduction(reduction, "nllLoss");
  const ignoreIndex = options.ignoreIndex ?? DEFAULT_IGNORE_INDEX;
  if (!Number.isInteger(ignoreIndex)) {
    throw new InvalidParameterError(
      `ignoreIndex must be an integer; got ${String(ignoreIndex)}`,
      "ignoreIndex",
      ignoreIndex
    );
  }

  const layout = flattenClassLayout(
    GradTensor.isGradTensor(logProbs) ? logProbs : GradTensor.fromTensor(logProbs),
    GradTensor.fromTensor(toTensor(targets)),
    "nllLoss",
    false
  );
  const lp = layout.input;
  const tgt = layout.target.tensor;
  numericDtype(lp, "nllLoss");
  ensureNumeric(tgt, "nllLoss");

  const N = lp.shape[0] ?? 0;
  const C = lp.shape[1] ?? 0;

  if (tgt.ndim !== 1) {
    throw new ShapeError(`nllLoss expects 1D targets; got ${tgt.ndim}D`);
  }
  if ((tgt.shape[0] ?? 0) !== N) {
    throw new ShapeError(`targets length ${tgt.shape[0]} doesn't match batch size ${N}`);
  }
  if (C === 0) {
    throw new ShapeError("nllLoss expects at least one class; got 0 columns");
  }

  const loss = classIndexLoss(lp, tgt, {
    reduction,
    weight: options.weight === undefined ? null : readClassWeights(options.weight, C),
    ignoreIndex,
    labelSmoothing: 0,
  });
  const shaped = reduction === "none" ? loss.reshape([...layout.outShape]) : loss;
  return GradTensor.isGradTensor(logProbs) || GradTensor.isGradTensor(targets)
    ? shaped
    : shaped.tensor;
}

/**
 * KL Divergence loss.
 *
 * Measures how one probability distribution diverges from a second.
 *
 * **Formula**: KL(P || Q) = sum(P * (log(P) - log(Q)))
 *
 * Input should be log-probabilities, target should be probabilities (or
 * log-probabilities when `logTarget` is true). Terms with a target of exactly 0 are
 * 0, by the convention 0 * log(0) = 0. A negative or NaN target gives NaN.
 *
 * Reductions: `"mean"` averages over all elements, `"batchmean"` divides the sum by the
 * size of the first dimension (the mathematically correct KL per sample), `"sum"` adds
 * everything, `"none"` keeps the element-wise terms.
 *
 * Passing a GradTensor as `input` returns a GradTensor; gradients flow to `input` only
 * (the target is treated as a constant).
 *
 * @param input - Log-probabilities (from LogSoftmax)
 * @param target - Target probability distribution
 * @param reduction - How to reduce the loss
 * @param logTarget - Whether `target` holds log-probabilities (default: false)
 *
 * @category Loss Functions
 */
export function klDivLoss(
  input: Tensor,
  target: Tensor,
  reduction?: LossReduction | "batchmean",
  logTarget?: boolean
): Tensor;
export function klDivLoss(
  input: GradTensor,
  target: AnyTensor,
  reduction?: LossReduction | "batchmean",
  logTarget?: boolean
): GradTensor;
export function klDivLoss(
  input: AnyTensor,
  target: AnyTensor,
  reduction?: LossReduction | "batchmean",
  logTarget?: boolean
): AnyTensor;
export function klDivLoss(
  input: AnyTensor,
  target: AnyTensor,
  reduction: LossReduction | "batchmean" = "mean",
  logTarget = false
): AnyTensor {
  if (reduction !== "batchmean") {
    validateReduction(reduction, "klDivLoss");
  }
  const inputIsGrad = GradTensor.isGradTensor(input);
  let inp = toTensor(input);
  let tgt = toTensor(target);
  ensureNumeric(inp, "klDivLoss");
  ensureNumeric(tgt, "klDivLoss");
  [inp, tgt] = alignShapes(inp, tgt);
  ensureSameShape(inp, tgt, "klDivLoss");

  // loss = target * (log(target) - input), or exp(t) * (t - input) with log targets.
  // Terms whose target is exactly 0 are 0 (0 * log(0) = 0 by convention).
  const q = numbersOf(inp, "klDivLoss");
  const t = numbersOf(tgt, "klDivLoss");
  const size = inp.size;
  const dtype = lossDtypeOf(inp);
  const lossData = newLossBuffer(dtype, size);
  const prob = logTarget ? new Float64Array(size) : null;
  for (let i = 0; i < size; i++) {
    const ti = t[i] ?? 0;
    const qi = q[i] ?? 0;
    if (logTarget) {
      const p = Math.exp(ti);
      if (prob) prob[i] = p;
      lossData[i] = p === 0 ? 0 : p * (ti - qi);
    } else {
      lossData[i] = ti === 0 ? 0 : ti * (Math.log(ti) - qi);
    }
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: inp.shape,
    dtype,
    device: inp.device,
  });

  // batchmean divides the total by the batch size; the others reuse the shared reducers.
  const batchSize = inp.shape[0] ?? 1;
  const reduceTensor = (l: Tensor): Tensor => {
    if (reduction !== "batchmean") return reduceLoss(l, reduction);
    const total = scalarValue(sum(l), "klDivLoss");
    return Tensor.fromTypedArray({
      data: newLossBuffer(dtype, 1).fill(total / batchSize),
      shape: [],
      dtype,
      device: inp.device,
    });
  };

  if (!inputIsGrad) {
    return reduceTensor(loss);
  }

  // Autograd: d loss / d input = -target (or -exp(target) for log targets).
  const grads = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const p = prob ? (prob[i] ?? 0) : (t[i] ?? 0);
    grads[i] = p === 0 ? 0 : -p;
  }
  const shape = [...inp.shape];
  const elementwise = customOp(loss, [
    [
      input as GradTensor,
      (g: Tensor): Tensor => {
        const go = numbersOf(g, "klDivLoss");
        const out = newLossBuffer(dtype, size);
        for (let i = 0; i < size; i++) {
          out[i] = (go[i] ?? 0) * (grads[i] ?? 0);
        }
        return Tensor.fromTypedArray({ data: out, shape, dtype, device: inp.device });
      },
    ],
  ]);
  if (reduction === "batchmean") {
    return elementwise
      .sum()
      .div(GradTensor.scalar(batchSize, { dtype: numericDtype(elementwise, "klDivLoss") }));
  }
  return reduceGradLoss(elementwise, reduction);
}

/**
 * Smooth L1 Loss (Huber-like loss used in object detection).
 *
 * **Formula**:
 * ```
 * loss = 0.5 * x^2 / beta    if |x| < beta
 *      = |x| - 0.5 * beta     otherwise
 * ```
 *
 * With `beta = 0` this is the L1 loss. Passing a GradTensor as `predictions` returns a
 * GradTensor that supports `.backward()`.
 *
 * @param predictions - Predicted values
 * @param targets - Target values
 * @param beta - Threshold for switching between L1 and L2, non-negative (default: 1.0)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function smoothL1Loss(
  predictions: Tensor,
  targets: Tensor,
  beta?: number,
  reduction?: LossReduction
): Tensor;
export function smoothL1Loss(
  predictions: GradTensor,
  targets: AnyTensor,
  beta?: number,
  reduction?: LossReduction
): GradTensor;
export function smoothL1Loss(
  predictions: AnyTensor,
  targets: AnyTensor,
  beta?: number,
  reduction?: LossReduction
): AnyTensor;
export function smoothL1Loss(
  predictions: AnyTensor,
  targets: AnyTensor,
  beta = 1.0,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "smoothL1Loss");
  if (!Number.isFinite(beta) || beta < 0) {
    throw new InvalidParameterError(`beta must be non-negative; got ${beta}`, "beta", beta);
  }

  if (GradTensor.isGradTensor(predictions)) {
    const [pred, tgt] = gradOperands(predictions, targets, "smoothL1Loss");
    const absDiff = pred.sub(tgt).abs();
    if (beta === 0) return reduceGradLoss(absDiff, reduction);
    // With c = min(|x|, beta): loss = 0.5 * c^2 / beta + (|x| - c).
    const c = absDiff.clip(0, beta);
    const dtype = numericDtype(pred, "smoothL1Loss");
    const halfOverBeta = GradTensor.scalar(0.5 / beta, { dtype });
    return reduceGradLoss(halfOverBeta.mul(c.mul(c)).add(absDiff.sub(c)), reduction);
  }

  let preds = predictions;
  let tgts = toTensor(targets);
  ensureNumeric(preds, "smoothL1Loss");
  ensureNumeric(tgts, "smoothL1Loss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "smoothL1Loss");

  const p = numbersOf(preds, "smoothL1Loss");
  const t = numbersOf(tgts, "smoothL1Loss");
  const dtype = lossDtypeOf(preds);
  const lossData = newLossBuffer(dtype, preds.size);
  for (let i = 0; i < preds.size; i++) {
    const absVal = Math.abs((p[i] ?? 0) - (t[i] ?? 0));
    lossData[i] = absVal < beta ? (0.5 * absVal * absVal) / beta : absVal - 0.5 * beta;
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: preds.shape, dtype, device: preds.device }),
    reduction
  );
}

/**
 * Cosine Embedding Loss for similarity/contrastive learning.
 *
 * **Formula**:
 * ```
 * loss = 1 - cos(x1, x2)           if y == 1
 *      = max(0, cos(x1, x2) - margin) if y == -1
 * ```
 *
 * The cosine is `dot(x1, x2) / sqrt((|x1|^2 + 1e-12) * (|x2|^2 + 1e-12))`, as in PyTorch.
 * Passing a GradTensor as `x1` or `x2` returns a GradTensor that supports `.backward()`.
 *
 * @param x1 - First input, shape (N, D) or (D,)
 * @param x2 - Second input, same shape as `x1`
 * @param y - Labels: 1 (similar) or -1 (dissimilar), one per row of `x1`
 * @param margin - Margin for dissimilar pairs (default: 0)
 * @param reduction - How to reduce the loss
 * @throws {ShapeError} If `x1` and `x2` differ in shape, have more than 2 dimensions, or `y` has the wrong length
 * @throws {InvalidParameterError} If a label is neither 1 nor -1
 *
 * @category Loss Functions
 */
export function cosineEmbeddingLoss(
  x1: Tensor,
  x2: Tensor,
  y: Tensor,
  margin?: number,
  reduction?: LossReduction
): Tensor;
export function cosineEmbeddingLoss(
  x1: GradTensor,
  x2: AnyTensor,
  y: Tensor,
  margin?: number,
  reduction?: LossReduction
): GradTensor;
export function cosineEmbeddingLoss(
  x1: AnyTensor,
  x2: GradTensor,
  y: Tensor,
  margin?: number,
  reduction?: LossReduction
): GradTensor;
export function cosineEmbeddingLoss(
  x1: AnyTensor,
  x2: AnyTensor,
  y: Tensor,
  margin?: number,
  reduction?: LossReduction
): AnyTensor;
export function cosineEmbeddingLoss(
  x1: AnyTensor,
  x2: AnyTensor,
  y: Tensor,
  margin = 0,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "cosineEmbeddingLoss");
  const x1t = toTensor(x1);
  const x2t = toTensor(x2);
  ensureNumeric(x1t, "cosineEmbeddingLoss");
  ensureNumeric(x2t, "cosineEmbeddingLoss");
  ensureNumeric(y, "cosineEmbeddingLoss");

  if (x1t.ndim < 1 || x1t.ndim > 2) {
    throw new ShapeError(`cosineEmbeddingLoss expects 1D or 2D inputs; got ${x1t.ndim}D`);
  }
  ensureSameShape(x1t, x2t, "cosineEmbeddingLoss (x1 vs x2)");

  // For 2D inputs: (N, D), compute cosine similarity per row
  // For 1D inputs: (D,), compute single cosine similarity
  const batchSize = x1t.ndim === 2 ? (x1t.shape[0] ?? 1) : 1;
  const dim = x1t.ndim === 2 ? (x1t.shape[1] ?? 1) : (x1t.shape[0] ?? 1);
  if (y.size !== batchSize) {
    throw new ShapeError(
      `cosineEmbeddingLoss expects ${batchSize} label(s) (one per row of x1); got ${y.size}`
    );
  }

  const labels = numbersOf(y, "cosineEmbeddingLoss");
  for (let r = 0; r < batchSize; r++) {
    const label = labels[r] ?? Number.NaN;
    if (label !== 1 && label !== -1) {
      throw new InvalidParameterError(
        `cosineEmbeddingLoss labels must be 1 or -1; got ${label} at index ${r}`,
        "y",
        label
      );
    }
  }
  const EPS = 1e-12;
  const dtype = lossDtypeOf(x1t);

  if (anyGrad(x1, x2)) {
    // cos = dot / sqrt((|x1|^2 + eps) * (|x2|^2 + eps)); positive pairs use 1 - cos, negative
    // pairs use relu(cos - margin). Constant 0/1 masks pick the branch of each row.
    const a = gradOperand(x1, dtype).reshape([batchSize, dim]);
    const b = gradOperand(x2, dtype).reshape([batchSize, dim]);
    const dot = a.mul(b).sum(1);
    const norm1 = a.mul(a).sum(1).add(EPS);
    const norm2 = b.mul(b).sum(1).add(EPS);
    const cos = dot.div(norm1.mul(norm2).sqrt());
    const positive = newLossBuffer(dtype, batchSize);
    const negative = newLossBuffer(dtype, batchSize);
    for (let r = 0; r < batchSize; r++) {
      if (labels[r] === 1) positive[r] = 1;
      else negative[r] = 1;
    }
    const mask = (data: Float32Array | Float64Array): GradTensor =>
      GradTensor.fromTensor(
        Tensor.fromTypedArray({ data, shape: [batchSize], dtype, device: x1t.device })
      );
    const positiveLoss = mask(positive).mul(GradTensor.scalar(1, { dtype }).sub(cos));
    const negativeLoss = mask(negative).mul(cos.sub(margin).relu());
    return reduceGradLoss(positiveLoss.add(negativeLoss), reduction);
  }

  const a = numbersOf(x1t, "cosineEmbeddingLoss");
  const b = numbersOf(x2t, "cosineEmbeddingLoss");
  const lossData = newLossBuffer(dtype, batchSize);

  for (let r = 0; r < batchSize; r++) {
    let dot = 0;
    let norm1 = 0;
    let norm2 = 0;
    for (let d = 0; d < dim; d++) {
      const v1 = a[r * dim + d] ?? 0;
      const v2 = b[r * dim + d] ?? 0;
      dot += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }
    const cosSim = dot / Math.sqrt((norm1 + EPS) * (norm2 + EPS));
    lossData[r] = labels[r] === 1 ? 1 - cosSim : Math.max(0, cosSim - margin);
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: [batchSize], dtype, device: x1t.device }),
    reduction
  );
}

/** Options for {@link tripletMarginLoss}. */
export interface TripletMarginLossOptions {
  /** Margin between the positive and the negative distance (default: 1). */
  readonly margin?: number;
  /** Norm degree of the pairwise distance: a positive number or `Infinity` (default: 2). */
  readonly p?: number;
  /** Small constant added to the difference before the norm, as in PyTorch (default: 1e-6). */
  readonly eps?: number;
  /**
   * Use the distance swap of Balntas et al. (2016): the negative distance becomes
   * `min(d(anchor, negative), d(positive, negative))` (default: false).
   */
  readonly swap?: boolean;
  /** `"mean"` (default), `"sum"`, or `"none"` to return one loss per row. */
  readonly reduction?: LossReduction;
}

/** Row-wise `p`-norm of `a - b + eps` over the last axis (PyTorch's `pairwise_distance`). */
function tripletDistances(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  rows: number,
  dim: number,
  p: number,
  eps: number
): Float64Array {
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    for (let d = 0; d < dim; d++) {
      const diff = Math.abs((a[r * dim + d] ?? 0) - (b[r * dim + d] ?? 0) + eps);
      if (p === 2) acc += diff * diff;
      else if (p === 1) acc += diff;
      else if (p === Number.POSITIVE_INFINITY) acc = Math.max(acc, diff);
      else acc += diff ** p;
    }
    out[r] =
      p === 2 ? Math.sqrt(acc) : p === 1 || p === Number.POSITIVE_INFINITY ? acc : acc ** (1 / p);
  }
  return out;
}

/**
 * Apply `root` to the non-negative `s` so that entries equal to zero get the gradient 0
 * instead of an infinite slope times a zero upstream gradient (NaN). PyTorch's norm also
 * uses the subgradient 0 at the origin. Only needed when `eps` is 0.
 */
function rootWithZeroGrad(s: GradTensor, root: (x: GradTensor) => GradTensor): GradTensor {
  const values = numbersOf(s.tensor, "tripletMarginLoss");
  const dtype = s.dtype === "float64" ? "float64" : "float32";
  const data =
    dtype === "float64" ? new Float64Array(values.length) : new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) data[i] = values[i] === 0 ? 1 : 0;
  const mask = GradTensor.fromTensor(
    Tensor.fromTypedArray({ data, shape: [...s.shape], dtype, device: s.tensor.device }),
    { requiresGrad: false }
  );
  const keep = mask.neg().add(GradTensor.scalar(1, { dtype }));
  return root(s.add(mask)).mul(keep);
}

/** Differentiable row-wise `p`-norm of `a - b + eps` over the last axis. */
function tripletGradDistance(
  a: GradTensor,
  b: GradTensor,
  p: number,
  eps: GradTensor,
  zeroSafe: boolean
): GradTensor {
  const diff = a.sub(b).add(eps);
  if (p === 2) {
    const squares = diff.mul(diff).sum(-1);
    return zeroSafe ? rootWithZeroGrad(squares, (x) => x.sqrt()) : squares.sqrt();
  }
  if (p === 1) return diff.abs().sum(-1);
  if (p === Number.POSITIVE_INFINITY) return diff.abs().max(-1);
  const powered = diff.abs().pow(p).sum(-1);
  return zeroSafe ? rootWithZeroGrad(powered, (x) => x.pow(1 / p)) : powered.pow(1 / p);
}

/**
 * Triplet Margin Loss for metric learning.
 *
 * **Formula**: `loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)`
 *
 * `d(x, y)` is the `p`-norm (Euclidean by default) of `x - y + eps` taken over the last axis,
 * as in PyTorch's `triplet_margin_loss`; the small `eps` keeps the distance differentiable
 * when two rows coincide. With `swap`, the negative distance is replaced by the smaller of
 * `d(anchor, negative)` and `d(positive, negative)`.
 *
 * If any of the three inputs is a `GradTensor` the loss is a `GradTensor` that supports
 * `.backward()`. Inputs are 1-D `(D,)` (one triplet, scalar loss) or 2-D `(N, D)` (one loss per
 * row). For compatibility the margin and the reduction can also be passed positionally.
 *
 * @param anchor - Anchor embeddings, shape (N, D) or (D,)
 * @param positive - Positive (similar) embeddings, same shape as `anchor`
 * @param negative - Negative (dissimilar) embeddings, same shape as `anchor`
 * @param marginOrOptions - Margin (default: 1.0), or an options object
 *   ({@link TripletMarginLossOptions})
 * @param reduction - How to reduce the loss (positional form only; default: `"mean"`)
 * @throws {ShapeError} If the three inputs differ in shape or have more than 2 dimensions
 * @throws {InvalidParameterError} If `margin`, `p` or `eps` is invalid
 *
 * @example
 * ```ts
 * import { tripletMarginLoss } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const anchor = tensor([[0, 0], [1, 1]]);
 * const positive = tensor([[0, 1], [1, 2]]);
 * const negative = tensor([[3, 3], [4, 4]]);
 * const loss = tripletMarginLoss(anchor, positive, negative, { margin: 1, swap: true });
 * ```
 *
 * @category Loss Functions
 */
export function tripletMarginLoss(
  anchor: GradTensor,
  positive: AnyTensor,
  negative: AnyTensor,
  marginOrOptions?: number | TripletMarginLossOptions,
  reduction?: LossReduction
): GradTensor;
export function tripletMarginLoss(
  anchor: Tensor,
  positive: Tensor,
  negative: Tensor,
  marginOrOptions?: number | TripletMarginLossOptions,
  reduction?: LossReduction
): Tensor;
export function tripletMarginLoss(
  anchor: AnyTensor,
  positive: AnyTensor,
  negative: AnyTensor,
  marginOrOptions?: number | TripletMarginLossOptions,
  reduction?: LossReduction
): AnyTensor;
export function tripletMarginLoss(
  anchor: AnyTensor,
  positive: AnyTensor,
  negative: AnyTensor,
  marginOrOptions: number | TripletMarginLossOptions = 1.0,
  reductionArg: LossReduction = "mean"
): AnyTensor {
  const options: TripletMarginLossOptions =
    typeof marginOrOptions === "object" && marginOrOptions !== null
      ? marginOrOptions
      : { margin: marginOrOptions, reduction: reductionArg };
  const margin = options.margin ?? 1.0;
  const p = options.p ?? 2;
  const eps = options.eps ?? 1e-6;
  const swap = options.swap ?? false;
  const reduction = options.reduction ?? "mean";

  validateReduction(reduction, "tripletMarginLoss");
  if (typeof margin !== "number" || !Number.isFinite(margin)) {
    throw new InvalidParameterError("margin must be a finite number", "margin", margin);
  }
  if (typeof p !== "number" || Number.isNaN(p) || p <= 0) {
    throw new InvalidParameterError("p must be a positive number", "p", p);
  }
  if (typeof eps !== "number" || !Number.isFinite(eps) || eps < 0) {
    throw new InvalidParameterError("eps must be a non-negative finite number", "eps", eps);
  }

  const a = toTensor(anchor);
  const pos = toTensor(positive);
  const neg = toTensor(negative);
  ensureNumeric(a, "tripletMarginLoss");
  ensureNumeric(pos, "tripletMarginLoss");
  ensureNumeric(neg, "tripletMarginLoss");
  ensureSameShape(a, pos, "tripletMarginLoss (anchor vs positive)");
  ensureSameShape(a, neg, "tripletMarginLoss (anchor vs negative)");
  if (a.ndim < 1 || a.ndim > 2) {
    throw new ShapeError(`tripletMarginLoss expects 1D or 2D inputs; got ${a.ndim}D`);
  }

  const dtype = lossDtypeOf(a);
  const batchSize = a.ndim === 2 ? (a.shape[0] ?? 1) : 1;
  const dim = a.ndim === 2 ? (a.shape[1] ?? 1) : (a.shape[0] ?? 1);

  if (
    GradTensor.isGradTensor(anchor) ||
    GradTensor.isGradTensor(positive) ||
    GradTensor.isGradTensor(negative)
  ) {
    const toGrad = (t: AnyTensor): GradTensor =>
      asDtype(
        GradTensor.isGradTensor(t) ? t : GradTensor.fromTensor(t, { requiresGrad: false }),
        dtype
      );
    const ga = toGrad(anchor);
    const gp = toGrad(positive);
    const gn = toGrad(negative);
    const epsT = GradTensor.scalar(eps, { dtype });
    const zeroSafe = eps === 0;
    const distPos = tripletGradDistance(ga, gp, p, epsT, zeroSafe);
    let distNeg = tripletGradDistance(ga, gn, p, epsT, zeroSafe);
    if (swap) {
      // min(a, b) = a - relu(a - b)
      const distSwap = tripletGradDistance(gp, gn, p, epsT, zeroSafe);
      distNeg = distNeg.sub(distNeg.sub(distSwap).relu());
    }
    const loss = distPos.sub(distNeg).add(GradTensor.scalar(margin, { dtype })).relu();
    return reduceGradLoss(loss, reduction);
  }

  const anc = numbersOf(a, "tripletMarginLoss");
  const posData = numbersOf(pos, "tripletMarginLoss");
  const negData = numbersOf(neg, "tripletMarginLoss");
  const distPos = tripletDistances(anc, posData, batchSize, dim, p, eps);
  const distNeg = tripletDistances(anc, negData, batchSize, dim, p, eps);
  if (swap) {
    const distSwap = tripletDistances(posData, negData, batchSize, dim, p, eps);
    for (let r = 0; r < batchSize; r++) {
      distNeg[r] = Math.min(distNeg[r] as number, distSwap[r] as number);
    }
  }

  const lossData = newLossBuffer(dtype, batchSize);
  for (let r = 0; r < batchSize; r++) {
    lossData[r] = Math.max(0, (distPos[r] as number) - (distNeg[r] as number) + margin);
  }

  // One triplet (1-D input) gives a scalar, like PyTorch.
  const shape = a.ndim === 2 ? [batchSize] : [];
  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape, dtype, device: a.device }),
    reduction
  );
}

/**
 * Gaussian Negative Log Likelihood Loss.
 *
 * **Formula**:
 * ```
 * loss = 0.5 * (log(var) + (input - target)^2 / var + log(2π))
 * ```
 *
 * **Use Cases:**
 * - Probabilistic regression where the model predicts both mean and variance
 * - Heteroscedastic regression (varying noise)
 *
 * `variance` may have the same shape as `input`, the shape of `input` with the last
 * dimension removed, or the shape of `input` with the last dimension set to 1 (one
 * variance shared by all outputs of a sample). Variances below `eps` are clamped to `eps`;
 * as in PyTorch the clamp does not block the gradient. Passing a GradTensor as `input`,
 * `target` or `variance` returns a GradTensor that supports `.backward()`.
 *
 * @param input - Predicted means
 * @param target - Target values
 * @param variance - Predicted variances (must be non-negative)
 * @param options - Configuration options
 * @param options.full - Include constant log(2π) term (default: false)
 * @param options.eps - Small value for clamping variance (default: 1e-6)
 * @param options.reduction - How to reduce: 'mean', 'sum', or 'none'
 * @returns Loss value
 * @throws {InvalidParameterError} If `variance` contains a negative value
 * @throws {ShapeError} If `variance` has an incompatible shape
 *
 * @category Loss Functions
 */
export function gaussianNLLLoss(
  input: Tensor,
  target: Tensor,
  variance: Tensor,
  options?: GaussianNLLLossOptions
): Tensor;
export function gaussianNLLLoss(
  input: GradTensor,
  target: AnyTensor,
  variance: AnyTensor,
  options?: GaussianNLLLossOptions
): GradTensor;
export function gaussianNLLLoss(
  input: AnyTensor,
  target: AnyTensor,
  variance: GradTensor,
  options?: GaussianNLLLossOptions
): GradTensor;
export function gaussianNLLLoss(
  input: AnyTensor,
  target: AnyTensor,
  variance: AnyTensor,
  options?: GaussianNLLLossOptions
): AnyTensor;
export function gaussianNLLLoss(
  input: AnyTensor,
  target: AnyTensor,
  variance: AnyTensor,
  options: GaussianNLLLossOptions = {}
): AnyTensor {
  const { full = false, eps = 1e-6, reduction = "mean" } = options;
  validateReduction(reduction, "gaussianNLLLoss");
  const inputT = toTensor(input);
  const targetT = toTensor(target);
  const varianceT = toTensor(variance);
  ensureNumeric(inputT, "gaussianNLLLoss");
  ensureNumeric(targetT, "gaussianNLLLoss");
  ensureNumeric(varianceT, "gaussianNLLLoss");
  ensureSameShape(inputT, targetT, "gaussianNLLLoss (input vs target)");

  if (!Number.isFinite(eps) || eps < 0) {
    throw new InvalidParameterError(`eps must be non-negative; got ${eps}`, "eps", eps);
  }

  // How variance entries map onto input elements: one per element, one per row
  // (variance without the last dimension), or one per row with a trailing 1.
  const lastDim = inputT.shape[inputT.ndim - 1] ?? 1;
  const leading = inputT.shape.slice(0, -1);
  let varStep: "element" | "row";
  if (shapesEqual(varianceT.shape, inputT.shape)) {
    varStep = "element";
  } else if (
    inputT.ndim >= 1 &&
    (shapesEqual(varianceT.shape, leading) || shapesEqual(varianceT.shape, [...leading, 1]))
  ) {
    varStep = "row";
  } else {
    throw new ShapeError(
      `Shape mismatch in gaussianNLLLoss (input vs variance): [${inputT.shape}] vs [${varianceT.shape}]`
    );
  }

  const vr = numbersOf(varianceT, "gaussianNLLLoss");
  for (let i = 0; i < vr.length; i++) {
    if ((vr[i] ?? 0) < 0) {
      throw new InvalidParameterError("variance must be non-negative", "variance", vr[i]);
    }
  }

  const dtype = lossDtypeOf(inputT);
  const LOG2PI = Math.log(2 * Math.PI);

  if (anyGrad(input, target, variance)) {
    // As in PyTorch, variances below eps are replaced by eps without blocking the gradient:
    // the clamp is added as a constant offset, so the derivative is taken at the clamped value.
    const offset = newLossBuffer(dtype, varianceT.size);
    for (let i = 0; i < varianceT.size; i++) {
      const v = vr[i] ?? 0;
      offset[i] = v < eps ? eps - v : 0;
    }
    const offsetT = Tensor.fromTypedArray({
      data: offset,
      shape: varianceT.shape.slice(),
      dtype,
      device: varianceT.device,
    });
    let v = gradOperand(variance, dtype).add(offsetT);
    if (varStep === "row" && !shapesEqual(varianceT.shape, [...leading, 1])) {
      v = v.reshape([...leading, 1]);
    }
    const diff = gradOperand(input, dtype).sub(gradOperand(target, dtype));
    let loss = v.log().add(diff.square().div(v)).mul(GradTensor.scalar(0.5, { dtype }));
    if (full) loss = loss.add(0.5 * LOG2PI);
    return reduceGradLoss(loss, reduction);
  }

  const mu = numbersOf(inputT, "gaussianNLLLoss");
  const tg = numbersOf(targetT, "gaussianNLLLoss");
  const lossData = newLossBuffer(dtype, inputT.size);

  for (let i = 0; i < inputT.size; i++) {
    let v = vr[varStep === "element" ? i : Math.floor(i / lastDim)] ?? 0;
    // Clamp variance to eps
    if (v < eps) v = eps;

    const diff = (mu[i] ?? 0) - (tg[i] ?? 0);
    let lossVal = 0.5 * (Math.log(v) + (diff * diff) / v);
    if (full) {
      lossVal += 0.5 * LOG2PI;
    }
    lossData[i] = lossVal;
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: inputT.shape, dtype, device: inputT.device }),
    reduction
  );
}

/**
 * Poisson Negative Log Likelihood Loss.
 *
 * **Formula**:
 * ```
 * loss = exp(input) - target * input
 * ```
 * When `logInput=false`:
 * ```
 * loss = input - target * log(input + eps)
 * ```
 *
 * **Use Cases:**
 * - Count data modeling (Poisson regression)
 * - Event rate prediction
 *
 * Passing a GradTensor as `input` or `target` returns a GradTensor that supports `.backward()`.
 *
 * @param input - Log of expected rate (or rate itself if logInput=false)
 * @param target - Target counts (non-negative)
 * @param options - Configuration options
 * @param options.logInput - If true, input is log(rate) (default: true)
 * @param options.full - Include Stirling approximation term (default: false)
 * @param options.eps - Small value for numerical stability (default: 1e-8)
 * @param options.reduction - How to reduce: 'mean', 'sum', or 'none'
 * @returns Loss value
 *
 * @category Loss Functions
 */
export function poissonNLLLoss(
  input: Tensor,
  target: Tensor,
  options?: PoissonNLLLossOptions
): Tensor;
export function poissonNLLLoss(
  input: GradTensor,
  target: AnyTensor,
  options?: PoissonNLLLossOptions
): GradTensor;
export function poissonNLLLoss(
  input: AnyTensor,
  target: GradTensor,
  options?: PoissonNLLLossOptions
): GradTensor;
export function poissonNLLLoss(
  input: AnyTensor,
  target: AnyTensor,
  options?: PoissonNLLLossOptions
): AnyTensor;
export function poissonNLLLoss(
  input: AnyTensor,
  target: AnyTensor,
  options: PoissonNLLLossOptions = {}
): AnyTensor {
  const { logInput = true, full = false, eps = 1e-8, reduction = "mean" } = options;
  validateReduction(reduction, "poissonNLLLoss");
  const inputT = toTensor(input);
  const targetT = toTensor(target);
  ensureNumeric(inputT, "poissonNLLLoss");
  ensureNumeric(targetT, "poissonNLLLoss");
  ensureSameShape(inputT, targetT, "poissonNLLLoss");
  const dtype = lossDtypeOf(inputT);

  if (anyGrad(input, target)) {
    const x = gradOperand(input, dtype);
    const t = gradOperand(target, dtype);
    let loss = logInput ? x.exp().sub(t.mul(x)) : x.sub(t.mul(x.add(eps).log()));
    if (full) {
      // Stirling term t * log(t) - t + 0.5 * log(2 * pi * t), kept only where t > 1. The
      // operand is raised to 1 elsewhere so that neither value nor gradient hits log(0).
      const safe = t.maximum(1);
      const stirling = safe
        .mul(safe.log())
        .sub(safe)
        .add(
          safe
            .mul(2 * Math.PI)
            .log()
            .mul(GradTensor.scalar(0.5, { dtype }))
        );
      const keep = GradTensor.fromTensor(t.gt(1).astype(dtype));
      loss = loss.add(stirling.mul(keep));
    }
    return reduceGradLoss(loss, reduction);
  }

  const inp = numbersOf(inputT, "poissonNLLLoss");
  const tg = numbersOf(targetT, "poissonNLLLoss");
  const lossData = newLossBuffer(dtype, inputT.size);

  for (let i = 0; i < inputT.size; i++) {
    const x = inp[i] ?? 0;
    const tgt = tg[i] ?? 0;

    let lossVal = logInput
      ? Math.exp(x) - tgt * x // exp(input) - target * input
      : x - tgt * Math.log(x + eps); // input - target * log(input + eps)

    if (full && tgt > 1) {
      // Stirling approximation: target * log(target) - target + 0.5 * log(2π * target)
      lossVal += tgt * Math.log(tgt) - tgt + 0.5 * Math.log(2 * Math.PI * tgt);
    }

    lossData[i] = lossVal;
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: inputT.shape, dtype, device: inputT.device }),
    reduction
  );
}

/**
 * Log-space addition: log(exp(a) + exp(b)) with numerical stability.
 */
function logSumExp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const maxVal = Math.max(a, b);
  return maxVal + Math.log(Math.exp(a - maxVal) + Math.exp(b - maxVal));
}

/** One CTC sample: its number of input frames and its label sequence with interleaved blanks. */
type CtcSample = {
  readonly inputLength: number;
  readonly extended: Int32Array;
};

/**
 * CTC negative log-likelihood of one sample from log-probabilities laid out as (T, N, C).
 * With `grad` given, the exact derivative of the loss with respect to this sample's
 * log-probabilities is added to it (same layout), using the forward (alpha) and backward
 * (beta) recursions in log-space. An infeasible sample has an infinite loss and no gradient.
 */
function ctcSampleLoss(
  lp: ArrayLike<number>,
  N: number,
  C: number,
  b: number,
  sample: CtcSample,
  grad: Float64Array | null
): number {
  const { inputLength, extended } = sample;
  const S = extended.length;
  const labelCount = (S - 1) / 2;
  if (inputLength === 0) return labelCount === 0 ? 0 : Infinity;
  if (inputLength < labelCount) return Infinity;

  const emit = (t: number, s: number): number =>
    lp[(t * N + b) * C + (extended[s] ?? 0)] ?? -Infinity;
  const blank = extended[0] ?? 0;
  const canSkipBack = (s: number): boolean =>
    s > 1 && extended[s] !== blank && extended[s] !== extended[s - 2];

  const alpha = new Float64Array(inputLength * S).fill(-Infinity);
  alpha[0] = emit(0, 0);
  if (S > 1) alpha[1] = emit(0, 1);
  for (let t = 1; t < inputLength; t++) {
    for (let s = 0; s < S; s++) {
      let acc = alpha[(t - 1) * S + s] ?? -Infinity;
      if (s > 0) acc = logSumExp(acc, alpha[(t - 1) * S + s - 1] ?? -Infinity);
      if (canSkipBack(s)) acc = logSumExp(acc, alpha[(t - 1) * S + s - 2] ?? -Infinity);
      alpha[t * S + s] = acc + emit(t, s);
    }
  }
  const last = (inputLength - 1) * S;
  const logLikelihood = logSumExp(
    alpha[last + S - 1] ?? -Infinity,
    S > 1 ? (alpha[last + S - 2] ?? -Infinity) : -Infinity
  );
  const nll = -logLikelihood;
  if (grad === null || !Number.isFinite(nll)) return nll;

  const beta = new Float64Array(inputLength * S).fill(-Infinity);
  beta[last + S - 1] = emit(inputLength - 1, S - 1);
  if (S > 1) beta[last + S - 2] = emit(inputLength - 1, S - 2);
  for (let t = inputLength - 2; t >= 0; t--) {
    for (let s = 0; s < S; s++) {
      let acc = beta[(t + 1) * S + s] ?? -Infinity;
      if (s < S - 1) acc = logSumExp(acc, beta[(t + 1) * S + s + 1] ?? -Infinity);
      if (s < S - 2 && canSkipBack(s + 2)) {
        acc = logSumExp(acc, beta[(t + 1) * S + s + 2] ?? -Infinity);
      }
      beta[t * S + s] = acc + emit(t, s);
    }
  }

  // d(-log p) / d lp[t, c] = -exp(log(sum over s with label c of alpha * beta) - lp[t, c] + nll)
  const through = new Float64Array(C);
  for (let t = 0; t < inputLength; t++) {
    through.fill(-Infinity);
    for (let s = 0; s < S; s++) {
      const c = extended[s] ?? 0;
      through[c] = logSumExp(
        through[c] ?? -Infinity,
        (alpha[t * S + s] ?? 0) + (beta[t * S + s] ?? 0)
      );
    }
    for (let c = 0; c < C; c++) {
      const joint = through[c] ?? -Infinity;
      const logProb = lp[(t * N + b) * C + c] ?? -Infinity;
      if (joint === -Infinity || logProb === -Infinity) continue;
      const index = (t * N + b) * C + c;
      grad[index] = (grad[index] ?? 0) - Math.exp(joint - logProb + nll);
    }
  }
  return nll;
}

/**
 * Read a vector of non-negative integer lengths, rejecting fractional values.
 */
function readLengths(t: Tensor, name: string, context: string): number[] {
  const values = numbersOf(t, context);
  const out: number[] = [];
  for (let i = 0; i < t.size; i++) {
    const v = values[i] ?? Number.NaN;
    if (!Number.isInteger(v)) {
      throw new InvalidParameterError(`${name}[${i}] = ${v} must be an integer`, name, v);
    }
    out.push(v);
  }
  return out;
}

/**
 * Connectionist Temporal Classification (CTC) Loss.
 *
 * CTC loss for sequence-to-sequence models where the alignment between
 * input and output is unknown (e.g., speech recognition, OCR).
 *
 * **Algorithm**: Forward (alpha) computation in log-space over an extended
 * label sequence with interleaved blanks.
 *
 * With the default `"mean"` reduction each sample's loss is divided by its target
 * length (at least 1) before averaging over the batch, matching PyTorch's `CTCLoss`.
 * A sample whose input is too short to emit its target has an infinite loss unless
 * `zeroInfinity` is set. Target length 0 is allowed (the loss is then the
 * negative log-probability of emitting only blanks).
 *
 * Passing a GradTensor as `logProbs` returns a GradTensor that supports `.backward()`. The
 * gradient is the exact derivative of the loss with respect to the log-probabilities, computed
 * with the forward-backward recursions in log-space. PyTorch adds `exp(logProbs)` to its
 * gradient with respect to the log-probabilities themselves, a term that cancels when they come
 * from a log-softmax; the gradients with respect to the logits are identical. An infeasible
 * sample (infinite loss, or zero with `zeroInfinity`) has a zero gradient.
 *
 * @param logProbs - Log-probabilities of shape (T, N, C) where T = input
 *   length (time steps), N = batch size, C = number of classes (including blank)
 * @param targets - Either the concatenated target sequences (1D, length at least the sum of
 *   all target lengths) or a padded matrix of shape (N, S). Values must be in [0, C) and
 *   must not equal `blank`.
 * @param inputLengths - Lengths of each input sequence, shape (N,)
 * @param targetLengths - Lengths of each target sequence, shape (N,)
 * @param options - Configuration options
 * @param options.blank - Index of the blank label (default: 0)
 * @param options.reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @param options.zeroInfinity - Replace infinite losses with zero (default: false)
 * @returns Loss value
 *
 * @category Loss Functions
 */
export function ctcLoss(
  logProbs: Tensor,
  targets: Tensor,
  inputLengths: Tensor,
  targetLengths: Tensor,
  options?: CtcLossOptions
): Tensor;
export function ctcLoss(
  logProbs: GradTensor,
  targets: Tensor,
  inputLengths: Tensor,
  targetLengths: Tensor,
  options?: CtcLossOptions
): GradTensor;
export function ctcLoss(
  logProbs: AnyTensor,
  targets: Tensor,
  inputLengths: Tensor,
  targetLengths: Tensor,
  options?: CtcLossOptions
): AnyTensor;
export function ctcLoss(
  logProbsArg: AnyTensor,
  targets: Tensor,
  inputLengths: Tensor,
  targetLengths: Tensor,
  options: CtcLossOptions = {}
): AnyTensor {
  const logProbs = toTensor(logProbsArg);
  const { blank = 0, reduction = "mean", zeroInfinity = false } = options;
  validateReduction(reduction, "ctcLoss");
  ensureNumeric(logProbs, "ctcLoss");
  ensureNumeric(targets, "ctcLoss");
  ensureNumeric(inputLengths, "ctcLoss");
  ensureNumeric(targetLengths, "ctcLoss");

  if (logProbs.ndim !== 3) {
    throw new ShapeError(`ctcLoss expects 3D logProbs (T, N, C); got ${logProbs.ndim}D`);
  }
  if (targets.ndim !== 1 && targets.ndim !== 2) {
    throw new ShapeError(`ctcLoss expects 1D or 2D targets; got ${targets.ndim}D`);
  }
  if (inputLengths.ndim !== 1) {
    throw new ShapeError(`ctcLoss expects 1D inputLengths; got ${inputLengths.ndim}D`);
  }
  if (targetLengths.ndim !== 1) {
    throw new ShapeError(`ctcLoss expects 1D targetLengths; got ${targetLengths.ndim}D`);
  }

  const T = logProbs.shape[0] ?? 0;
  const N = logProbs.shape[1] ?? 0;
  const C = logProbs.shape[2] ?? 0;

  if ((inputLengths.shape[0] ?? 0) !== N) {
    throw new ShapeError(
      `inputLengths length ${inputLengths.shape[0]} doesn't match batch size ${N}`
    );
  }
  if ((targetLengths.shape[0] ?? 0) !== N) {
    throw new ShapeError(
      `targetLengths length ${targetLengths.shape[0]} doesn't match batch size ${N}`
    );
  }
  if (targets.ndim === 2 && (targets.shape[0] ?? 0) !== N) {
    throw new ShapeError(`padded targets have ${targets.shape[0]} rows but batch size is ${N}`);
  }

  if (!Number.isInteger(blank) || blank < 0 || blank >= C) {
    throw new InvalidParameterError(
      `blank must be an integer in [0, ${C}); got ${blank}`,
      "blank",
      blank
    );
  }

  const lp = numbersOf(logProbs, "ctcLoss");
  const tgt = numbersOf(targets, "ctcLoss");
  const inLens = readLengths(inputLengths, "inputLengths", "ctcLoss");
  const tgtLens = readLengths(targetLengths, "targetLengths", "ctcLoss");

  const padded = targets.ndim === 2;
  const paddedWidth = padded ? (targets.shape[1] ?? 0) : 0;
  let totalTargets = 0;
  for (let b = 0; b < N; b++) {
    const tgtLen = tgtLens[b] ?? 0;
    if (tgtLen < 0 || (padded && tgtLen > paddedWidth)) {
      throw new InvalidParameterError(
        padded
          ? `targetLengths[${b}] = ${tgtLen} out of range [0, ${paddedWidth}]`
          : `targetLengths[${b}] = ${tgtLen} must be >= 0`,
        "targetLengths",
        tgtLen
      );
    }
    totalTargets += tgtLen;
  }
  if (!padded && totalTargets > targets.size) {
    throw new ShapeError(
      `targets has ${targets.size} elements but targetLengths sum to ${totalTargets}`
    );
  }

  const dtype = lossDtypeOf(logProbs);
  const samples: CtcSample[] = [];

  let targetOffset = 0;
  for (let b = 0; b < N; b++) {
    const inpLen = inLens[b] ?? 0;
    const tgtLen = tgtLens[b] ?? 0;

    if (inpLen < 0 || inpLen > T) {
      throw new InvalidParameterError(
        `inputLengths[${b}] = ${inpLen} out of range [0, ${T}]`,
        "inputLengths",
        inpLen
      );
    }

    // Read target labels for this batch element
    const labelBase = padded ? b * paddedWidth : targetOffset;
    const labels: number[] = [];
    for (let i = 0; i < tgtLen; i++) {
      const label = tgt[labelBase + i] ?? Number.NaN;
      if (!Number.isInteger(label) || label < 0 || label >= C) {
        throw new InvalidParameterError(
          `target label ${label} out of range [0, ${C})`,
          "targets",
          label
        );
      }
      if (label === blank) {
        throw new InvalidParameterError(
          `target label must not equal blank index ${blank}`,
          "targets",
          label
        );
      }
      labels.push(label);
    }
    targetOffset += tgtLen;

    // Extended label sequence with interleaved blanks:
    // [blank, l0, blank, l1, blank, ..., l_{L-1}, blank]
    const extended = new Int32Array(2 * tgtLen + 1);
    for (let s = 0; s < extended.length; s++) {
      extended[s] = s % 2 === 0 ? blank : (labels[(s - 1) / 2] ?? 0);
    }
    samples.push({ inputLength: inpLen, extended });
  }

  const rawLosses = newLossBuffer(dtype, N);
  const feasible = new Uint8Array(N);
  for (const [b, sample] of samples.entries()) {
    const loss = ctcSampleLoss(lp, N, C, b, sample, null);
    feasible[b] = Number.isFinite(loss) ? 1 : 0;
    rawLosses[b] = zeroInfinity && loss === Infinity ? 0 : loss;
  }
  const rawTensor = Tensor.fromTypedArray({
    data: rawLosses,
    shape: [N],
    dtype,
    device: logProbs.device,
  });
  // PyTorch's "mean" divides each sample by its target length (at least 1) before averaging.
  const lengthDivisors = newLossBuffer(dtype, N);
  for (let b = 0; b < N; b++) lengthDivisors[b] = Math.max(tgtLens[b] ?? 0, 1);
  const divisors = Tensor.fromTypedArray({
    data: lengthDivisors,
    shape: [N],
    dtype,
    device: logProbs.device,
  });

  if (GradTensor.isGradTensor(logProbsArg)) {
    let cachedGrad: Float64Array | null = null;
    const gradient = (): Float64Array => {
      if (cachedGrad !== null) return cachedGrad;
      const out = new Float64Array(logProbs.size);
      for (const [b, sample] of samples.entries()) {
        // An infeasible sample (infinite loss) has no gradient.
        if (feasible[b] === 1) ctcSampleLoss(lp, N, C, b, sample, out);
      }
      cachedGrad = out;
      return out;
    };
    const perSample = customOp(rawTensor, [
      [
        logProbsArg,
        (go: Tensor): Tensor => {
          const upstream = numbersOf(go, "ctcLoss");
          const base = gradient();
          const result = newLossBuffer(dtype, base.length);
          for (let t = 0; t < T; t++) {
            for (let b = 0; b < N; b++) {
              const scale = upstream[b] ?? 0;
              const row = (t * N + b) * C;
              for (let c = 0; c < C; c++) result[row + c] = scale * (base[row + c] ?? 0);
            }
          }
          return Tensor.fromTypedArray({
            data: result,
            shape: logProbs.shape.slice(),
            dtype,
            device: logProbs.device,
          });
        },
      ],
    ]);
    if (reduction === "mean") return perSample.div(divisors).mean();
    return reduceGradLoss(perSample, reduction);
  }

  if (reduction === "mean") return reduceLoss(div(rawTensor, divisors), reduction);
  return reduceLoss(rawTensor, reduction);
}

/**
 * Margin Ranking Loss.
 *
 * Creates a criterion that measures the loss given inputs x1, x2 and a
 * label tensor y (containing 1 or -1).
 *
 * **Formula**: loss_i = max(0, -y_i * (x1_i - x2_i) + margin)
 *
 * If y == 1, x1 should be ranked higher (larger) than x2.
 * If y == -1, x2 should be ranked higher (larger) than x1.
 *
 * `x1` and `x2` must have the same shape. `y` must have that shape too, or contain a
 * single element that applies to every pair. Passing a GradTensor as `x1` returns a
 * GradTensor that supports `.backward()`.
 *
 * @param x1 - First input tensor
 * @param x2 - Second input tensor (same shape as x1)
 * @param y - Label tensor containing 1 or -1 (same shape as x1)
 * @param margin - Margin value (default: 0)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function marginRankingLoss(
  x1: Tensor,
  x2: Tensor,
  y: Tensor,
  margin?: number,
  reduction?: LossReduction
): Tensor;
export function marginRankingLoss(
  x1: GradTensor,
  x2: AnyTensor,
  y: AnyTensor,
  margin?: number,
  reduction?: LossReduction
): GradTensor;
export function marginRankingLoss(
  x1: AnyTensor,
  x2: AnyTensor,
  y: AnyTensor,
  margin?: number,
  reduction?: LossReduction
): AnyTensor;
export function marginRankingLoss(
  x1: AnyTensor,
  x2: AnyTensor,
  y: AnyTensor,
  margin = 0,
  reduction: LossReduction = "mean"
): AnyTensor {
  validateReduction(reduction, "marginRankingLoss");

  const x2Tensor = toTensor(x2);
  const yTensor = toTensor(y);
  if (!shapesEqual(toTensor(x1).shape, x2Tensor.shape)) {
    throw new ShapeError(
      `marginRankingLoss: x1 and x2 must have the same shape; got [${toTensor(x1).shape}] vs [${x2Tensor.shape}]`
    );
  }
  if (
    !shapesEqual(x2Tensor.shape, yTensor.shape) &&
    yTensor.size !== 1 &&
    yTensor.size !== x2Tensor.size
  ) {
    throw new ShapeError(
      `marginRankingLoss: y must have as many elements as x1 or a single element; got [${yTensor.shape}] vs [${x2Tensor.shape}]`
    );
  }

  if (GradTensor.isGradTensor(x1)) {
    const dtype = numericDtype(x1, "marginRankingLoss");
    const b = asDtype(
      GradTensor.isGradTensor(x2) ? x2 : GradTensor.fromTensor(x2, { requiresGrad: false }),
      dtype
    );
    let label = asDtype(
      GradTensor.isGradTensor(y) ? y : GradTensor.fromTensor(y, { requiresGrad: false }),
      dtype
    );
    if (label.size > 1 && !shapesEqual(label.shape, x1.shape)) {
      // Same element count, different shape (for example (N, 1) vs (N,)): match element-wise.
      label = label.reshape([...x1.shape]);
    }
    const marginScalar = GradTensor.scalar(margin, { dtype });
    // max(0, -y * (x1 - x2) + margin)
    const loss = x1.sub(b).mul(label).neg().add(marginScalar).relu();
    return reduceGradLoss(loss, reduction);
  }

  const x1t = toTensor(x1);
  ensureNumeric(x1t, "marginRankingLoss");
  ensureNumeric(x2Tensor, "marginRankingLoss");
  ensureNumeric(yTensor, "marginRankingLoss");

  const a = numbersOf(x1t, "marginRankingLoss");
  const b = numbersOf(x2Tensor, "marginRankingLoss");
  const labels = numbersOf(yTensor, "marginRankingLoss");
  const single = yTensor.size === 1;

  const n = x1t.size;
  const dtype = lossDtypeOf(x1t);
  const lossData = newLossBuffer(dtype, n);
  for (let i = 0; i < n; i++) {
    const label = labels[single ? 0 : i] ?? 0;
    lossData[i] = Math.max(0, -label * ((a[i] ?? 0) - (b[i] ?? 0)) + margin);
  }

  return reduceLoss(
    Tensor.fromTypedArray({ data: lossData, shape: x1t.shape.slice(), dtype, device: x1t.device }),
    reduction
  );
}

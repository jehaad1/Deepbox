import { DTypeError, getElementAsNumber, InvalidParameterError, ShapeError } from "../../core";
import {
  abs,
  add,
  clip,
  log,
  mean,
  mul,
  neg,
  pow,
  sqrt,
  sub,
  sum,
  Tensor,
  tensor,
} from "../../ndarray";
import { offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";

export {
  binaryCrossEntropyWithLogitsLoss,
  crossEntropyLoss,
} from "./crossEntropy";

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if ((a[i] ?? 0) !== (b[i] ?? 0)) return false;
  }
  return true;
}

function ensureSameShape(a: Tensor, b: Tensor, context: string): void {
  if (!shapesEqual(a.shape, b.shape)) {
    throw new ShapeError(`Shape mismatch in ${context}: [${a.shape}] vs [${b.shape}]`);
  }
}

function ensureNumeric(t: Tensor, context: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
}

type NumericTensorData = Exclude<Tensor["data"], string[]>;

function validateReduction(reduction: "mean" | "sum" | "none", context: string): void {
  if (reduction !== "mean" && reduction !== "sum" && reduction !== "none") {
    throw new InvalidParameterError(
      `${context} reduction must be 'mean', 'sum', or 'none'`,
      "reduction",
      reduction
    );
  }
}

function readNumericFlat(
  data: NumericTensorData,
  flat: number,
  logicalStrides: readonly number[],
  strides: readonly number[],
  offset: number
): number {
  const dataOffset = offsetFromFlatIndex(flat, logicalStrides, strides, offset);
  return getElementAsNumber(data, dataOffset);
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
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @example
 * ```ts
 * import { mseLoss } from 'deepbox/nn/losses';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const predictions = tensor([2.5, 0.0, 2.1, 7.8]);
 * const targets = tensor([3.0, -0.5, 2.0, 8.0]);
 * const loss = mseLoss(predictions, targets); // Scalar tensor
 * ```
 *
 * @category Loss Functions
 */
export function mseLoss(
  predictions: Tensor,
  targets: Tensor,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "mseLoss");
  ensureNumeric(predictions, "mseLoss");
  ensureNumeric(targets, "mseLoss");
  ensureSameShape(predictions, targets, "mseLoss");

  const diff = sub(predictions, targets);
  const squaredDiff = pow(diff, tensor(2, { dtype: diff.dtype, device: diff.device }));

  if (reduction === "none") {
    return squaredDiff;
  }
  if (reduction === "sum") {
    return sum(squaredDiff);
  }
  return mean(squaredDiff);
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
 * - More robust to outliers than MSE
 *
 * **Properties:**
 * - Always non-negative
 * - Linear penalty for errors
 * - Less sensitive to outliers than MSE
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @category Loss Functions
 */
export function maeLoss(
  predictions: Tensor,
  targets: Tensor,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "maeLoss");
  ensureNumeric(predictions, "maeLoss");
  ensureNumeric(targets, "maeLoss");
  ensureSameShape(predictions, targets, "maeLoss");

  const diff = sub(predictions, targets);
  const absDiff = abs(diff);

  if (reduction === "none") {
    return absDiff;
  }
  if (reduction === "sum") {
    return sum(absDiff);
  }
  return mean(absDiff);
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
 * - Requires predictions in range (0, 1) - use sigmoid activation
 * - Targets should be 0 or 1
 * - Numerically stable with epsilon for log
 *
 * @param predictions - Predicted probabilities (0 to 1)
 * @param targets - True binary labels (0 or 1)
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @category Loss Functions
 */
export function binaryCrossEntropyLoss(
  predictions: Tensor,
  targets: Tensor,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "binaryCrossEntropyLoss");
  ensureNumeric(predictions, "binaryCrossEntropyLoss");
  ensureNumeric(targets, "binaryCrossEntropyLoss");
  ensureSameShape(predictions, targets, "binaryCrossEntropyLoss");

  const epsilon = 1e-7;
  const predClamped = clip(predictions, epsilon, 1 - epsilon);

  const logPred = log(predClamped);
  const term1 = mul(targets, logPred);

  const one = tensor(1, {
    dtype: predictions.dtype === "float64" ? "float64" : "float32",
    device: predictions.device,
  });
  const oneMinusTargets = sub(one, targets);
  const oneMinusPred = sub(one, predClamped);
  const logOneMinusPred = log(oneMinusPred);
  const term2 = mul(oneMinusTargets, logOneMinusPred);

  const loss = neg(add(term1, term2));

  if (reduction === "none") {
    return loss;
  }
  if (reduction === "sum") {
    return sum(loss);
  }
  return mean(loss);
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
 * @param predictions - Predicted values
 * @param targets - True target values
 * @returns Scalar loss value
 *
 * @category Loss Functions
 */
export function rmseLoss(predictions: Tensor, targets: Tensor): Tensor {
  ensureNumeric(predictions, "rmseLoss");
  ensureNumeric(targets, "rmseLoss");
  ensureSameShape(predictions, targets, "rmseLoss");

  const mse = mseLoss(predictions, targets, "mean");
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
 * - Robust to outliers while maintaining MSE benefits for small errors
 *
 * **Properties:**
 * - Quadratic for small errors (like MSE)
 * - Linear for large errors (like MAE)
 * - Controlled by delta parameter
 *
 * @param predictions - Predicted values
 * @param targets - True target values
 * @param delta - Threshold where loss transitions from quadratic to linear
 * @param reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Scalar loss value (or tensor if reduction='none')
 *
 * @category Loss Functions
 */
export function huberLoss(
  predictions: Tensor,
  targets: Tensor,
  delta = 1.0,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "huberLoss");
  ensureNumeric(predictions, "huberLoss");
  ensureNumeric(targets, "huberLoss");
  ensureSameShape(predictions, targets, "huberLoss");

  if (!Number.isFinite(delta) || delta <= 0) {
    throw new InvalidParameterError(`delta must be positive; got ${delta}`, "delta", delta);
  }

  const diff = sub(predictions, targets);
  const absDiff = abs(diff);

  const absData = absDiff.data;
  if (Array.isArray(absData)) {
    throw new DTypeError("huberLoss does not support string dtype");
  }
  const dtype = predictions.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(diff.size) : new Float32Array(diff.size);
  const logicalStrides = computeStrides(absDiff.shape);
  for (let i = 0; i < diff.size; i++) {
    const absVal = readNumericFlat(absData, i, logicalStrides, absDiff.strides, absDiff.offset);
    if (absVal <= delta) {
      lossData[i] = 0.5 * absVal * absVal;
    } else {
      lossData[i] = delta * (absVal - 0.5 * delta);
    }
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: predictions.shape,
    dtype,
    device: predictions.device,
  });

  if (reduction === "none") {
    return loss;
  }
  if (reduction === "sum") {
    return sum(loss);
  }
  return mean(loss);
}

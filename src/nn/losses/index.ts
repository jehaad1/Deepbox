/**
 * Neural network loss functions (`deepbox/nn` losses barrel).
 *
 * @see https://deepbox.dev/docs/nn-losses
 */
import {
  DTypeError,
  getElementAsNumber,
  InvalidParameterError,
  ShapeError,
  shapesEqual,
} from "../../core";
import type { AnyTensor } from "../../ndarray";
import {
  abs,
  add,
  clip,
  GradTensor,
  log,
  mean,
  mul,
  neg,
  pow,
  reshape,
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
} from "./cross_entropy";

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
  reduction?: "mean" | "sum" | "none"
): Tensor;
export function mseLoss(
  predictions: GradTensor,
  targets: GradTensor,
  reduction?: "mean" | "sum" | "none"
): GradTensor;
export function mseLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: "mean" | "sum" | "none" = "mean"
): AnyTensor {
  validateReduction(reduction, "mseLoss");

  // GradTensor path — preserves computation graph for .backward()
  if (GradTensor.isGradTensor(predictions)) {
    const pred = predictions;
    const tgt = GradTensor.isGradTensor(targets)
      ? targets
      : GradTensor.fromTensor(targets as Tensor, { requiresGrad: false });
    const diff = pred.sub(tgt);
    const squared = diff.mul(diff);
    if (reduction === "none") return squared;
    if (reduction === "sum") return squared.sum();
    return squared.mean();
  }

  // Plain Tensor path
  let preds = predictions as Tensor;
  let tgts = GradTensor.isGradTensor(targets) ? targets.tensor : (targets as Tensor);
  ensureNumeric(preds, "mseLoss");
  ensureNumeric(tgts, "mseLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "mseLoss");

  const diff = sub(preds, tgts);
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
  reduction?: "mean" | "sum" | "none"
): Tensor;
export function maeLoss(
  predictions: GradTensor,
  targets: GradTensor,
  reduction?: "mean" | "sum" | "none"
): GradTensor;
export function maeLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: "mean" | "sum" | "none" = "mean"
): AnyTensor {
  validateReduction(reduction, "maeLoss");

  if (GradTensor.isGradTensor(predictions)) {
    const pred = predictions;
    const tgt = GradTensor.isGradTensor(targets)
      ? targets
      : GradTensor.fromTensor(targets as Tensor, { requiresGrad: false });
    const diff = pred.sub(tgt);
    const absDiff = diff.abs();
    if (reduction === "none") return absDiff;
    if (reduction === "sum") return absDiff.sum();
    return absDiff.mean();
  }

  let preds = predictions as Tensor;
  let tgts = GradTensor.isGradTensor(targets) ? targets.tensor : (targets as Tensor);
  ensureNumeric(preds, "maeLoss");
  ensureNumeric(tgts, "maeLoss");
  [preds, tgts] = alignShapes(preds, tgts);
  ensureSameShape(preds, tgts, "maeLoss");

  const diff = sub(preds, tgts);
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
  reduction?: "mean" | "sum" | "none"
): Tensor;
export function binaryCrossEntropyLoss(
  predictions: GradTensor,
  targets: GradTensor,
  reduction?: "mean" | "sum" | "none"
): GradTensor;
export function binaryCrossEntropyLoss(
  predictions: AnyTensor,
  targets: AnyTensor,
  reduction: "mean" | "sum" | "none" = "mean"
): AnyTensor {
  validateReduction(reduction, "binaryCrossEntropyLoss");

  if (GradTensor.isGradTensor(predictions)) {
    const pred = predictions;
    const tgt = GradTensor.isGradTensor(targets)
      ? targets
      : GradTensor.fromTensor(targets as Tensor, { requiresGrad: false });
    const epsilon = 1e-7;
    const predClamped = pred.clip(epsilon, 1 - epsilon);
    const logPred = predClamped.log();
    const term1 = tgt.mul(logPred);
    const one = GradTensor.scalar(1, {
      dtype: pred.dtype === "float64" ? "float64" : (pred.dtype as "float32"),
    });
    const oneMinusTargets = one.sub(tgt);
    const oneMinusPred = one.sub(predClamped);
    const logOneMinusPred = oneMinusPred.log();
    const term2 = oneMinusTargets.mul(logOneMinusPred);
    const loss = term1.add(term2).neg();
    if (reduction === "none") return loss;
    if (reduction === "sum") return loss.sum();
    return loss.mean();
  }

  const preds = predictions as Tensor;
  const tgts = GradTensor.isGradTensor(targets) ? targets.tensor : (targets as Tensor);
  ensureNumeric(preds, "binaryCrossEntropyLoss");
  ensureNumeric(tgts, "binaryCrossEntropyLoss");
  ensureSameShape(preds, tgts, "binaryCrossEntropyLoss");

  const epsilon = 1e-7;
  const predClamped = clip(preds, epsilon, 1 - epsilon);

  const logPred = log(predClamped);
  const term1 = mul(tgts, logPred);

  const one = tensor(1, {
    dtype: preds.dtype === "float64" ? "float64" : "float32",
    device: preds.device,
  });
  const oneMinusTargets = sub(one, tgts);
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
export function rmseLoss(predictions: Tensor, targets: Tensor): Tensor;
export function rmseLoss(predictions: GradTensor, targets: GradTensor): GradTensor;
export function rmseLoss(predictions: AnyTensor, targets: AnyTensor): AnyTensor {
  if (GradTensor.isGradTensor(predictions)) {
    const pred = predictions;
    const tgt = GradTensor.isGradTensor(targets)
      ? targets
      : GradTensor.fromTensor(targets as Tensor, { requiresGrad: false });
    const diff = pred.sub(tgt);
    const squared = diff.mul(diff);
    return squared.mean().sqrt();
  }

  const preds = predictions as Tensor;
  const tgts = GradTensor.isGradTensor(targets) ? targets.tensor : (targets as Tensor);
  ensureNumeric(preds, "rmseLoss");
  ensureNumeric(tgts, "rmseLoss");
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
  [predictions, targets] = alignShapes(predictions, targets);
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

/**
 * Negative Log Likelihood (NLL) loss.
 *
 * Expects log-probabilities as input (use LogSoftmax before this loss).
 *
 * **Formula**: NLL = -sum(log_probs[i][target[i]]) / N
 *
 * @param logProbs - Log-probabilities of shape (N, C) where C = number of classes
 * @param targets - Class indices of shape (N,) with values in [0, C)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function nllLoss(
  logProbs: Tensor,
  targets: Tensor,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "nllLoss");
  ensureNumeric(logProbs, "nllLoss");
  ensureNumeric(targets, "nllLoss");

  if (logProbs.ndim !== 2) {
    throw new ShapeError(`nllLoss expects 2D log-probabilities; got ${logProbs.ndim}D`);
  }
  if (targets.ndim !== 1) {
    throw new ShapeError(`nllLoss expects 1D targets; got ${targets.ndim}D`);
  }

  const N = logProbs.shape[0] ?? 0;
  const C = logProbs.shape[1] ?? 0;

  if ((targets.shape[0] ?? 0) !== N) {
    throw new ShapeError(`targets length ${targets.shape[0]} doesn't match batch size ${N}`);
  }

  const logData = logProbs.data;
  const tgtData = targets.data;
  if (Array.isArray(logData) || Array.isArray(tgtData)) {
    throw new DTypeError("nllLoss does not support string dtype");
  }

  const logStrides = computeStrides(logProbs.shape);
  const tgtStrides = computeStrides(targets.shape);
  const dtype = logProbs.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(N) : new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const tgtIdx = Math.round(
      readNumericFlat(tgtData, i, tgtStrides, targets.strides, targets.offset)
    );
    if (tgtIdx < 0 || tgtIdx >= C) {
      throw new InvalidParameterError(
        `nllLoss target index ${tgtIdx} out of range [0, ${C})`,
        "targets",
        tgtIdx
      );
    }
    const flatIdx = i * C + tgtIdx;
    const logVal = readNumericFlat(logData, flatIdx, logStrides, logProbs.strides, logProbs.offset);
    lossData[i] = -logVal;
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: [N],
    dtype,
    device: logProbs.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
}

/**
 * KL Divergence loss.
 *
 * Measures how one probability distribution diverges from a second.
 *
 * **Formula**: KL(P || Q) = sum(P * (log(P) - log(Q)))
 *
 * Input should be log-probabilities, target should be probabilities.
 *
 * @param input - Log-probabilities (from LogSoftmax)
 * @param target - Target probability distribution
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function klDivLoss(
  input: Tensor,
  target: Tensor,
  reduction: "mean" | "sum" | "batchmean" | "none" = "mean"
): Tensor {
  ensureNumeric(input, "klDivLoss");
  ensureNumeric(target, "klDivLoss");
  [input, target] = alignShapes(input, target);
  ensureSameShape(input, target, "klDivLoss");

  // KL(target || input) = target * (log(target) - input)
  // Since input is already log-probs, and target is probs:
  // loss = target * (log(target) - input)
  // We skip terms where target == 0 (0 * log(0) = 0 by convention)
  const tgtData = target.data;
  const inpData = input.data;
  if (Array.isArray(tgtData) || Array.isArray(inpData)) {
    throw new DTypeError("klDivLoss does not support string dtype");
  }

  const dtype = input.dtype === "float64" ? "float64" : "float32";
  const lossData =
    dtype === "float64" ? new Float64Array(input.size) : new Float32Array(input.size);
  const inpStrides = computeStrides(input.shape);
  const tgtStrides = computeStrides(target.shape);

  for (let i = 0; i < input.size; i++) {
    const t = readNumericFlat(tgtData, i, tgtStrides, target.strides, target.offset);
    const q = readNumericFlat(inpData, i, inpStrides, input.strides, input.offset);
    if (t > 0) {
      lossData[i] = t * (Math.log(t) - q);
    }
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: input.shape,
    dtype,
    device: input.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  if (reduction === "batchmean") {
    const batchSize = input.shape[0] ?? 1;
    const totalLoss = sum(loss);
    const totalData = totalLoss.data;
    if (Array.isArray(totalData)) {
      throw new DTypeError("klDivLoss does not support string dtype");
    }
    const totalVal = getElementAsNumber(totalData, 0);
    return Tensor.fromTypedArray({
      data:
        dtype === "float64"
          ? new Float64Array([totalVal / batchSize])
          : new Float32Array([totalVal / batchSize]),
      shape: [],
      dtype,
      device: input.device,
    });
  }
  return mean(loss);
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
 * @param predictions - Predicted values
 * @param targets - Target values
 * @param beta - Threshold for switching between L1 and L2 (default: 1.0)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function smoothL1Loss(
  predictions: Tensor,
  targets: Tensor,
  beta = 1.0,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "smoothL1Loss");
  ensureNumeric(predictions, "smoothL1Loss");
  ensureNumeric(targets, "smoothL1Loss");
  [predictions, targets] = alignShapes(predictions, targets);
  ensureSameShape(predictions, targets, "smoothL1Loss");

  if (!Number.isFinite(beta) || beta <= 0) {
    throw new InvalidParameterError(`beta must be positive; got ${beta}`, "beta", beta);
  }

  const diff = sub(predictions, targets);
  const absDiff = abs(diff);
  const absData = absDiff.data;
  if (Array.isArray(absData)) {
    throw new DTypeError("smoothL1Loss does not support string dtype");
  }

  const dtype = predictions.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(diff.size) : new Float32Array(diff.size);
  const logicalStrides = computeStrides(absDiff.shape);

  for (let i = 0; i < diff.size; i++) {
    const absVal = readNumericFlat(absData, i, logicalStrides, absDiff.strides, absDiff.offset);
    if (absVal < beta) {
      lossData[i] = (0.5 * absVal * absVal) / beta;
    } else {
      lossData[i] = absVal - 0.5 * beta;
    }
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: predictions.shape,
    dtype,
    device: predictions.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
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
 * @param x1 - First input tensor
 * @param x2 - Second input tensor
 * @param y - Labels: 1 (similar) or -1 (dissimilar)
 * @param margin - Margin for dissimilar pairs (default: 0)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function cosineEmbeddingLoss(
  x1: Tensor,
  x2: Tensor,
  y: Tensor,
  margin = 0,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "cosineEmbeddingLoss");
  ensureNumeric(x1, "cosineEmbeddingLoss");
  ensureNumeric(x2, "cosineEmbeddingLoss");
  ensureNumeric(y, "cosineEmbeddingLoss");

  if (x1.ndim < 1 || x2.ndim < 1) {
    throw new ShapeError("cosineEmbeddingLoss expects at least 1D inputs");
  }

  const x1Data = x1.data;
  const x2Data = x2.data;
  const yData = y.data;
  if (Array.isArray(x1Data) || Array.isArray(x2Data) || Array.isArray(yData)) {
    throw new DTypeError("cosineEmbeddingLoss does not support string dtype");
  }

  // For 2D inputs: (N, D), compute cosine similarity per row
  // For 1D inputs: (D,), compute single cosine similarity
  const batchSize = x1.ndim >= 2 ? (x1.shape[0] ?? 1) : 1;
  const dim = x1.ndim >= 2 ? (x1.shape[1] ?? 1) : (x1.shape[0] ?? 1);

  const dtype = x1.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(batchSize) : new Float32Array(batchSize);
  const x1Strides = computeStrides(x1.shape);
  const x2Strides = computeStrides(x2.shape);
  const yStrides = computeStrides(y.shape);

  for (let b = 0; b < batchSize; b++) {
    // Compute cosine similarity for this batch element
    let dot = 0,
      norm1 = 0,
      norm2 = 0;
    for (let d = 0; d < dim; d++) {
      const flatIdx = b * dim + d;
      const v1 = readNumericFlat(x1Data, flatIdx, x1Strides, x1.strides, x1.offset);
      const v2 = readNumericFlat(x2Data, flatIdx, x2Strides, x2.strides, x2.offset);
      dot += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }
    const cosSim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8);

    const label = readNumericFlat(yData, b, yStrides, y.strides, y.offset);
    if (label === 1) {
      lossData[b] = 1 - cosSim;
    } else {
      lossData[b] = Math.max(0, cosSim - margin);
    }
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: [batchSize],
    dtype,
    device: x1.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
}

/**
 * Triplet Margin Loss for metric learning.
 *
 * **Formula**: loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
 *
 * @param anchor - Anchor embeddings
 * @param positive - Positive (similar) embeddings
 * @param negative - Negative (dissimilar) embeddings
 * @param margin - Margin between positive and negative distances (default: 1.0)
 * @param reduction - How to reduce the loss
 *
 * @category Loss Functions
 */
export function tripletMarginLoss(
  anchor: Tensor,
  positive: Tensor,
  negative: Tensor,
  margin = 1.0,
  reduction: "mean" | "sum" | "none" = "mean"
): Tensor {
  validateReduction(reduction, "tripletMarginLoss");
  ensureNumeric(anchor, "tripletMarginLoss");
  ensureNumeric(positive, "tripletMarginLoss");
  ensureNumeric(negative, "tripletMarginLoss");
  ensureSameShape(anchor, positive, "tripletMarginLoss (anchor vs positive)");
  ensureSameShape(anchor, negative, "tripletMarginLoss (anchor vs negative)");

  const ancData = anchor.data;
  const posData = positive.data;
  const negData = negative.data;
  if (Array.isArray(ancData) || Array.isArray(posData) || Array.isArray(negData)) {
    throw new DTypeError("tripletMarginLoss does not support string dtype");
  }

  const batchSize = anchor.ndim >= 2 ? (anchor.shape[0] ?? 1) : 1;
  const dim = anchor.ndim >= 2 ? (anchor.shape[1] ?? 1) : (anchor.shape[0] ?? 1);

  const dtype = anchor.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(batchSize) : new Float32Array(batchSize);
  const ancStrides = computeStrides(anchor.shape);
  const posStrides = computeStrides(positive.shape);
  const negStrides = computeStrides(negative.shape);

  for (let b = 0; b < batchSize; b++) {
    let distPos = 0,
      distNeg = 0;
    for (let d = 0; d < dim; d++) {
      const flatIdx = b * dim + d;
      const a = readNumericFlat(ancData, flatIdx, ancStrides, anchor.strides, anchor.offset);
      const p = readNumericFlat(posData, flatIdx, posStrides, positive.strides, positive.offset);
      const n = readNumericFlat(negData, flatIdx, negStrides, negative.strides, negative.offset);
      distPos += (a - p) * (a - p);
      distNeg += (a - n) * (a - n);
    }
    lossData[b] = Math.max(0, Math.sqrt(distPos) - Math.sqrt(distNeg) + margin);
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: [batchSize],
    dtype,
    device: anchor.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
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
 * @param input - Predicted means
 * @param target - Target values
 * @param variance - Predicted variances (must be positive)
 * @param options - Configuration options
 * @param options.full - Include constant log(2π) term (default: false)
 * @param options.eps - Small value for clamping variance (default: 1e-6)
 * @param options.reduction - How to reduce: 'mean', 'sum', or 'none'
 * @returns Loss value
 *
 * @category Loss Functions
 */
export function gaussianNLLLoss(
  input: Tensor,
  target: Tensor,
  variance: Tensor,
  options: {
    full?: boolean;
    eps?: number;
    reduction?: "mean" | "sum" | "none";
  } = {}
): Tensor {
  const { full = false, eps = 1e-6, reduction = "mean" } = options;
  validateReduction(reduction, "gaussianNLLLoss");
  ensureNumeric(input, "gaussianNLLLoss");
  ensureNumeric(target, "gaussianNLLLoss");
  ensureNumeric(variance, "gaussianNLLLoss");
  ensureSameShape(input, target, "gaussianNLLLoss (input vs target)");
  ensureSameShape(input, variance, "gaussianNLLLoss (input vs variance)");

  if (!Number.isFinite(eps) || eps < 0) {
    throw new InvalidParameterError(`eps must be non-negative; got ${eps}`, "eps", eps);
  }

  const inpData = input.data;
  const tgtData = target.data;
  const varData = variance.data;
  if (Array.isArray(inpData) || Array.isArray(tgtData) || Array.isArray(varData)) {
    throw new DTypeError("gaussianNLLLoss does not support string dtype");
  }

  const dtype = input.dtype === "float64" ? "float64" : "float32";
  const lossData =
    dtype === "float64" ? new Float64Array(input.size) : new Float32Array(input.size);
  const inpStrides = computeStrides(input.shape);
  const tgtStrides = computeStrides(target.shape);
  const varStrides = computeStrides(variance.shape);
  const LOG2PI = Math.log(2 * Math.PI);

  for (let i = 0; i < input.size; i++) {
    const mu = readNumericFlat(inpData, i, inpStrides, input.strides, input.offset);
    const t = readNumericFlat(tgtData, i, tgtStrides, target.strides, target.offset);
    let v = readNumericFlat(varData, i, varStrides, variance.strides, variance.offset);
    // Clamp variance to eps
    if (v < eps) v = eps;

    const diff = mu - t;
    let lossVal = 0.5 * (Math.log(v) + (diff * diff) / v);
    if (full) {
      lossVal += 0.5 * LOG2PI;
    }
    lossData[i] = lossVal;
  }

  const lossT = Tensor.fromTypedArray({
    data: lossData,
    shape: input.shape,
    dtype,
    device: input.device,
  });

  if (reduction === "none") return lossT;
  if (reduction === "sum") return sum(lossT);
  return mean(lossT);
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
  options: {
    logInput?: boolean;
    full?: boolean;
    eps?: number;
    reduction?: "mean" | "sum" | "none";
  } = {}
): Tensor {
  const { logInput = true, full = false, eps = 1e-8, reduction = "mean" } = options;
  validateReduction(reduction, "poissonNLLLoss");
  ensureNumeric(input, "poissonNLLLoss");
  ensureNumeric(target, "poissonNLLLoss");
  ensureSameShape(input, target, "poissonNLLLoss");

  const inpData = input.data;
  const tgtData = target.data;
  if (Array.isArray(inpData) || Array.isArray(tgtData)) {
    throw new DTypeError("poissonNLLLoss does not support string dtype");
  }

  const dtype = input.dtype === "float64" ? "float64" : "float32";
  const lossData =
    dtype === "float64" ? new Float64Array(input.size) : new Float32Array(input.size);
  const inpStrides = computeStrides(input.shape);
  const tgtStrides = computeStrides(target.shape);

  for (let i = 0; i < input.size; i++) {
    const inp = readNumericFlat(inpData, i, inpStrides, input.strides, input.offset);
    const tgt = readNumericFlat(tgtData, i, tgtStrides, target.strides, target.offset);

    let lossVal: number;
    if (logInput) {
      // loss = exp(input) - target * input
      lossVal = Math.exp(inp) - tgt * inp;
    } else {
      // loss = input - target * log(input + eps)
      lossVal = inp - tgt * Math.log(inp + eps);
    }

    if (full) {
      // Add Stirling approximation: target * log(target) - target + 0.5 * log(2π * target)
      if (tgt > 1) {
        lossVal += tgt * Math.log(tgt) - tgt + 0.5 * Math.log(2 * Math.PI * tgt);
      }
    }

    lossData[i] = lossVal;
  }

  const lossT = Tensor.fromTypedArray({
    data: lossData,
    shape: input.shape,
    dtype,
    device: input.device,
  });

  if (reduction === "none") return lossT;
  if (reduction === "sum") return sum(lossT);
  return mean(lossT);
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

/**
 * Connectionist Temporal Classification (CTC) Loss.
 *
 * CTC loss for sequence-to-sequence models where the alignment between
 * input and output is unknown (e.g., speech recognition, OCR).
 *
 * **Algorithm**: Forward (alpha) computation in log-space over an extended
 * label sequence with interleaved blanks.
 *
 * @param logProbs - Log-probabilities of shape (T, N, C) where T = input
 *   length (time steps), N = batch size, C = number of classes (including blank)
 * @param targets - Concatenated target sequences (1D tensor of length = sum of
 *   all target lengths). Values must be in [0, C) and must not equal `blank`.
 * @param inputLengths - Lengths of each input sequence, shape (N,)
 * @param targetLengths - Lengths of each target sequence, shape (N,)
 * @param options - Configuration options
 * @param options.blank - Index of the blank label (default: 0)
 * @param options.reduction - How to reduce the loss: 'mean', 'sum', or 'none'
 * @returns Loss value
 *
 * @category Loss Functions
 */
export function ctcLoss(
  logProbs: Tensor,
  targets: Tensor,
  inputLengths: Tensor,
  targetLengths: Tensor,
  options: {
    blank?: number;
    reduction?: "mean" | "sum" | "none";
  } = {}
): Tensor {
  const { blank = 0, reduction = "mean" } = options;
  validateReduction(reduction, "ctcLoss");
  ensureNumeric(logProbs, "ctcLoss");
  ensureNumeric(targets, "ctcLoss");
  ensureNumeric(inputLengths, "ctcLoss");
  ensureNumeric(targetLengths, "ctcLoss");

  if (logProbs.ndim !== 3) {
    throw new ShapeError(`ctcLoss expects 3D logProbs (T, N, C); got ${logProbs.ndim}D`);
  }
  if (targets.ndim !== 1) {
    throw new ShapeError(`ctcLoss expects 1D targets; got ${targets.ndim}D`);
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

  if (!Number.isInteger(blank) || blank < 0 || blank >= C) {
    throw new InvalidParameterError(
      `blank must be an integer in [0, ${C}); got ${blank}`,
      "blank",
      blank
    );
  }

  const lpData = logProbs.data;
  const tgtData = targets.data;
  const ilData = inputLengths.data;
  const tlData = targetLengths.data;
  if (
    Array.isArray(lpData) ||
    Array.isArray(tgtData) ||
    Array.isArray(ilData) ||
    Array.isArray(tlData)
  ) {
    throw new DTypeError("ctcLoss does not support string dtype");
  }

  const lpStrides = computeStrides(logProbs.shape);
  const tgtStrides = computeStrides(targets.shape);
  const ilStrides = computeStrides(inputLengths.shape);
  const tlStrides = computeStrides(targetLengths.shape);

  const dtype = logProbs.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(N) : new Float32Array(N);

  let targetOffset = 0;
  for (let b = 0; b < N; b++) {
    const inpLen = Math.round(
      readNumericFlat(ilData, b, ilStrides, inputLengths.strides, inputLengths.offset)
    );
    const tgtLen = Math.round(
      readNumericFlat(tlData, b, tlStrides, targetLengths.strides, targetLengths.offset)
    );

    if (inpLen < 1 || inpLen > T) {
      throw new InvalidParameterError(
        `inputLengths[${b}] = ${inpLen} out of range [1, ${T}]`,
        "inputLengths",
        inpLen
      );
    }
    if (tgtLen < 1) {
      throw new InvalidParameterError(
        `targetLengths[${b}] = ${tgtLen} must be >= 1`,
        "targetLengths",
        tgtLen
      );
    }

    // Read target labels for this batch element
    const labels: number[] = [];
    for (let i = 0; i < tgtLen; i++) {
      const label = Math.round(
        readNumericFlat(tgtData, targetOffset + i, tgtStrides, targets.strides, targets.offset)
      );
      if (label < 0 || label >= C) {
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
    const S = 2 * tgtLen + 1;
    const extLabels: number[] = [];
    for (let s = 0; s < S; s++) {
      if (s % 2 === 0) {
        extLabels.push(blank);
      } else {
        extLabels.push(labels[(s - 1) / 2] ?? 0);
      }
    }

    if (inpLen < tgtLen) {
      lossData[b] = Infinity;
      continue;
    }

    // Forward algorithm in log-space
    const NEG_INF = -Infinity;
    let prev = new Float64Array(S);
    prev.fill(NEG_INF);

    // Initialization at t=0: can start at s=0 (blank) or s=1 (first label)
    prev[0] = readNumericFlat(
      lpData,
      0 * N * C + b * C + (extLabels[0] ?? 0),
      lpStrides,
      logProbs.strides,
      logProbs.offset
    );
    if (S > 1) {
      prev[1] = readNumericFlat(
        lpData,
        0 * N * C + b * C + (extLabels[1] ?? 0),
        lpStrides,
        logProbs.strides,
        logProbs.offset
      );
    }

    for (let t = 1; t < inpLen; t++) {
      const curr = new Float64Array(S);
      curr.fill(NEG_INF);

      for (let s = 0; s < S; s++) {
        let logAlpha: number = prev[s] ?? NEG_INF;

        if (s > 0) {
          logAlpha = logSumExp(logAlpha, prev[s - 1] ?? NEG_INF);
        }

        if (s > 1) {
          const currLabel = extLabels[s] ?? 0;
          const prevPrevLabel = extLabels[s - 2] ?? 0;
          if (currLabel !== blank && currLabel !== prevPrevLabel) {
            logAlpha = logSumExp(logAlpha, prev[s - 2] ?? NEG_INF);
          }
        }

        const lp = readNumericFlat(
          lpData,
          t * N * C + b * C + (extLabels[s] ?? 0),
          lpStrides,
          logProbs.strides,
          logProbs.offset
        );
        curr[s] = logAlpha + lp;
      }
      prev = curr;
    }

    // Total log-probability
    const logProb = logSumExp(prev[S - 1] ?? NEG_INF, prev[S - 2] ?? NEG_INF);
    lossData[b] = -logProb;
  }

  const ctcLossT = Tensor.fromTypedArray({
    data: lossData,
    shape: [N],
    dtype,
    device: logProbs.device,
  });

  if (reduction === "none") return ctcLossT;
  if (reduction === "sum") return sum(ctcLossT);
  return mean(ctcLossT);
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
  reduction?: "mean" | "sum" | "none"
): Tensor;
export function marginRankingLoss(
  x1: GradTensor,
  x2: GradTensor,
  y: GradTensor,
  margin?: number,
  reduction?: "mean" | "sum" | "none"
): GradTensor;
export function marginRankingLoss(
  x1: AnyTensor,
  x2: AnyTensor,
  y: AnyTensor,
  margin = 0,
  reduction: "mean" | "sum" | "none" = "mean"
): AnyTensor {
  validateReduction(reduction, "marginRankingLoss");

  if (GradTensor.isGradTensor(x1)) {
    const a = x1;
    const b = GradTensor.isGradTensor(x2)
      ? x2
      : GradTensor.fromTensor(x2 as Tensor, { requiresGrad: false });
    const label = GradTensor.isGradTensor(y)
      ? y
      : GradTensor.fromTensor(y as Tensor, { requiresGrad: false });
    const diff = a.sub(b);
    const marginScalar = GradTensor.scalar(margin, {
      dtype: a.dtype === "float64" ? "float64" : (a.dtype as "float32"),
    });
    const lossRaw = diff.mul(label).neg().add(marginScalar);
    const lossRelu = lossRaw.abs().add(lossRaw).mul(GradTensor.scalar(0.5));
    if (reduction === "none") return lossRelu;
    if (reduction === "sum") return lossRelu.sum();
    return lossRelu.mean();
  }

  let x1t = x1 as Tensor;
  let x2t = x2 as Tensor;
  let yt = y as Tensor;
  if (GradTensor.isGradTensor(x1t)) x1t = x1t.tensor;
  if (GradTensor.isGradTensor(x2t)) x2t = x2t.tensor;
  if (GradTensor.isGradTensor(yt)) yt = yt.tensor;
  ensureNumeric(x1t, "marginRankingLoss");
  ensureNumeric(x2t, "marginRankingLoss");
  ensureNumeric(yt, "marginRankingLoss");

  if (!shapesEqual(x1t.shape, x2t.shape)) {
    throw new ShapeError(
      `marginRankingLoss: x1 and x2 must have the same shape; got [${x1t.shape}] vs [${x2t.shape}]`
    );
  }
  if (!shapesEqual(x1t.shape, yt.shape) && yt.size !== x1t.size) {
    throw new ShapeError(
      `marginRankingLoss: y must be broadcastable to x1 shape; got [${yt.shape}] vs [${x1t.shape}]`
    );
  }

  const x1Data = x1t.data;
  const x2Data = x2t.data;
  const yData = yt.data;
  if (Array.isArray(x1Data) || Array.isArray(x2Data) || Array.isArray(yData)) {
    throw new DTypeError("marginRankingLoss does not support string dtype");
  }

  const n = x1t.size;
  const dtype = x1t.dtype === "float64" ? "float64" : "float32";
  const lossData = dtype === "float64" ? new Float64Array(n) : new Float32Array(n);

  const x1Strides = computeStrides(x1t.shape);
  const x2Strides = computeStrides(x2t.shape);
  const yStrides = computeStrides(yt.shape);

  for (let i = 0; i < n; i++) {
    const v1 = readNumericFlat(x1Data, i, x1Strides, x1t.strides, x1t.offset);
    const v2 = readNumericFlat(x2Data, i, x2Strides, x2t.strides, x2t.offset);
    const label = readNumericFlat(yData, yt.size === 1 ? 0 : i, yStrides, yt.strides, yt.offset);
    lossData[i] = Math.max(0, -label * (v1 - v2) + margin);
  }

  const loss = Tensor.fromTypedArray({
    data: lossData,
    shape: x1t.shape.slice(),
    dtype,
    device: x1t.device,
  });

  if (reduction === "none") return loss;
  if (reduction === "sum") return sum(loss);
  return mean(loss);
}

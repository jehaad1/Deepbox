import {
  DTypeError,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import { type AnyTensor, GradTensor, logSoftmaxGrad, Tensor } from "../../ndarray";

/**
 * Helper to convert class indices to one-hot encoded tensor.
 */
function toOneHot(indices: Tensor, numClasses: number): Tensor {
  const nSamples = indices.size;
  const outData = new Float32Array(nSamples * numClasses);

  const data = indices.data;
  if (Array.isArray(data)) {
    throw new DTypeError("crossEntropyLoss target indices must be numeric");
  }

  const stride0 = indices.strides[0] ?? 0;
  const base = indices.offset;

  for (let i = 0; i < nSamples; i++) {
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
    outData[i * numClasses + idx] = 1.0;
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [nSamples, numClasses],
    dtype: "float32",
    device: indices.device,
  });
}

/**
 * Cross Entropy Loss.
 *
 * Computes the cross entropy loss between predictions and targets.
 * Commonly used for multi-class classification problems.
 *
 * Supports both integer class indices and one-hot encoded probabilities for targets.
 *
 * **Formula**: L = -mean(sum(target * log_softmax(input), dim=1))
 *
 * @param input - Predicted logits of shape (n_samples, n_classes)
 * @param target - True labels. Either:
 *                 - Class indices of shape (n_samples,)
 *                 - Probabilities/One-hot of shape (n_samples, n_classes)
 * @returns Scalar loss value (GradTensor)
 *
 * @example
 * ```ts
 * import { crossEntropyLoss } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const pred = tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]);
 * const true_idx = tensor([0, 1]);
 * const loss = crossEntropyLoss(pred, true_idx);
 * ```
 */
export function crossEntropyLoss(input: Tensor, target: Tensor): number;
export function crossEntropyLoss(input: GradTensor, target: AnyTensor): GradTensor;
export function crossEntropyLoss(input: AnyTensor, target: AnyTensor): number | GradTensor;
export function crossEntropyLoss(input: AnyTensor, target: AnyTensor): number | GradTensor {
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

  let targetTensor = yTrue;

  // Handle class indices (1D)
  if (yTrue.ndim === 1) {
    // Class indices are discrete labels — gradients through them are meaningless.
    // Accept 1D GradTensor by extracting the underlying tensor for one-hot conversion.
    if (yTrue.shape[0] !== nSamples) {
      throw new ShapeError(
        `Target must have same number of samples as input; got ${yTrue.shape[0]} and ${nSamples}`
      );
    }
    // Convert to one-hot
    // We need to access the underlying tensor data
    const oneHot = toOneHot(yTrue.tensor, nClasses);
    targetTensor = GradTensor.fromTensor(oneHot, { requiresGrad: false });
  } else if (yTrue.ndim === 2) {
    if (yTrue.shape[0] !== nSamples || yTrue.shape[1] !== nClasses) {
      throw new ShapeError(
        "Target must be 1-dimensional class indices or have the same shape as input"
      );
    }
  } else {
    throw new ShapeError(`Target must be 1D (indices) or 2D (probs); got ${yTrue.ndim}D`);
  }

  // Compute Log Softmax
  const logProbs = logSoftmaxGrad(yPred, 1);

  // Compute NLL: -sum(target * log_prob) / N
  // Element-wise multiply
  const weighted = logProbs.mul(targetTensor);

  // Sum over classes (dim 1) -> (N,)
  const sampleLoss = weighted.sum(1);

  // Mean over batch -> scalar
  // Note: sum(1) returns (N,), mean() returns scalar.
  // But wait, `weighted` is negative log likelihood * target.
  // For one-hot target, sum(target * log_prob) is log_prob[class].
  // We want negative of that.
  // And then mean.

  const meanLoss = sampleLoss.mean().neg();
  if (!GradTensor.isGradTensor(input) && !targetIsGrad) {
    const data = meanLoss.tensor.data;
    if (Array.isArray(data)) {
      throw new DTypeError("crossEntropyLoss does not support string dtype");
    }
    if (data instanceof BigInt64Array) {
      const raw = getBigIntElement(data, meanLoss.tensor.offset);
      return Number(raw);
    }
    return getNumericElement(data, meanLoss.tensor.offset);
  }
  return meanLoss;
}

/**
 * Binary Cross Entropy Loss with logits.
 *
 * Combines sigmoid activation and binary cross entropy loss for numerical stability.
 *
 * @param input - Predicted logits of shape (n_samples,) or (n_samples, 1)
 * @param target - True binary labels of same shape as input
 * @returns Scalar loss value (GradTensor)
 */
export function binaryCrossEntropyWithLogitsLoss(input: Tensor, target: Tensor): number;
export function binaryCrossEntropyWithLogitsLoss(input: GradTensor, target: AnyTensor): GradTensor;
export function binaryCrossEntropyWithLogitsLoss(
  input: AnyTensor,
  target: AnyTensor
): number | GradTensor {
  const yPred = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
  const yTrue = GradTensor.isGradTensor(target)
    ? target
    : GradTensor.fromTensor(target, { requiresGrad: false });

  // Check shapes
  // Support (N,) and (N, 1)
  let pred = yPred;
  let truth = yTrue;

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
    throw new ShapeError(`Input must have shape (N,) or (N, 1)`);
  }
  if (truth.ndim !== 2 || truth.shape[1] !== 1) {
    throw new ShapeError(`Target must be 1-dimensional or have shape (N, 1)`);
  }
  if ((pred.shape[0] ?? 0) !== (truth.shape[0] ?? 0)) {
    throw new ShapeError(`Batch size mismatch`);
  }

  const predDtype = pred.dtype;
  if (predDtype === "string") {
    throw new DTypeError("Binary cross entropy does not support string dtype");
  }

  // max(x, 0) - x * z + log(1 + exp(-abs(x)))
  // We use autograd ops

  // term1 = max(x, 0) -> relu(x)
  const term1 = pred.relu();

  // term2 = x * z
  const term2 = pred.mul(truth);

  // term3 = log(1 + exp(-abs(x)))
  // abs(x) = relu(x) + relu(-x) or just use abs op if available?
  // GradTensor doesn't have abs yet?
  // We can use sign? Or just: abs(x) = sqrt(x^2)? No, gradient at 0.
  // Or: abs(x) = max(x, -x).
  // We have neg().
  // We have max() (which I added).
  // So abs(x) = max(x, x.neg())? No, max takes reduction axis.
  // Elementwise max is not exposed on GradTensor yet.
  // But wait, we can implement `softplus` (log(1+exp(x))) directly?
  // The formula uses `log(1 + exp(-abs(x)))`.
  // Alternative: `softplus(x)` if z=0, `softplus(x) - x` if z=1?
  // Standard stable BCEWithLogits:
  // loss = (1-z)*x + softplus(-x)   if x > 0
  // loss = (1-z)*x + x + softplus(-x) ? No.

  // Uses: max(x, 0) - x*z + log(1 + exp(-abs(x)))
  // To implement `abs(x)` without primitive:
  // `x.pow(2).sqrt()` is abs(x) but unstable grad at 0.
  // `relu(x) + relu(-x)` works.

  const negPred = pred.neg();
  const absPred = pred.relu().add(negPred.relu());

  const expNegAbs = absPred.neg().exp();
  const scalarDtype = expNegAbs.dtype;
  if (scalarDtype === "string") {
    throw new DTypeError("binaryCrossEntropyWithLogitsLoss does not support string dtype");
  }
  const one = GradTensor.scalar(1, { dtype: scalarDtype });
  const term3 = one.add(expNegAbs).log();

  const loss = term1.sub(term2).add(term3).mean();
  if (!GradTensor.isGradTensor(input) && !GradTensor.isGradTensor(target)) {
    const data = loss.tensor.data;
    if (Array.isArray(data)) {
      throw new DTypeError("binaryCrossEntropyWithLogitsLoss does not support string dtype");
    }
    if (data instanceof BigInt64Array) {
      const raw = getBigIntElement(data, loss.tensor.offset);
      return Number(raw);
    }
    return getNumericElement(data, loss.tensor.offset);
  }
  return loss;
}

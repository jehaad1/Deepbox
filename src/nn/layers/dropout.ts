import { DTypeError, InvalidParameterError } from "../../core";
import { type AnyTensor, dropoutGrad, GradTensor } from "../../ndarray";
import { Module } from "../module/Module";

/**
 * Applies Dropout regularization during training.
 *
 * **Mathematical Formulation:**
 * During training:
 * ```
 * y = x * mask / (1 - p)
 * ```
 * where mask is a binary tensor with probability (1-p) of being 1.
 *
 * During evaluation:
 * ```
 * y = x
 * ```
 *
 * **Purpose:**
 * - Prevents overfitting by randomly zeroing elements during training
 * - Forces network to learn redundant representations
 * - Improves generalization performance
 *
 * **Scaling:**
 * The output is scaled by 1/(1-p) during training to maintain expected value.
 * This is called "inverted dropout" and eliminates the need for scaling during inference.
 *
 * @example
 * ```ts
 * import { Dropout } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const dropout = new Dropout(0.5); // Drop 50% of neurons
 * const input = tensor([[1, 2, 3, 4]]);
 *
 * // Training mode: randomly zeros ~50% of elements
 * dropout.train();
 * const output = dropout.forward(input);
 *
 * // Evaluation mode: passes input unchanged
 * dropout.eval();
 * const output2 = dropout.forward(input); // Same as input
 * ```
 *
 * References:
 * - Dropout paper: https://jmlr.org/papers/v15/srivastava14a.html
 * - PyTorch Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
 *
 * @category Neural Network Layers
 */
export class Dropout extends Module {
  /** Probability of an element being zeroed (dropout rate) */
  private readonly p: number;

  /**
   * Create a new Dropout layer.
   *
   * @param p - Probability of an element being zeroed (0 <= p < 1)
   * @throws {InvalidParameterError} If p is not in valid range [0, 1)
   */
  constructor(p = 0.5) {
    super();

    // Validate dropout probability is in valid range
    if (!Number.isFinite(p) || p < 0 || p >= 1) {
      throw new InvalidParameterError(`Dropout probability must be in [0, 1), got ${p}`, "p", p);
    }

    this.p = p;
  }

  /**
   * Forward pass: apply dropout during training, identity during evaluation.
   *
   * @param input - Input tensor of any shape (Tensor or GradTensor)
   * @returns Output tensor with same shape as input
   */
  forward(input: AnyTensor): GradTensor {
    // Convert to GradTensor if needed
    const inputTensor = input instanceof GradTensor ? input : GradTensor.fromTensor(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Dropout does not support string dtype");
    }

    // Use vectorized dropout implementation from autograd
    // This handles training/eval mode and mask generation
    return dropoutGrad(inputTensor, this.p, this.training);
  }

  /**
   * Get string representation of the layer.
   *
   * @returns String representation with dropout probability
   */
  override toString(): string {
    return `Dropout(p=${this.p})`;
  }

  /**
   * Get the dropout probability.
   */
  get dropoutRate(): number {
    return this.p;
  }
}

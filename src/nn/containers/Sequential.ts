import { DeepboxError, IndexError, InvalidParameterError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import { Module } from "../module/Module";

/**
 * Sequential container for stacking layers in a linear pipeline.
 *
 * **Purpose:**
 * - Simplifies model construction by chaining layers sequentially
 * - Automatically manages forward pass through all layers
 * - Provides clean API for building feedforward networks
 *
 * **Behavior:**
 * The output of each layer becomes the input to the next layer.
 * Layers are executed in the order they were added.
 *
 * @example
 * ```ts
 * import { Sequential, Linear, ReLU, Dropout } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Create a simple feedforward network
 * const model = new Sequential(
 *   new Linear(784, 256),
 *   new ReLU(),
 *   new Dropout(0.5),
 *   new Linear(256, 10)
 * );
 *
 * const input = tensor(new Array(784).fill(0));
 * const output = model.forward(input);
 * ```
 *
 * @example
 * ```ts
 * // Access individual layers
 * const model = new Sequential(
 *   new Linear(10, 5),
 *   new ReLU()
 * );
 *
 * const firstLayer = model.getLayer(0); // Linear layer
 * const layerCount = model.length; // 2
 * ```
 *
 * References:
 * - PyTorch Sequential: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
 * - Keras Sequential: https://keras.io/guides/sequential_model/
 *
 * @category Neural Network Containers
 */
export class Sequential extends Module {
  /** Array of layers in sequential order */
  private readonly layers: Module[];

  /**
   * Create a new Sequential container.
   *
   * @param layers - Variable number of Module instances to stack sequentially
   * @throws {InvalidParameterError} If no layers are provided
   * @throws {DeepboxError} If a layer is undefined
   */
  constructor(...layers: Module[]) {
    super();

    // Validate that at least one layer is provided
    if (layers.length === 0) {
      throw new InvalidParameterError(
        "Sequential requires at least one layer",
        "layers",
        layers.length
      );
    }

    // Store layers in execution order
    this.layers = layers;

    // Register each layer as a child module with numeric index as name
    // This enables parameter tracking and hierarchical naming
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (!layer) {
        throw new DeepboxError(`Layer at index ${i} is undefined`);
      }
      this.registerModule(String(i), layer);
    }
  }

  /**
   * Forward pass: sequentially apply all layers.
   *
   * The output of each layer becomes the input to the next layer.
   *
   * @param input - Input tensor (Tensor or GradTensor)
   * @returns Output tensor after passing through all layers
   * @throws {InvalidParameterError} If the input count is invalid or a layer returns multiple outputs
   * @throws {DeepboxError} If a layer is undefined
   */
  forward(...inputs: AnyTensor[]): AnyTensor {
    if (inputs.length !== 1) {
      throw new InvalidParameterError(
        "Sequential.forward expects a single input tensor",
        "inputs",
        inputs.length
      );
    }
    const input = inputs[0];
    if (!input) {
      throw new InvalidParameterError(
        "Sequential.forward expects a single input tensor",
        "input",
        input
      );
    }
    // Start with the input tensor
    let output = input;

    // Sequentially apply each layer's forward pass
    // Each layer transforms the output from the previous layer
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      if (!layer) {
        throw new DeepboxError(`Layer at index ${i} is undefined`);
      }

      // Apply current layer's transformation
      // Type assertion needed because forward can return Tensor | Tensor[]
      const result = layer.call(output);
      if (Array.isArray(result)) {
        throw new InvalidParameterError(
          `Sequential does not support layers that return multiple tensors (layer ${i})`,
          "layer",
          i
        );
      }
      output = result;
    }

    // Return final output after all transformations
    return output;
  }

  /**
   * Get a layer by index.
   *
   * @param index - Zero-based index of the layer
   * @returns The layer at the specified index
   * @throws {IndexError} If index is out of bounds
   * @throws {DeepboxError} If a layer is undefined
   */
  getLayer(index: number): Module {
    // Validate index is within bounds
    if (index < 0 || index >= this.layers.length) {
      throw new IndexError(`Layer index ${index} out of bounds [0, ${this.layers.length})`, {
        index,
        validRange: [0, this.layers.length - 1],
      });
    }

    const layer = this.layers[index];
    if (!layer) {
      throw new DeepboxError(`Layer at index ${index} is undefined`);
    }

    return layer;
  }

  /**
   * Get the number of layers in the sequential container.
   */
  get length(): number {
    return this.layers.length;
  }

  /**
   * Get string representation showing all layers.
   *
   * @returns Multi-line string with each layer on a separate line
   */
  override toString(): string {
    // Build hierarchical representation
    const lines = ["Sequential("];

    // Add each layer with its index
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      if (!layer) continue;

      // Get layer's string representation and indent continuation lines
      const childLines = layer.toString().split("\n");
      const layerStr = childLines.map((line, idx) => (idx === 0 ? line : `  ${line}`)).join("\n");

      // Format as: (index): LayerType(...)
      lines.push(`  (${i}): ${layerStr}`);
    }

    lines.push(")");
    return lines.join("\n");
  }

  /**
   * Iterate over all layers.
   *
   * @returns Iterator of layers
   */
  *[Symbol.iterator](): Iterator<Module> {
    for (const layer of this.layers) {
      yield layer;
    }
  }
}

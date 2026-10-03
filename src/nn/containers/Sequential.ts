import { IndexError, InvalidParameterError } from "../../core";
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
 * A plain `Tensor` input needs no wrapping for training: the first layer with trainable
 * weights returns a `GradTensor` that tracks them, and the layers after it pass it on. Inside
 * `noGrad()`, or when every weight is frozen, the result is a plain `Tensor`.
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
 * - Keras Sequential: https://keras.io/guides/sequential_model/
 *
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox Module & Sequential}
 * @category Neural Network Containers
 */
export class Sequential extends Module {
  /** Array of layers in sequential order */
  private readonly layers: Module[];

  /**
   * Create a new Sequential container.
   *
   * @param layers - Variable number of Module instances to stack sequentially
   * @throws {InvalidParameterError} If no layers are provided or a layer is not a Module
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

    // Validate every layer before registering any of them
    for (let i = 0; i < layers.length; i++) {
      Sequential.assertLayer(layers[i], i);
    }

    // Store layers in execution order
    this.layers = layers;
    this.reregisterAll();
  }

  private static assertLayer(layer: unknown, index: number): asserts layer is Module {
    if (
      typeof layer !== "object" ||
      layer === null ||
      typeof (layer as Module).forward !== "function"
    ) {
      throw new InvalidParameterError(
        `Layer at index ${index} is not a Module (received ${layer === null ? "null" : typeof layer})`,
        "layers",
        layer
      );
    }
  }

  /** Register each layer as a child module under its numeric index (parameter names like "0.weight"). */
  private reregisterAll(): void {
    this._modules.clear();
    for (let i = 0; i < this.layers.length; i++) {
      this.registerModule(String(i), this.layers[i] as Module);
    }
  }

  /**
   * Forward pass: sequentially apply all layers.
   *
   * The output of each layer becomes the input to the next layer, and each
   * layer is invoked through `call()` so its forward hooks run.
   *
   * @param inputs - Exactly one input tensor (Tensor or GradTensor)
   * @returns Output tensor after passing through all layers
   * @throws {InvalidParameterError} If the input count is not one or a layer returns multiple outputs
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
    let output = input;

    // Each layer transforms the output from the previous layer
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i] as Module;
      const result: AnyTensor | AnyTensor[] = layer.call(output);
      if (Array.isArray(result)) {
        throw new InvalidParameterError(
          `Sequential does not support layers that return multiple tensors (layer ${i})`,
          "layer",
          i
        );
      }
      output = result;
    }

    return output;
  }

  /**
   * Get a layer by index.
   *
   * @param index - Zero-based index of the layer
   * @returns The layer at the specified index
   * @throws {IndexError} If index is not an integer in `[0, length)`
   */
  getLayer(index: number): Module {
    const layer = Number.isInteger(index) ? this.layers[index] : undefined;
    if (layer === undefined) {
      throw new IndexError(`Layer index ${index} out of bounds [0, ${this.layers.length})`, {
        index,
        validRange: [0, this.layers.length - 1],
      });
    }
    return layer;
  }

  /**
   * Append a layer to the end of the pipeline.
   *
   * @throws {InvalidParameterError} If `layer` is not a Module
   */
  append(layer: Module): this {
    Sequential.assertLayer(layer, this.layers.length);
    this.layers.push(layer);
    this.registerModule(String(this.layers.length - 1), layer);
    return this;
  }

  /**
   * Append several layers, in order.
   *
   * @throws {InvalidParameterError} If an element is not a Module
   */
  extend(layers: Iterable<Module>): this {
    for (const layer of layers) {
      this.append(layer);
    }
    return this;
  }

  /**
   * Insert a layer at `index`; later layers shift one position up and their
   * parameter names are renumbered.
   *
   * @throws {IndexError} If `index` is not an integer in `[0, length]`
   * @throws {InvalidParameterError} If `layer` is not a Module
   */
  insert(index: number, layer: Module): this {
    if (!Number.isInteger(index) || index < 0 || index > this.layers.length) {
      throw new IndexError(`Insert index ${index} out of bounds [0, ${this.layers.length}]`, {
        index,
        validRange: [0, this.layers.length],
      });
    }
    Sequential.assertLayer(layer, index);
    this.layers.splice(index, 0, layer);
    this.reregisterAll();
    return this;
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

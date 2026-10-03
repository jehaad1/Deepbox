/**
 * Embedding layers for mapping discrete indices to dense vectors.
 *
 * - {@link Embedding}: Standard lookup table embedding
 * - {@link EmbeddingBag}: Embedding with built-in reduction (sum, mean, max)
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { DeviceError, type DType, DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, GradTensor, parameter, randn } from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { allPlain, type LayerDType, resolveLayerDtype, settle } from "./_shared";

/** Materialize a numeric tensor into a contiguous logical-order Float64Array. */
function denseFloat64Embed(t: Tensor): Float64Array {
  if (t.isDeviceTensor) {
    throw new DeviceError(
      `This layer runs on the host and cannot read a tensor stored on device "${t.device}"; ` +
        "move it back with `await tensor.cpu()` first"
    );
  }
  const out = new Float64Array(t.size);
  const data = requireNumericData(t.data, "Embedding");
  const contig = isContiguous(t.shape, t.strides);
  const logical = computeStrides(t.shape);
  for (let i = 0; i < t.size; i++) {
    const off = contig ? t.offset + i : offsetFromFlatIndex(i, logical, t.strides, t.offset);
    out[i] = readAsNumber(data, off);
  }
  return out;
}

/**
 * Read the logical elements of an index tensor and check that each one is an
 * integer in `[0, limit)`.
 *
 * @throws {InvalidParameterError} If an index is NaN, infinite, not an integer, or out of range
 */
function readIndices(t: Tensor, layer: string, limit: number): Int32Array {
  const values = denseFloat64Embed(t);
  const out = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isInteger(v)) {
      throw new InvalidParameterError(
        `${layer} indices must be integers, got ${v} at position ${i}`,
        "indices",
        v
      );
    }
    if (v < 0 || v >= limit) {
      throw new InvalidParameterError(
        `${layer} index ${v} out of range [0, ${limit})`,
        "indices",
        v
      );
    }
    out[i] = v;
  }
  return out;
}

/**
 * Resolve `paddingIdx` (negative values count from the end, as in PyTorch) and
 * validate it against the table size.
 */
function resolvePaddingIdx(
  paddingIdx: number | undefined,
  numEmbeddings: number
): number | undefined {
  if (paddingIdx === undefined) return undefined;
  const resolved =
    Number.isInteger(paddingIdx) && paddingIdx < 0 ? paddingIdx + numEmbeddings : paddingIdx;
  if (!Number.isInteger(resolved) || resolved < 0 || resolved >= numEmbeddings) {
    throw new InvalidParameterError(
      `paddingIdx must be in [-${numEmbeddings}, ${numEmbeddings})`,
      "paddingIdx",
      paddingIdx
    );
  }
  return resolved;
}

function validateTableSize(numEmbeddings: number, embeddingDim: number): void {
  if (!Number.isInteger(numEmbeddings) || numEmbeddings <= 0) {
    throw new InvalidParameterError(
      "numEmbeddings must be a positive integer",
      "numEmbeddings",
      numEmbeddings
    );
  }
  if (!Number.isInteger(embeddingDim) || embeddingDim <= 0) {
    throw new InvalidParameterError(
      "embeddingDim must be a positive integer",
      "embeddingDim",
      embeddingDim
    );
  }
}

/** Set row `row` of a freshly initialized weight parameter to zero. */
function zeroRow(weight: GradTensor, row: number, embeddingDim: number): void {
  const data = weight.tensor.data;
  if (data instanceof BigInt64Array || Array.isArray(data)) return;
  const rowStart = row * embeddingDim;
  for (let j = 0; j < embeddingDim; j++) {
    const idx = rowStart + j;
    if (idx < data.length) {
      data[idx] = 0;
    }
  }
}

/** Cast a float64 result to the dtype of the weight table. */
function weightTyped(data: Float64Array, shape: readonly number[], like: Tensor): Tensor {
  const out = Tensor.fromTypedArray({
    data,
    shape: [...shape],
    dtype: "float64",
    device: like.device,
  });
  const dtype: DType = like.dtype;
  return dtype === "float64" || dtype === "string" ? out : out.astype(dtype);
}

/**
 * A lookup table that stores embeddings of a fixed dictionary and size.
 *
 * This module is often used to store word embeddings and retrieve them using indices.
 * The input to the module is a list of indices, and the output is the corresponding
 * word embeddings.
 *
 * **Mathematical Formulation**:
 * ```
 * output[i] = weight[indices[i]]
 * ```
 *
 * The weight table is initialized from a standard normal distribution. With
 * `paddingIdx`, that row is initialized to zeros and receives no gradient. As in PyTorch,
 * the lookup returns the stored row, which stays zero unless you write to it.
 *
 * The output has the dtype of the weight table. A `GradTensor` index gives a `GradTensor`;
 * a plain index tensor gives a `GradTensor` that tracks the weight while it requires grad
 * and gradient tracking is on, and a plain `Tensor` otherwise (for example inside
 * `noGrad()` or for a frozen table).
 *
 * @example
 * ```ts
 * import { Embedding } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const embedding = new Embedding(10, 3); // 10 words, 3-dim embeddings
 * const indices = tensor([1, 2, 4, 5]);
 * const output = embedding.forward(indices); // shape: [4, 3]
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */
export class Embedding extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  /** Number of embeddings (vocabulary size) */
  readonly numEmbeddings: number;

  /** Dimension of each embedding vector */
  readonly embeddingDim: number;

  /** Padding index with negative values already resolved; this row receives no gradient */
  readonly paddingIdx: number | undefined;

  /** Weight matrix of shape (numEmbeddings, embeddingDim) */
  private weight_: GradTensor;

  /**
   * Create a new Embedding layer.
   *
   * @param numEmbeddings - Size of the embedding dictionary (vocabulary size)
   * @param embeddingDim - Size of each embedding vector
   * @param options - Configuration options
   * @param options.paddingIdx - If given, the row of this index starts at zero and receives no
   *   gradient. Negative values count from the end, so `-1` is the last row.
   * @param options.dtype - Dtype of the weight table (default: the global default dtype)
   * @throws {InvalidParameterError} If a size is not a positive integer or `paddingIdx` is out of range
   */
  constructor(
    numEmbeddings: number,
    embeddingDim: number,
    options: {
      readonly paddingIdx?: number;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();

    validateTableSize(numEmbeddings, embeddingDim);

    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.paddingIdx = resolvePaddingIdx(options.paddingIdx, numEmbeddings);

    // Initialize weight with standard normal
    const w = randn([numEmbeddings, embeddingDim], { dtype: resolveLayerDtype(options.dtype) });
    this.weight_ = parameter(w);
    this.registerParameter("weight", this.weight_);

    // Zero out padding index row if specified
    if (this.paddingIdx !== undefined) {
      zeroRow(this.weight_, this.paddingIdx, embeddingDim);
    }
  }

  /**
   * Create an Embedding layer from an existing weight matrix.
   *
   * @param weights - Matrix of shape `(numEmbeddings, embeddingDim)`; its values are copied
   * @param options.freeze - If true (the default), the weights are not trained
   * @param options.paddingIdx - Index whose row receives no gradient. The stored row is left
   *   as given and the lookup returns it.
   * @param options.dtype - Dtype of the new table (default: the global default dtype)
   * @returns A new Embedding layer holding a copy of `weights`
   * @throws {ShapeError} If `weights` is not 2-D
   * @throws {DTypeError} If `weights` has string dtype
   */
  static fromPretrained(
    weights: AnyTensor,
    options: {
      readonly freeze?: boolean;
      readonly paddingIdx?: number;
      readonly dtype?: LayerDType;
    } = {}
  ): Embedding {
    const source = GradTensor.isGradTensor(weights) ? weights.tensor : weights;
    if (source.dtype === "string") {
      throw new DTypeError("Embedding.fromPretrained weights must be numeric, not string");
    }
    if (source.ndim !== 2) {
      throw new ShapeError(`Embedding.fromPretrained expects 2-D weights; got ${source.ndim}-D`);
    }
    const [rows, dim] = source.shape as [number, number];
    const layer = new Embedding(rows, dim, {
      ...(options.paddingIdx === undefined ? {} : { paddingIdx: options.paddingIdx }),
      ...(options.dtype === undefined ? {} : { dtype: options.dtype }),
    });
    const target = layer.weight_.tensor.data;
    if (target instanceof BigInt64Array || Array.isArray(target)) {
      throw new DTypeError("Embedding weight storage must be floating point");
    }
    const values = denseFloat64Embed(source);
    for (let i = 0; i < values.length; i++) target[i] = values[i] as number;
    if (options.freeze ?? true) layer.freezeParameters(["weight"]);
    return layer;
  }

  /**
   * Forward pass: look up embeddings for the given indices.
   *
   * @param indices - Integer tensor of any shape containing indices into the embedding table
   * @returns Tensor of shape (*indices.shape, embeddingDim), in the dtype of the weight table
   * @throws {DTypeError} If `indices` has string dtype
   * @throws {InvalidParameterError} If an index is not an integer or is outside `[0, numEmbeddings)`
   */
  forward(indices: GradTensor): GradTensor;
  forward(indices: Tensor): AnyTensor;
  forward(indices: AnyTensor): AnyTensor;
  forward(indices: AnyTensor): AnyTensor {
    return settle(this.run(indices), allPlain(indices));
  }

  private run(indices: AnyTensor): GradTensor {
    const idx = GradTensor.isGradTensor(indices) ? indices.tensor : indices;

    if (idx.dtype === "string") {
      throw new DTypeError("Embedding indices must be numeric, not string");
    }

    const embDim = this.embeddingDim;
    const numEmbeddings = this.numEmbeddings;
    const paddingIdx = this.paddingIdx;
    const numIdx = idx.size;
    // Resolved once so the backward can scatter gradients to the matching rows
    // without re-reading the (possibly strided) index tensor.
    const resolved = readIndices(idx, "Embedding", numEmbeddings);

    const weightTensor = this.weight_.tensor;
    const weightData = requireNumericData(weightTensor.data, "Embedding.forward");
    // Rows are copied straight out of contiguous storage; other layouts are densified first.
    const direct =
      !(weightData instanceof BigInt64Array) &&
      isContiguous(weightTensor.shape, weightTensor.strides);
    const table: ArrayLike<number> = direct ? weightData : denseFloat64Embed(weightTensor);
    const tableBase = direct ? weightTensor.offset : 0;

    // Output shape: (*indices.shape, embeddingDim). The padding row is returned like any
    // other row (PyTorch); only its gradient is blocked in the backward pass.
    const outShape = [...idx.shape, embDim];
    const outData = new Float64Array(numIdx * embDim);
    for (let i = 0; i < numIdx; i++) {
      const index = resolved[i] as number;
      const rowStart = tableBase + index * embDim;
      const outStart = i * embDim;
      for (let j = 0; j < embDim; j++) outData[outStart + j] = table[rowStart + j] as number;
    }

    const outTensor = weightTyped(outData, outShape, weightTensor);

    return customOp(outTensor, [
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64Embed(g);
          const gw = new Float64Array(numEmbeddings * embDim);
          for (let i = 0; i < numIdx; i++) {
            const index = resolved[i] as number;
            if (index === paddingIdx) continue;
            const rowStart = index * embDim;
            const outStart = i * embDim;
            for (let j = 0; j < embDim; j++) gw[rowStart + j]! += go[outStart + j] ?? 0;
          }
          return weightTyped(gw, [numEmbeddings, embDim], weightTensor);
        },
      ],
    ]);
  }

  /** Access the weight parameter */
  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    const pad = this.paddingIdx !== undefined ? `, padding_idx=${this.paddingIdx}` : "";
    return `Embedding(${this.numEmbeddings}, ${this.embeddingDim}${pad})`;
  }
}

/** Reduction applied to every bag of an {@link EmbeddingBag}. */
export type EmbeddingBagMode = "sum" | "mean" | "max";

/**
 * Computes sums, means, or maxes of "bags" of embeddings, without
 * instantiating the intermediate per-embedding matrix.
 *
 * Indices can be given as a 1-D tensor together with `offsets` that mark where each
 * bag starts, or as a 2-D tensor `(numBags, bagSize)` without offsets. Entries equal to
 * `paddingIdx` are left out of the reduction (and receive no gradient). An empty bag,
 * or a bag made only of padding, gives a row of zeros.
 *
 * @example
 * ```ts
 * import { EmbeddingBag } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const bag = new EmbeddingBag(10, 3, { mode: 'mean' });
 * const indices = tensor([1, 2, 4, 5, 4, 3, 2, 9]);
 * const offsets = tensor([0, 4]); // two bags: [1,2,4,5] and [4,3,2,9]
 * const output = bag.forward(indices, offsets); // shape: [2, 3]
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */
export class EmbeddingBag extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  /** Number of embeddings (vocabulary size) */
  readonly numEmbeddings: number;
  /** Dimension of each embedding vector */
  readonly embeddingDim: number;
  /** Reduction applied to each bag */
  readonly mode: EmbeddingBagMode;
  /** Padding index with negative values already resolved; it is skipped in every bag */
  readonly paddingIdx: number | undefined;
  /** Whether `offsets` carries a final entry that marks the end of the last bag */
  readonly includeLastOffset: boolean;
  private weight_: GradTensor;

  /**
   * @param numEmbeddings - Size of the embedding dictionary
   * @param embeddingDim - Size of each embedding vector
   * @param options.mode - Reduction over each bag: `"sum"`, `"mean"` (default) or `"max"`
   * @param options.paddingIdx - Index that is skipped in every bag; its weight row starts at
   *   zero. Negative values count from the end.
   * @param options.includeLastOffset - If true, `offsets` has `numBags + 1` entries and the
   *   last one is the end of the final bag (default false)
   * @param options.dtype - Dtype of the weight table (default: the global default dtype)
   * @throws {InvalidParameterError} If a size is not a positive integer, `mode` is unknown, or
   *   `paddingIdx` is out of range
   */
  constructor(
    numEmbeddings: number,
    embeddingDim: number,
    options: {
      readonly mode?: EmbeddingBagMode;
      readonly paddingIdx?: number;
      readonly includeLastOffset?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();

    validateTableSize(numEmbeddings, embeddingDim);

    const mode = options.mode ?? "mean";
    if (mode !== "sum" && mode !== "mean" && mode !== "max") {
      throw new InvalidParameterError(
        `mode must be "sum", "mean" or "max", got ${String(mode)}`,
        "mode",
        mode
      );
    }

    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.mode = mode;
    this.paddingIdx = resolvePaddingIdx(options.paddingIdx, numEmbeddings);
    this.includeLastOffset = options.includeLastOffset ?? false;

    const w = randn([numEmbeddings, embeddingDim], { dtype: resolveLayerDtype(options.dtype) });
    this.weight_ = parameter(w);
    this.registerParameter("weight", this.weight_);

    if (this.paddingIdx !== undefined) {
      zeroRow(this.weight_, this.paddingIdx, embeddingDim);
    }
  }

  /**
   * Forward pass: compute bag embeddings.
   *
   * @param indices - 1-D tensor of indices (with `offsets`), or 2-D `(numBags, bagSize)` (without)
   * @param offsets - 1-D tensor with the starting position of each bag in `indices`. It must
   *   start at 0 and be non-decreasing, and no entry may exceed `indices.size`.
   * @returns Tensor of shape (numBags, embeddingDim), in the dtype of the weight table
   * @throws {InvalidParameterError} If `offsets` is missing for 1-D indices, an offset is
   *   invalid, or an index is not an integer or out of range
   * @throws {ShapeError} If the ranks of `indices` and `offsets` are not supported
   * @throws {DTypeError} If `indices` or `offsets` has string dtype
   */
  forward(indices: GradTensor, ...rest: AnyTensor[]): GradTensor;
  forward(indices: Tensor, ...rest: AnyTensor[]): AnyTensor;
  forward(indices: AnyTensor, ...rest: AnyTensor[]): AnyTensor;
  forward(indices: AnyTensor, ...rest: AnyTensor[]): AnyTensor {
    return settle(this.run(indices, ...rest), allPlain(indices, ...rest));
  }

  private run(indices: AnyTensor, ...rest: AnyTensor[]): GradTensor {
    const offsetsInput = rest[0];

    const idx = GradTensor.isGradTensor(indices) ? indices.tensor : indices;
    const off =
      offsetsInput === undefined
        ? undefined
        : GradTensor.isGradTensor(offsetsInput)
          ? offsetsInput.tensor
          : offsetsInput;

    if (idx.dtype === "string" || off?.dtype === "string") {
      throw new DTypeError("EmbeddingBag indices and offsets must be numeric");
    }

    const embDim = this.embeddingDim;
    const numEmbeddings = this.numEmbeddings;
    const paddingIdx = this.paddingIdx;
    const mode = this.mode;

    // Bag boundaries in the flattened index list: bag b is [starts[b], starts[b + 1]).
    let starts: Int32Array;
    if (idx.ndim === 2) {
      if (off !== undefined) {
        throw new ShapeError("EmbeddingBag offsets must not be given when indices are 2-D");
      }
      const bags = idx.shape[0] ?? 0;
      const bagSize = idx.shape[1] ?? 0;
      starts = new Int32Array(bags + 1);
      for (let b = 0; b <= bags; b++) starts[b] = b * bagSize;
    } else if (idx.ndim === 1) {
      if (off === undefined) {
        throw new InvalidParameterError(
          "EmbeddingBag requires offsets tensor as second argument",
          "offsets",
          undefined
        );
      }
      if (off.ndim !== 1) {
        throw new ShapeError(`EmbeddingBag offsets must be 1-D; got ${off.ndim}-D`);
      }
      const given = denseFloat64Embed(off);
      const total = idx.size;
      const numBags = this.includeLastOffset ? Math.max(0, given.length - 1) : given.length;
      starts = new Int32Array(numBags + 1);
      for (let b = 0; b < given.length; b++) {
        const v = given[b] as number;
        if (!Number.isInteger(v) || v < 0 || v > total) {
          throw new InvalidParameterError(
            `EmbeddingBag offsets must be integers in [0, ${total}], got ${v} at position ${b}`,
            "offsets",
            v
          );
        }
        if (b === 0 && v !== 0) {
          throw new InvalidParameterError(
            `EmbeddingBag offsets[0] must be 0, got ${v}`,
            "offsets",
            v
          );
        }
        if (b > 0 && v < (given[b - 1] as number)) {
          throw new InvalidParameterError(
            `EmbeddingBag offsets must be non-decreasing, got ${v} after ${given[b - 1]} at position ${b}`,
            "offsets",
            v
          );
        }
        starts[b] = v;
      }
      if (!this.includeLastOffset && given.length > 0) starts[given.length] = total;
    } else {
      throw new ShapeError(`EmbeddingBag indices must be 1-D or 2-D; got ${idx.ndim}-D`);
    }

    const numBags = starts.length - 1;
    const resolved = readIndices(idx, "EmbeddingBag", numEmbeddings);

    const weightTensor = this.weight_.tensor;
    const weightData = requireNumericData(weightTensor.data, "EmbeddingBag");
    const direct =
      !(weightData instanceof BigInt64Array) &&
      isContiguous(weightTensor.shape, weightTensor.strides);
    const table: ArrayLike<number> = direct ? weightData : denseFloat64Embed(weightTensor);
    const tableBase = direct ? weightTensor.offset : 0;

    const outData = new Float64Array(numBags * embDim);
    // Backward bookkeeping: number of rows reduced in each bag (padding excluded) and, for
    // max mode, the table row that supplied each (bag, feature) value.
    const bagCounts = new Int32Array(numBags);
    const argmaxRow = mode === "max" ? new Int32Array(numBags * embDim).fill(-1) : null;

    for (let b = 0; b < numBags; b++) {
      const outStart = b * embDim;
      let count = 0;
      for (let i = starts[b] as number; i < (starts[b + 1] as number); i++) {
        const index = resolved[i] as number;
        if (index === paddingIdx) continue;
        const rowStart = tableBase + index * embDim;
        count++;

        for (let j = 0; j < embDim; j++) {
          const val = table[rowStart + j] as number;
          const outIdx = outStart + j;
          if (argmaxRow) {
            // Max mode: the first row seeds the value; a NaN always wins, like torch.
            if (
              (argmaxRow[outIdx] as number) < 0 ||
              val > (outData[outIdx] as number) ||
              Number.isNaN(val)
            ) {
              outData[outIdx] = val;
              argmaxRow[outIdx] = index;
            }
          } else {
            outData[outIdx] = (outData[outIdx] as number) + val;
          }
        }
      }
      bagCounts[b] = count;

      // For mean mode, divide by count
      if (mode === "mean" && count > 0) {
        for (let j = 0; j < embDim; j++) {
          outData[outStart + j] = (outData[outStart + j] as number) / count;
        }
      }
    }

    const outTensor = weightTyped(outData, [numBags, embDim], weightTensor);

    return customOp(outTensor, [
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64Embed(g);
          const gw = new Float64Array(numEmbeddings * embDim);
          for (let b = 0; b < numBags; b++) {
            const outStart = b * embDim;
            if (argmaxRow) {
              for (let j = 0; j < embDim; j++) {
                const row = argmaxRow[outStart + j] ?? -1;
                if (row >= 0) gw[row * embDim + j]! += go[outStart + j] ?? 0;
              }
            } else {
              const count = bagCounts[b] ?? 0;
              const scale = mode === "mean" && count > 0 ? 1 / count : 1;
              for (let i = starts[b] as number; i < (starts[b + 1] as number); i++) {
                const row = resolved[i] as number;
                if (row === paddingIdx) continue;
                const rowStart = row * embDim;
                for (let j = 0; j < embDim; j++) {
                  gw[rowStart + j]! += (go[outStart + j] ?? 0) * scale;
                }
              }
            }
          }
          return weightTyped(gw, [numEmbeddings, embDim], weightTensor);
        },
      ],
    ]);
  }

  /** Access the weight parameter */
  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    const pad = this.paddingIdx !== undefined ? `, padding_idx=${this.paddingIdx}` : "";
    return `EmbeddingBag(${this.numEmbeddings}, ${this.embeddingDim}, mode=${this.mode}${pad})`;
  }
}

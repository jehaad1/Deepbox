/**
 * Embedding layers for mapping discrete indices to dense vectors.
 *
 * - {@link Embedding} — Standard lookup table embedding
 * - {@link EmbeddingBag} — Embedding with built-in reduction (sum, mean, max)
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, GradTensor, parameter, randn } from "../../ndarray";
import { flatOffset, readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";

/** Materialize a numeric tensor into a contiguous logical-order Float64Array. */
function denseFloat64Embed(t: Tensor): Float64Array {
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
 */
export class Embedding extends Module {
  /** Number of embeddings (vocabulary size) */
  private readonly numEmbeddings: number;

  /** Dimension of each embedding vector */
  private readonly embeddingDim: number;

  /** Optional padding index (embedding at this index is always zero) */
  private readonly paddingIdx: number | undefined;

  /** Weight matrix of shape (numEmbeddings, embeddingDim) */
  private weight_: GradTensor;

  /**
   * Create a new Embedding layer.
   *
   * @param numEmbeddings - Size of the embedding dictionary (vocabulary size)
   * @param embeddingDim - Size of each embedding vector
   * @param options - Configuration options
   * @param options.paddingIdx - If given, pads the output with zeros whenever it encounters the index
   */
  constructor(
    numEmbeddings: number,
    embeddingDim: number,
    options: {
      readonly paddingIdx?: number;
    } = {}
  ) {
    super();

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

    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.paddingIdx = options.paddingIdx;

    if (this.paddingIdx !== undefined) {
      if (
        !Number.isInteger(this.paddingIdx) ||
        this.paddingIdx < 0 ||
        this.paddingIdx >= numEmbeddings
      ) {
        throw new InvalidParameterError(
          `paddingIdx must be in [0, ${numEmbeddings})`,
          "paddingIdx",
          this.paddingIdx
        );
      }
    }

    // Initialize weight with standard normal
    const w = randn([numEmbeddings, embeddingDim]);
    this.weight_ = parameter(w);
    this.registerParameter("weight", this.weight_);

    // Zero out padding index row if specified
    if (this.paddingIdx !== undefined) {
      this.zeroPaddingRow();
    }
  }

  private zeroPaddingRow(): void {
    if (this.paddingIdx === undefined) return;
    const t = this.weight_.tensor;
    const data = t.data;
    if (data instanceof BigInt64Array || Array.isArray(data)) return;

    const rowStart = this.paddingIdx * this.embeddingDim;
    for (let j = 0; j < this.embeddingDim; j++) {
      const idx = rowStart + j;
      if (idx < data.length) {
        data[idx] = 0;
      }
    }
  }

  /**
   * Forward pass: look up embeddings for the given indices.
   *
   * @param indices - Integer tensor of any shape containing indices into the embedding table
   * @returns Tensor of shape (*indices.shape, embeddingDim)
   */
  forward(indices: AnyTensor): GradTensor {
    const idx = GradTensor.isGradTensor(indices) ? indices.tensor : indices;

    if (idx.dtype === "string") {
      throw new DTypeError("Embedding indices must be numeric, not string");
    }

    const idxData = requireNumericData(idx.data, "Embedding.forward");
    const idxContig = isContiguous(idx.shape, idx.strides);
    const idxLogical = computeStrides(idx.shape);
    const numIdx = idx.size;

    const weightTensor = this.weight_.tensor;
    const weightData = requireNumericData(weightTensor.data, "Embedding.forward");

    // Output shape: (*indices.shape, embeddingDim)
    const outShape = [...idx.shape, this.embeddingDim];
    const outData = new Float64Array(numIdx * this.embeddingDim);
    // Resolve indices once so the backward can scatter gradients to the
    // matching rows without re-reading the (possibly strided) index tensor.
    const resolved = new Int32Array(numIdx);

    for (let i = 0; i < numIdx; i++) {
      const off = flatOffset(i, idx.offset, idxContig, idxLogical, idx.strides);
      const index = Math.round(readAsNumber(idxData, off));

      if (index < 0 || index >= this.numEmbeddings) {
        throw new InvalidParameterError(
          `Embedding index ${index} out of range [0, ${this.numEmbeddings})`,
          "indices",
          index
        );
      }
      resolved[i] = index;

      // If paddingIdx, output zeros for that index
      if (this.paddingIdx !== undefined && index === this.paddingIdx) {
        // Already zero from Float64Array initialization
        continue;
      }

      // Copy the embedding row
      const rowStart = index * this.embeddingDim;
      const outStart = i * this.embeddingDim;
      for (let j = 0; j < this.embeddingDim; j++) {
        outData[outStart + j] = readAsNumber(weightData, rowStart + j);
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outData,
      shape: outShape,
      dtype: "float64",
      device: weightTensor.device,
    });

    const embDim = this.embeddingDim;
    const numEmbeddings = this.numEmbeddings;
    const paddingIdx = this.paddingIdx;
    return customOp(outTensor, [
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64Embed(g);
          const gw = new Float64Array(numEmbeddings * embDim);
          for (let i = 0; i < numIdx; i++) {
            const index = resolved[i] ?? 0;
            if (paddingIdx !== undefined && index === paddingIdx) continue;
            const rowStart = index * embDim;
            const outStart = i * embDim;
            for (let j = 0; j < embDim; j++) gw[rowStart + j]! += go[outStart + j] ?? 0;
          }
          return Tensor.fromTypedArray({
            data: gw,
            shape: [numEmbeddings, embDim],
            dtype: "float64",
            device: weightTensor.device,
          });
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

type EmbeddingBagMode = "sum" | "mean" | "max";

/**
 * Computes sums, means, or maxes of "bags" of embeddings, without
 * instantiating the intermediate per-embedding matrix.
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
 */
export class EmbeddingBag extends Module {
  private readonly numEmbeddings: number;
  private readonly embeddingDim: number;
  private readonly mode: EmbeddingBagMode;
  private readonly paddingIdx: number | undefined;
  private weight_: GradTensor;

  constructor(
    numEmbeddings: number,
    embeddingDim: number,
    options: {
      readonly mode?: EmbeddingBagMode;
      readonly paddingIdx?: number;
    } = {}
  ) {
    super();

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

    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.mode = options.mode ?? "mean";
    this.paddingIdx = options.paddingIdx;

    if (this.paddingIdx !== undefined) {
      if (
        !Number.isInteger(this.paddingIdx) ||
        this.paddingIdx < 0 ||
        this.paddingIdx >= numEmbeddings
      ) {
        throw new InvalidParameterError(
          `paddingIdx must be in [0, ${numEmbeddings})`,
          "paddingIdx",
          this.paddingIdx
        );
      }
    }

    const w = randn([numEmbeddings, embeddingDim]);
    this.weight_ = parameter(w);
    this.registerParameter("weight", this.weight_);
  }

  /**
   * Forward pass: compute bag embeddings.
   *
   * @param indices - 1-D tensor of indices
   * @param offsets - 1-D tensor of starting indices for each bag
   * @returns Tensor of shape (numBags, embeddingDim)
   */
  forward(indices: AnyTensor, ...rest: AnyTensor[]): GradTensor {
    const offsetsInput = rest[0];
    if (!offsetsInput) {
      throw new InvalidParameterError(
        "EmbeddingBag requires offsets tensor as second argument",
        "offsets",
        undefined
      );
    }

    const idx = GradTensor.isGradTensor(indices) ? indices.tensor : indices;
    const off = GradTensor.isGradTensor(offsetsInput) ? offsetsInput.tensor : offsetsInput;

    if (idx.dtype === "string" || off.dtype === "string") {
      throw new DTypeError("EmbeddingBag indices and offsets must be numeric");
    }
    if (idx.ndim !== 1) {
      throw new ShapeError(`EmbeddingBag indices must be 1-D; got ${idx.ndim}-D`);
    }
    if (off.ndim !== 1) {
      throw new ShapeError(`EmbeddingBag offsets must be 1-D; got ${off.ndim}-D`);
    }

    const idxData = requireNumericData(idx.data, "EmbeddingBag");
    const offData = requireNumericData(off.data, "EmbeddingBag");
    const idxContig = isContiguous(idx.shape, idx.strides);
    const idxLogical = computeStrides(idx.shape);
    const offContig = isContiguous(off.shape, off.strides);
    const offLogical = computeStrides(off.shape);

    const weightTensor = this.weight_.tensor;
    const weightData = requireNumericData(weightTensor.data, "EmbeddingBag");

    const numBags = off.size;
    const outData = new Float64Array(numBags * this.embeddingDim);
    // Backward bookkeeping: rows contributing to each bag (excluding padding),
    // and for max mode the source row per (bag, feature).
    const bagMembers: number[][] = [];
    const bagCounts: number[] = [];
    const argmaxRow =
      this.mode === "max" ? new Int32Array(numBags * this.embeddingDim).fill(-1) : null;

    for (let b = 0; b < numBags; b++) {
      const bagStartOff = flatOffset(b, off.offset, offContig, offLogical, off.strides);
      const bagStart = Math.round(readAsNumber(offData, bagStartOff));

      let bagEnd: number;
      if (b + 1 < numBags) {
        const nextOff = flatOffset(b + 1, off.offset, offContig, offLogical, off.strides);
        bagEnd = Math.round(readAsNumber(offData, nextOff));
      } else {
        bagEnd = idx.size;
      }

      const outStart = b * this.embeddingDim;

      if (this.mode === "max") {
        // Initialize with -Infinity for max mode
        for (let j = 0; j < this.embeddingDim; j++) {
          outData[outStart + j] = -Infinity;
        }
      }

      let count = 0;
      const members: number[] = [];
      for (let i = bagStart; i < bagEnd; i++) {
        const idxOff = flatOffset(i, idx.offset, idxContig, idxLogical, idx.strides);
        const index = Math.round(readAsNumber(idxData, idxOff));

        if (index < 0 || index >= this.numEmbeddings) {
          throw new InvalidParameterError(
            `EmbeddingBag index ${index} out of range [0, ${this.numEmbeddings})`,
            "indices",
            index
          );
        }

        if (this.paddingIdx !== undefined && index === this.paddingIdx) {
          continue;
        }

        const rowStart = index * this.embeddingDim;
        count++;
        members.push(index);

        for (let j = 0; j < this.embeddingDim; j++) {
          const val = readAsNumber(weightData, rowStart + j);
          const outIdx = outStart + j;
          if (this.mode === "sum" || this.mode === "mean") {
            outData[outIdx] = (outData[outIdx] ?? 0) + val;
          } else {
            // max mode
            const cur = outData[outIdx] ?? -Infinity;
            if (val > cur) {
              outData[outIdx] = val;
              if (argmaxRow) argmaxRow[outIdx] = index;
            }
          }
        }
      }
      bagMembers.push(members);
      bagCounts.push(count);

      // For mean mode, divide by count
      if (this.mode === "mean" && count > 0) {
        for (let j = 0; j < this.embeddingDim; j++) {
          outData[outStart + j] = (outData[outStart + j] ?? 0) / count;
        }
      }

      // For max mode, replace -Infinity with 0 for empty bags
      if (this.mode === "max" && count === 0) {
        for (let j = 0; j < this.embeddingDim; j++) {
          outData[outStart + j] = 0;
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outData,
      shape: [numBags, this.embeddingDim],
      dtype: "float64",
      device: weightTensor.device,
    });

    const embDim = this.embeddingDim;
    const numEmbeddings = this.numEmbeddings;
    const mode = this.mode;
    return customOp(outTensor, [
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64Embed(g);
          const gw = new Float64Array(numEmbeddings * embDim);
          for (let b = 0; b < numBags; b++) {
            const outStart = b * embDim;
            if (mode === "max") {
              for (let j = 0; j < embDim; j++) {
                const row = argmaxRow ? (argmaxRow[outStart + j] ?? -1) : -1;
                if (row >= 0) gw[row * embDim + j]! += go[outStart + j] ?? 0;
              }
            } else {
              const members = bagMembers[b] ?? [];
              const scale =
                mode === "mean" && (bagCounts[b] ?? 0) > 0 ? 1 / (bagCounts[b] ?? 1) : 1;
              for (const row of members) {
                const rowStart = row * embDim;
                for (let j = 0; j < embDim; j++)
                  gw[rowStart + j]! += (go[outStart + j] ?? 0) * scale;
              }
            }
          }
          return Tensor.fromTypedArray({
            data: gw,
            shape: [numEmbeddings, embDim],
            dtype: "float64",
            device: weightTensor.device,
          });
        },
      ],
    ]);
  }

  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    return `EmbeddingBag(${this.numEmbeddings}, ${this.embeddingDim}, mode=${this.mode})`;
  }
}

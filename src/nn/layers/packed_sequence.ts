/**
 * Packed sequences for variable-length RNN inputs.
 *
 * Provides utilities to pack variable-length sequences into a compact
 * representation for efficient processing with RNN/LSTM/GRU layers,
 * and to unpack them back to padded tensors.
 *
 * The functions keep the dtype of their input tensors. Packed data is a plain
 * `Tensor` and does not take part in autograd.
 *
 * @module nn/packed_sequence
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { type DType, DTypeError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { readNumbers } from "../../ndarray/ops/_internal";
import { Tensor as TensorClass } from "../../ndarray/tensor/Tensor";

/**
 * A packed representation of variable-length sequences.
 *
 * Stores sequence data sorted by length (longest first) in a flat
 * tensor, along with batch sizes at each timestep.
 *
 * @example
 * ```ts
 * import { packSequence, unpackSequence } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const sequences = [
 *   tensor([[1, 2], [3, 4], [5, 6]]),       // length 3
 *   tensor([[7, 8]]),                         // length 1
 *   tensor([[9, 10], [11, 12]]),              // length 2
 * ];
 * const packed = packSequence(sequences);
 * const [unpacked, lengths] = unpackSequence(packed);
 * ```
 */
export type PackedSequence = {
  /** Packed data tensor of shape (totalElements, features) */
  readonly data: Tensor;
  /** Number of active sequences at each timestep */
  readonly batchSizes: number[];
  /** Original index of each sequence, listed in sorted (longest first) order */
  readonly sortedIndices: number[];
  /** Position of each original sequence within the sorted order (inverse of `sortedIndices`) */
  readonly unsortedIndices: number[];
  /** Feature dimension size */
  readonly featureSize: number;
  /**
   * True when the packed sequences were 1D, so {@link unpackSequence} returns 1D
   * tensors of shape `(seqLen)` instead of `(seqLen, 1)`.
   */
  readonly squeezeFeature?: boolean;
};

/** Numeric dtype of a tensor, or a `DTypeError` for string tensors. */
function numericDtype(t: Tensor, fn: string): Exclude<DType, "string"> {
  if (t.dtype === "string") {
    throw new DTypeError(`${fn} does not support string tensors`);
  }
  return t.dtype;
}

/** Build a tensor of `dtype` from float64 values (exact for every supported dtype). */
function fromFloat64(
  values: Float64Array,
  shape: number[],
  dtype: Exclude<DType, "string">
): Tensor {
  const t = TensorClass.fromTypedArray({
    data: values,
    shape,
    dtype: "float64",
    device: "cpu",
  });
  return dtype === "float64" ? t : t.astype(dtype);
}

/**
 * Sort sequence indices by descending length (stable) and lay the rows out
 * timestep by timestep, as PyTorch does.
 */
function packRows(
  lengths: readonly number[],
  featureSize: number,
  enforcesSorted: boolean,
  copyRow: (seqIdx: number, t: number, dst: Float64Array, dstOffset: number) => void
): Pick<PackedSequence, "batchSizes" | "sortedIndices" | "unsortedIndices"> & {
  readonly values: Float64Array;
} {
  const count = lengths.length;
  const sortedIndices = Array.from({ length: count }, (_, i) => i);
  if (enforcesSorted) {
    for (let i = 1; i < count; i++) {
      if ((lengths[i] ?? 0) > (lengths[i - 1] ?? 0)) {
        throw new InvalidParameterError(
          "enforcesSorted is true but the sequences are not sorted by decreasing length; " +
            "sort them or pass enforcesSorted=false",
          "enforcesSorted",
          enforcesSorted
        );
      }
    }
  } else {
    sortedIndices.sort((a, b) => (lengths[b] ?? 0) - (lengths[a] ?? 0));
  }

  const unsortedIndices = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    unsortedIndices[sortedIndices[i] as number] = i;
  }

  const sortedLengths = sortedIndices.map((i) => lengths[i] ?? 0);
  const maxLen = sortedLengths[0] ?? 0;

  // Sequences are sorted, so the active ones at step t are a prefix of the order.
  const batchSizes: number[] = [];
  let active = count;
  for (let t = 0; t < maxLen; t++) {
    while (active > 0 && (sortedLengths[active - 1] ?? 0) <= t) active--;
    batchSizes.push(active);
  }

  let totalElements = 0;
  for (const bs of batchSizes) totalElements += bs;
  const values = new Float64Array(totalElements * featureSize);
  let row = 0;
  for (let t = 0; t < maxLen; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      copyRow(sortedIndices[b] as number, t, values, row * featureSize);
      row++;
    }
  }

  return { batchSizes, sortedIndices, unsortedIndices, values };
}

/**
 * Pack a list of variable-length tensors into a {@link PackedSequence}.
 *
 * Sequences are sorted by length (longest first) and interleaved so
 * that at each timestep only active sequences are present. All sequences are
 * converted to the dtype of the first one.
 *
 * @param sequences - Array of 1D or 2D tensors. If 2D, shape is (seqLen, features).
 *   If 1D, treated as (seqLen, 1).
 * @param enforcesSorted - If true, the sequences must already be sorted by
 *   descending length (an error is thrown otherwise) and no sorting is done.
 *   Default: false.
 * @returns Packed sequence representation
 * @throws {InvalidParameterError} If `sequences` is empty, a sequence has zero
 *   length, or `enforcesSorted` is set but the lengths are not descending
 * @throws {ShapeError} If a sequence is not 1D or 2D, or feature sizes differ
 * @throws {DTypeError} If a sequence is a string tensor
 *
 * @example
 * ```ts
 * import { packSequence } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const seqs = [
 *   tensor([[1, 2], [3, 4], [5, 6]]),
 *   tensor([[7, 8]]),
 * ];
 * const packed = packSequence(seqs);
 * ```
 */
export function packSequence(sequences: readonly Tensor[], enforcesSorted = false): PackedSequence {
  if (sequences.length === 0) {
    throw new InvalidParameterError(
      "packSequence requires at least one sequence",
      "sequences",
      sequences.length
    );
  }

  // Determine lengths and feature size
  const lengths: number[] = [];
  let featureSize = -1;
  let allVectors = true;
  for (let i = 0; i < sequences.length; i++) {
    const seq = sequences[i] as Tensor;
    numericDtype(seq, "packSequence");
    let features: number;
    if (seq.ndim === 1) {
      features = 1;
    } else if (seq.ndim === 2) {
      features = seq.shape[1] ?? 0;
      allVectors = false;
    } else {
      throw new ShapeError(`Sequences must be 1D or 2D tensors; got ndim=${seq.ndim}`);
    }
    if (featureSize === -1) {
      featureSize = features;
    } else if (features !== featureSize) {
      throw new ShapeError(
        `All sequences must have the same feature dimension; got ${featureSize} and ${features}`
      );
    }
    const length = seq.shape[0] ?? 0;
    if (length === 0) {
      throw new InvalidParameterError(
        "packSequence does not accept zero-length sequences",
        "sequences",
        i
      );
    }
    lengths.push(length);
  }
  if (featureSize === 0) {
    throw new ShapeError("Sequences must have a positive feature dimension; got 0");
  }

  const dtype = numericDtype(sequences[0] as Tensor, "packSequence");
  const dense = sequences.map((seq) => readNumbers(seq.astype(dtype), "packSequence"));

  const packed = packRows(lengths, featureSize, enforcesSorted, (seqIdx, t, dst, dstOffset) => {
    const src = dense[seqIdx] as ArrayLike<number>;
    const base = t * featureSize;
    for (let f = 0; f < featureSize; f++) {
      dst[dstOffset + f] = src[base + f] as number;
    }
  });

  const totalElements = packed.values.length / featureSize;
  return {
    data: fromFloat64(packed.values, [totalElements, featureSize], dtype),
    batchSizes: packed.batchSizes,
    sortedIndices: packed.sortedIndices,
    unsortedIndices: packed.unsortedIndices,
    featureSize,
    squeezeFeature: allVectors,
  };
}

/**
 * Check a {@link PackedSequence} for internal consistency and return the
 * per-sequence lengths in sorted order together with the dense packed values.
 */
function inspectPacked(
  packed: PackedSequence,
  fn: string
): { readonly sortedLengths: number[]; readonly values: ArrayLike<number> } {
  const { data, batchSizes, sortedIndices, unsortedIndices, featureSize } = packed;
  const batchCount = sortedIndices.length;

  if (!Number.isInteger(featureSize) || featureSize <= 0) {
    throw new InvalidParameterError(
      "PackedSequence.featureSize must be a positive integer",
      "packed",
      featureSize
    );
  }
  if (unsortedIndices.length !== batchCount) {
    throw new InvalidParameterError(
      "PackedSequence.unsortedIndices and sortedIndices must have the same length",
      "packed",
      unsortedIndices.length
    );
  }
  for (let i = 0; i < batchCount; i++) {
    const original = sortedIndices[i];
    if (
      original === undefined ||
      !Number.isInteger(original) ||
      original < 0 ||
      original >= batchCount ||
      unsortedIndices[original] !== i
    ) {
      throw new InvalidParameterError(
        "PackedSequence.sortedIndices must be a permutation and unsortedIndices its inverse",
        "packed",
        sortedIndices
      );
    }
  }
  let total = 0;
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    const prev = t === 0 ? batchCount : (batchSizes[t - 1] ?? 0);
    if (!Number.isInteger(bs) || bs <= 0 || bs > prev || (t === 0 && bs !== batchCount)) {
      throw new InvalidParameterError(
        "PackedSequence.batchSizes must be positive and non-increasing, starting at the number of sequences",
        "packed",
        batchSizes
      );
    }
    total += bs;
  }
  if (data.size !== total * featureSize) {
    throw new ShapeError(
      `${fn}: packed data has ${data.size} elements but batchSizes and featureSize imply ${
        total * featureSize
      }`
    );
  }

  const sortedLengths = new Array<number>(batchCount).fill(0);
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      sortedLengths[b] = (sortedLengths[b] ?? 0) + 1;
    }
  }
  return { sortedLengths, values: readNumbers(data, fn) };
}

/**
 * Unpack a {@link PackedSequence} back to a list of tensors.
 *
 * Returns the sequences in the **original** order (before sorting). Each
 * tensor has shape `(seqLen, features)`, or `(seqLen)` when the packed
 * sequences were 1D.
 *
 * @param packed - Packed sequence to unpack
 * @returns Tuple of [sequences, lengths] where sequences are in original order
 * @throws {InvalidParameterError} If the packed sequence is inconsistent
 * @throws {ShapeError} If the packed data size does not match `batchSizes`
 *
 * @example
 * ```ts
 * import { packSequence, unpackSequence } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const seqs = [tensor([[1, 2], [3, 4]]), tensor([[5, 6]])];
 * const packed = packSequence(seqs);
 * const [unpacked, lengths] = unpackSequence(packed);
 * ```
 */
export function unpackSequence(packed: PackedSequence): [Tensor[], number[]] {
  const { data, batchSizes, sortedIndices, unsortedIndices, featureSize } = packed;
  const dtype = numericDtype(data, "unpackSequence");
  const batchCount = sortedIndices.length;
  const { sortedLengths, values } = inspectPacked(packed, "unpackSequence");

  const seqBuffers = sortedLengths.map((len) => new Float64Array(len * featureSize));
  const seqOffsets = new Array<number>(batchCount).fill(0);
  let row = 0;
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      const buf = seqBuffers[b] as Float64Array;
      const dst = (seqOffsets[b] as number) * featureSize;
      const src = row * featureSize;
      for (let f = 0; f < featureSize; f++) {
        buf[dst + f] = values[src + f] as number;
      }
      seqOffsets[b] = (seqOffsets[b] as number) + 1;
      row++;
    }
  }

  const squeeze = packed.squeezeFeature === true && featureSize === 1;
  const sortedSequences = seqBuffers.map((buf, b) => {
    const len = sortedLengths[b] ?? 0;
    return fromFloat64(buf, squeeze ? [len] : [len, featureSize], dtype);
  });

  // Restore original order
  const sequences: Tensor[] = new Array(batchCount);
  const lengths: number[] = new Array(batchCount);
  for (let i = 0; i < batchCount; i++) {
    const sortedPos = unsortedIndices[i] as number;
    sequences[i] = sortedSequences[sortedPos] as Tensor;
    lengths[i] = sortedLengths[sortedPos] ?? 0;
  }

  return [sequences, lengths];
}

/**
 * Pad a {@link PackedSequence} to a dense tensor.
 *
 * Creates a padded 3D tensor of shape (batchSize, maxLen, features)
 * filled with `paddingValue` after the end of shorter sequences. The returned
 * tensor has sequences in their **original** order.
 *
 * @param packed - Packed sequence to pad
 * @param totalLength - Optional total length to pad to (defaults to max sequence
 *   length). Must be at least the length of the longest sequence.
 * @param paddingValue - Value written after the end of each sequence (default: 0)
 * @returns Tuple of [paddedTensor, lengths]
 * @throws {InvalidParameterError} If `totalLength` is not an integer or is shorter
 *   than the longest sequence, or the packed sequence is inconsistent
 *
 * @example
 * ```ts
 * import { packSequence, padPackedSequence } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const seqs = [tensor([[1, 2], [3, 4]]), tensor([[5, 6]])];
 * const packed = packSequence(seqs);
 * const [padded, lengths] = padPackedSequence(packed);
 * // padded.shape = [2, 2, 2], lengths = [2, 1]
 * ```
 */
export function padPackedSequence(
  packed: PackedSequence,
  totalLength?: number,
  paddingValue = 0
): [Tensor, number[]] {
  const { batchSizes, sortedIndices, unsortedIndices, featureSize } = packed;
  const dtype = numericDtype(packed.data, "padPackedSequence");
  const batchSize = sortedIndices.length;
  const { sortedLengths, values } = inspectPacked(packed, "padPackedSequence");

  const longest = batchSizes.length;
  if (totalLength !== undefined && (!Number.isInteger(totalLength) || totalLength < longest)) {
    throw new InvalidParameterError(
      `totalLength must be an integer of at least the longest sequence length (${longest})`,
      "totalLength",
      totalLength
    );
  }
  const maxLen = totalLength ?? longest;

  const paddedData = new Float64Array(batchSize * maxLen * featureSize);
  if (paddingValue !== 0) paddedData.fill(paddingValue);

  let row = 0;
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      const original = sortedIndices[b] as number;
      const dst = (original * maxLen + t) * featureSize;
      const src = row * featureSize;
      for (let f = 0; f < featureSize; f++) {
        paddedData[dst + f] = values[src + f] as number;
      }
      row++;
    }
  }

  const lengths = new Array<number>(batchSize);
  for (let i = 0; i < batchSize; i++) {
    lengths[i] = sortedLengths[unsortedIndices[i] as number] ?? 0;
  }

  return [fromFloat64(paddedData, [batchSize, maxLen, featureSize], dtype), lengths];
}

/**
 * Pack a padded 3D tensor into a {@link PackedSequence}.
 *
 * Takes a padded tensor of shape (batch, maxLen, features) and
 * corresponding lengths, and creates a packed representation. The input dtype
 * is kept.
 *
 * @param input - Padded tensor of shape (batch, maxLen, features)
 * @param lengths - Actual lengths of each sequence in the batch
 * @param enforcesSorted - If true, the lengths must already be sorted descending
 *   (an error is thrown otherwise)
 * @returns Packed sequence representation
 * @throws {ShapeError} If `input` is not 3D
 * @throws {InvalidParameterError} If `lengths` has the wrong number of entries or
 *   an entry is not an integer in `[1, maxLen]`
 *
 * @example
 * ```ts
 * import { packPaddedSequence } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const padded = tensor([[[1, 2], [3, 4]], [[5, 6], [0, 0]]]);
 * const packed = packPaddedSequence(padded, [2, 1]);
 * ```
 */
export function packPaddedSequence(
  input: Tensor,
  lengths: readonly number[],
  enforcesSorted = false
): PackedSequence {
  if (input.ndim !== 3) {
    throw new ShapeError(
      `packPaddedSequence expects a 3D input tensor (batch, seqLen, features); got ndim=${input.ndim}`
    );
  }
  const dtype = numericDtype(input, "packPaddedSequence");

  const batchSize = input.shape[0] ?? 0;
  const maxLen = input.shape[1] ?? 0;
  const featureSize = input.shape[2] ?? 0;

  if (lengths.length !== batchSize) {
    throw new InvalidParameterError(
      `lengths must have ${batchSize} elements; got ${lengths.length}`,
      "lengths",
      lengths.length
    );
  }
  for (let b = 0; b < batchSize; b++) {
    const len = lengths[b] ?? 0;
    if (!Number.isInteger(len) || len <= 0 || len > maxLen) {
      throw new InvalidParameterError(
        `length[${b}] = ${len} is out of range [1, ${maxLen}]`,
        "lengths",
        len
      );
    }
  }
  if (featureSize === 0) {
    throw new ShapeError("packPaddedSequence requires a positive feature dimension; got 0");
  }

  const dense = readNumbers(input, "packPaddedSequence");
  const packed = packRows(lengths, featureSize, enforcesSorted, (seqIdx, t, dst, dstOffset) => {
    const base = (seqIdx * maxLen + t) * featureSize;
    for (let f = 0; f < featureSize; f++) {
      dst[dstOffset + f] = dense[base + f] as number;
    }
  });

  const totalElements = packed.values.length / featureSize;
  return {
    data: fromFloat64(packed.values, [totalElements, featureSize], dtype),
    batchSizes: packed.batchSizes,
    sortedIndices: packed.sortedIndices,
    unsortedIndices: packed.unsortedIndices,
    featureSize,
    squeezeFeature: false,
  };
}

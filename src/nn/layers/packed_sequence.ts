/**
 * Packed sequences for variable-length RNN inputs.
 *
 * Provides utilities to pack variable-length sequences into a compact
 * representation for efficient processing with RNN/LSTM/GRU layers,
 * and to unpack them back to padded tensors.
 *
 * @module nn/packed_sequence
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";

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
  /** Original indices before sorting (for restoring original order) */
  readonly sortedIndices: number[];
  /** Inverse of sortedIndices (for restoring original order) */
  readonly unsortedIndices: number[];
  /** Feature dimension size */
  readonly featureSize: number;
};

/**
 * Pack a list of variable-length tensors into a {@link PackedSequence}.
 *
 * Sequences are sorted by length (longest first) and interleaved so
 * that at each timestep only active sequences are present.
 *
 * @param sequences - Array of 1D or 2D tensors. If 2D, shape is (seqLen, features).
 *   If 1D, treated as (seqLen, 1).
 * @param enforcesSorted - If true, assumes sequences are already sorted by
 *   descending length and skips sorting. Default: false.
 * @returns Packed sequence representation
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
  let featureSize = 0;
  for (let i = 0; i < sequences.length; i++) {
    const seq = sequences[i]!;
    if (seq.ndim === 1) {
      lengths.push(seq.shape[0] ?? 0);
      featureSize = 1;
    } else if (seq.ndim === 2) {
      lengths.push(seq.shape[0] ?? 0);
      const f = seq.shape[1] ?? 0;
      if (featureSize === 0) {
        featureSize = f;
      } else if (f !== featureSize) {
        throw new ShapeError(
          `All sequences must have the same feature dimension; got ${featureSize} and ${f}`
        );
      }
    } else {
      throw new ShapeError(`Sequences must be 1D or 2D tensors; got ndim=${seq.ndim}`);
    }
    if ((lengths[i] ?? 0) === 0) {
      throw new InvalidParameterError(
        "packSequence does not accept zero-length sequences",
        "sequences",
        i
      );
    }
  }

  // Sort by descending length
  const sortedIndices = Array.from({ length: sequences.length }, (_, i) => i);
  if (!enforcesSorted) {
    sortedIndices.sort((a, b) => (lengths[b] ?? 0) - (lengths[a] ?? 0));
  }

  const unsortedIndices = new Array<number>(sequences.length);
  for (let i = 0; i < sortedIndices.length; i++) {
    unsortedIndices[sortedIndices[i]!] = i;
  }

  const sortedLengths = sortedIndices.map((i) => lengths[i] ?? 0);
  const maxLen = sortedLengths[0] ?? 0;
  const batchCount = sequences.length;

  // Compute batch sizes
  const batchSizes: number[] = [];
  for (let t = 0; t < maxLen; t++) {
    let count = 0;
    for (let b = 0; b < batchCount; b++) {
      if (t < (sortedLengths[b] ?? 0)) count++;
    }
    batchSizes.push(count);
  }

  // Pack data: interleave timesteps
  const totalElements = batchSizes.reduce((a, b) => a + b, 0);
  const packedData = new Float64Array(totalElements * featureSize);
  let offset = 0;

  for (let t = 0; t < maxLen; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      const seqIdx = sortedIndices[b]!;
      const seq = sequences[seqIdx]!;
      for (let f = 0; f < featureSize; f++) {
        if (seq.ndim === 1) {
          packedData[offset * featureSize + f] = Number(seq.data[seq.offset + t] ?? 0);
        } else {
          packedData[offset * featureSize + f] = Number(
            seq.data[seq.offset + t * featureSize + f] ?? 0
          );
        }
      }
      offset++;
    }
  }

  const dataTensor =
    featureSize === 1
      ? tensor(Array.from(packedData))
      : tensor(Array.from(packedData)).reshape([totalElements, featureSize]);

  return {
    data: dataTensor,
    batchSizes,
    sortedIndices,
    unsortedIndices,
    featureSize,
  };
}

/**
 * Unpack a {@link PackedSequence} back to a list of tensors.
 *
 * Returns the sequences in the **original** order (before sorting).
 *
 * @param packed - Packed sequence to unpack
 * @returns Tuple of [sequences, lengths] where sequences are in original order
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
  const batchCount = sortedIndices.length;

  // Determine lengths from batch sizes
  const sortedLengths = new Array<number>(batchCount).fill(0);
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      sortedLengths[b] = (sortedLengths[b] ?? 0) + 1;
    }
  }

  // Extract sequences in sorted order
  const sortedSequences: Tensor[] = new Array(batchCount);
  let offset = 0;
  const seqBuffers: Float64Array[] = [];
  for (let b = 0; b < batchCount; b++) {
    seqBuffers.push(new Float64Array((sortedLengths[b] ?? 0) * featureSize));
  }

  const seqOffsets = new Array<number>(batchCount).fill(0);
  for (let t = 0; t < batchSizes.length; t++) {
    const bs = batchSizes[t] ?? 0;
    for (let b = 0; b < bs; b++) {
      const buf = seqBuffers[b]!;
      const so = seqOffsets[b]!;
      for (let f = 0; f < featureSize; f++) {
        buf[so * featureSize + f] = Number(data.data[data.offset + offset * featureSize + f] ?? 0);
      }
      seqOffsets[b] = so + 1;
      offset++;
    }
  }

  for (let b = 0; b < batchCount; b++) {
    const len = sortedLengths[b] ?? 0;
    const buf = seqBuffers[b]!;
    if (featureSize === 1) {
      sortedSequences[b] = tensor(Array.from(buf));
    } else {
      sortedSequences[b] = tensor(Array.from(buf)).reshape([len, featureSize]);
    }
  }

  // Restore original order
  const sequences: Tensor[] = new Array(batchCount);
  const lengths: number[] = new Array(batchCount);
  for (let i = 0; i < batchCount; i++) {
    const sortedPos = unsortedIndices[i]!;
    sequences[i] = sortedSequences[sortedPos]!;
    lengths[i] = sortedLengths[sortedPos] ?? 0;
  }

  return [sequences, lengths];
}

/**
 * Pad a {@link PackedSequence} to a dense tensor.
 *
 * Creates a padded 3D tensor of shape (batchSize, maxLen, features)
 * with zero-padding for shorter sequences. The returned tensor has
 * sequences in their **original** order.
 *
 * @param packed - Packed sequence to pad
 * @param totalLength - Optional total length to pad to (defaults to max sequence length)
 * @returns Tuple of [paddedTensor, lengths]
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
  totalLength?: number
): [Tensor, number[]] {
  const [sequences, lengths] = unpackSequence(packed);
  const batchSize = sequences.length;
  const maxLen = totalLength ?? Math.max(...lengths);
  const featureSize = packed.featureSize;

  const paddedData = new Float64Array(batchSize * maxLen * featureSize);

  for (let b = 0; b < batchSize; b++) {
    const seq = sequences[b]!;
    const len = lengths[b] ?? 0;
    for (let t = 0; t < len; t++) {
      for (let f = 0; f < featureSize; f++) {
        if (seq.ndim === 1) {
          paddedData[(b * maxLen + t) * featureSize + f] = Number(seq.data[seq.offset + t] ?? 0);
        } else {
          paddedData[(b * maxLen + t) * featureSize + f] = Number(
            seq.data[seq.offset + t * featureSize + f] ?? 0
          );
        }
      }
    }
  }

  const padded = tensor(Array.from(paddedData)).reshape([batchSize, maxLen, featureSize]);

  return [padded, lengths];
}

/**
 * Pack a padded 3D tensor into a {@link PackedSequence}.
 *
 * Takes a padded tensor of shape (batch, maxLen, features) and
 * corresponding lengths, and creates a packed representation.
 *
 * @param input - Padded tensor of shape (batch, maxLen, features)
 * @param lengths - Actual lengths of each sequence in the batch
 * @param enforcesSorted - If true, assumes lengths are already sorted descending
 * @returns Packed sequence representation
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

  // Build individual sequences from the padded tensor
  const sequences: Tensor[] = [];
  for (let b = 0; b < batchSize; b++) {
    const len = lengths[b] ?? 0;
    if (len <= 0 || len > maxLen) {
      throw new InvalidParameterError(
        `length[${b}] = ${len} is out of range [1, ${maxLen}]`,
        "lengths",
        len
      );
    }
    const seqData = new Float64Array(len * featureSize);
    for (let t = 0; t < len; t++) {
      for (let f = 0; f < featureSize; f++) {
        seqData[t * featureSize + f] = Number(
          input.data[input.offset + (b * maxLen + t) * featureSize + f] ?? 0
        );
      }
    }
    sequences.push(tensor(Array.from(seqData)).reshape([len, featureSize]));
  }

  return packSequence(sequences, enforcesSorted);
}

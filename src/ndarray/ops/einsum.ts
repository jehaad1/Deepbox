/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import { dot } from "../linalg/index";
import { tensor as createTensor } from "../tensor/index";
import { transpose } from "../tensor/shape";
import { Tensor } from "../tensor/Tensor";

/**
 * Parse an einsum subscript string into input subscripts and output subscript.
 * E.g. "ij,jk->ik" => { inputs: [["i","j"], ["j","k"]], output: ["i","k"] }
 */
function parseSubscripts(subscripts: string, nInputs: number) {
  const trimmed = subscripts.replace(/\s/g, "");
  if (trimmed.includes(".")) {
    // Treating '.' as an ordinary index label would silently compute the
    // wrong contraction for NumPy-style ellipsis specs like '...i'.
    throw new InvalidParameterError(
      "einsum: ellipsis ('...') notation is not supported; spell out all index labels explicitly",
      "subscripts",
      subscripts
    );
  }
  const arrowIdx = trimmed.indexOf("->");
  let inputPart: string;
  let outputPart: string | null;

  if (arrowIdx >= 0) {
    inputPart = trimmed.slice(0, arrowIdx);
    outputPart = trimmed.slice(arrowIdx + 2);
  } else {
    inputPart = trimmed;
    outputPart = null;
  }

  const inputStrs = inputPart.split(",");
  if (inputStrs.length !== nInputs) {
    throw new InvalidParameterError(
      `einsum: expected ${nInputs} input subscripts, got ${inputStrs.length}`,
      "subscripts",
      subscripts
    );
  }

  const inputs = inputStrs.map((s) => s.split(""));

  if (outputPart !== null) {
    return { inputs, output: outputPart.split("") };
  }

  // Implicit output: indices that appear exactly once, in alphabetical order
  const counts = new Map<string, number>();
  for (const inp of inputs) {
    for (const ch of inp) {
      counts.set(ch, (counts.get(ch) ?? 0) + 1);
    }
  }
  const output = [...counts.entries()]
    .filter(([, c]) => c === 1)
    .map(([ch]) => ch)
    .sort();
  return { inputs, output };
}

/**
 * Einstein summation convention.
 *
 * Supports arbitrary subscript notation for tensor contractions.
 * Common patterns:
 * - "ij,jk->ik" (matrix multiply)
 * - "ii->i" (diagonal)
 * - "ii->" (trace)
 * - "ij->ji" (transpose)
 * - "i,i->" (dot product)
 * - "ij,j->i" (matrix-vector multiply)
 *
 * @param subscripts - Subscript string in Einstein notation
 * @param tensors - Input tensors
 * @returns Result tensor
 */
export function einsum(subscripts: string, ...tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("einsum requires at least one tensor", "tensors");
  }

  const { inputs, output } = parseSubscripts(subscripts, tensors.length);

  // Build dimension map: label -> size
  const dimSizes = new Map<string, number>();
  for (let t = 0; t < tensors.length; t++) {
    const tsr = tensors[t]!;
    const labels = inputs[t]!;
    if (labels.length !== tsr.ndim) {
      throw new ShapeError(
        `einsum: operand ${t} has ${tsr.ndim} dimensions but subscript has ${labels.length} indices`
      );
    }
    for (let d = 0; d < labels.length; d++) {
      const label = labels[d]!;
      const size = tsr.shape[d]!;
      const existing = dimSizes.get(label);
      if (existing !== undefined && existing !== size) {
        throw new ShapeError(
          `einsum: dimension '${label}' has conflicting sizes ${existing} and ${size}`
        );
      }
      dimSizes.set(label, size);
    }
  }

  // Fast path: two-operand single-contraction matmul patterns
  // ("ij,jk->ik", "ij,kj->ik", "ji,jk->ik", ...) route to the tuned dot
  // kernel via transposed views (the generic per-element machinery is
  // ~100x slower for these).
  if (tensors.length === 2 && inputs.length === 2) {
    const [la, lb] = [inputs[0]!, inputs[1]!];
    const t0 = tensors[0]!;
    const t1 = tensors[1]!;
    if (
      la.length === 2 &&
      lb.length === 2 &&
      output.length === 2 &&
      t0.dtype === t1.dtype &&
      (t0.dtype === "float32" || t0.dtype === "float64")
    ) {
      const labels = new Set([...la, ...lb]);
      const shared = [...labels].filter((l) => la.includes(l) && lb.includes(l));
      const outSet = new Set(output);
      if (
        shared.length === 1 &&
        !outSet.has(shared[0]!) &&
        outSet.has(la.find((l) => l !== shared[0])!) &&
        outSet.has(lb.find((l) => l !== shared[0])!) &&
        la[0] !== la[1] &&
        lb[0] !== lb[1]
      ) {
        const k = shared[0]!;
        const aRow = la.find((l) => l !== k)!;
        const bCol = lb.find((l) => l !== k)!;
        // Orient operands as [row, k] x [k, col].
        const A = la[1] === k ? t0 : transpose(t0);
        const B = lb[0] === k ? t1 : transpose(t1);
        const prod = dot(A, B);
        // Output label order may be [bCol, aRow] -> transpose the result.
        return output[0] === aRow && output[1] === bCol ? prod : transpose(prod);
      }
    }
  }

  // Determine all unique labels and which are summed over
  const allLabels = new Set<string>();
  for (const inp of inputs) {
    for (const ch of inp) allLabels.add(ch);
  }
  const outputSet = new Set(output);
  if (outputSet.size !== output.length) {
    throw new InvalidParameterError(
      "einsum: output subscript contains a repeated index label",
      "subscripts",
      subscripts
    );
  }
  for (const l of output) {
    if (!allLabels.has(l)) {
      throw new InvalidParameterError(
        `einsum: output index '${l}' does not appear in any input subscript`,
        "subscripts",
        subscripts
      );
    }
  }
  const sumLabels = [...allLabels].filter((l) => !outputSet.has(l));

  // Compute output shape
  const outputShape = output.map((l) => dimSizes.get(l) ?? 1);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  // Get flat data from all tensors. Note: `t.data` is the underlying buffer,
  // which may be shared by a view, so element access must honour each tensor's
  // own strides and offset rather than assuming a contiguous row-major layout.
  const flatData: Float64Array[] = tensors.map((t) => {
    const d = t.data;
    if (d instanceof Float64Array) return d;
    if (d instanceof BigInt64Array) {
      // Float64Array's constructor cannot ingest BigInt elements directly
      const out = new Float64Array(d.length);
      for (let i = 0; i < d.length; i++) out[i] = Number(d[i]);
      return out;
    }
    if (Array.isArray(d)) {
      throw new InvalidParameterError("einsum is not defined for string dtype", "tensors");
    }
    return new Float64Array(d as ArrayLike<number>);
  });

  // Use each tensor's actual strides (handles transposed / non-contiguous views).
  const tensorStrides: readonly number[][] = tensors.map((t) => [...t.strides]);
  const tensorOffsets: number[] = tensors.map((t) => t.offset);

  // For each output element, iterate over all summed indices
  const result = new Float64Array(outputSize);

  // Precompute sum dimension sizes
  const sumSizes = sumLabels.map((l) => dimSizes.get(l) ?? 1);
  const nSum = sumSizes.reduce((a, b) => a * b, 1);

  // Build label-to-dimension mappings for output and sum indices
  for (let outIdx = 0; outIdx < outputSize; outIdx++) {
    // Decode outIdx into output label values
    const outputValues = new Map<string, number>();
    let rem = outIdx;
    for (let d = output.length - 1; d >= 0; d--) {
      const size = outputShape[d]!;
      outputValues.set(output[d]!, rem % size);
      rem = Math.floor(rem / size);
    }

    let total = 0;

    for (let sumIdx = 0; sumIdx < nSum; sumIdx++) {
      // Decode sumIdx into sum label values
      const labelValues = new Map(outputValues);
      let srem = sumIdx;
      for (let s = sumLabels.length - 1; s >= 0; s--) {
        const size = sumSizes[s]!;
        labelValues.set(sumLabels[s]!, srem % size);
        srem = Math.floor(srem / size);
      }

      // Compute product of all tensor elements at these indices
      let product = 1;
      for (let t = 0; t < tensors.length; t++) {
        const labels = inputs[t]!;
        const strides = tensorStrides[t]!;
        const data = flatData[t]!;
        let flatIdx = tensorOffsets[t]!;
        for (let d = 0; d < labels.length; d++) {
          flatIdx += (labelValues.get(labels[d]!) ?? 0) * strides[d]!;
        }
        product *= data[flatIdx] ?? 0;
      }

      total += product;
    }

    result[outIdx] = total;
  }

  if (outputShape.length === 0) {
    // Scalar result
    return createTensor(result[0] ?? 0);
  }

  return Tensor.fromTypedArray({
    data: result,
    shape: outputShape,
    dtype: "float64",
    device: "cpu",
  });
}

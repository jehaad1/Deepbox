/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { DType } from "../../core";
import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { dot } from "../linalg/index";
import { transpose } from "../tensor/shape";
import { Tensor } from "../tensor/Tensor";
import { contiguous } from "./utils";

/** A parsed operand or output term: a letter label or the ellipsis marker. */
type Token = string;
const ELLIPSIS = "...";
/** Prefix of the internal labels given to ellipsis dimensions (never a letter). */
const ELLIPSIS_LABEL = "…";

function isLetter(ch: string): boolean {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z");
}

/** Split one subscript term into letter tokens and at most one ellipsis token. */
function tokenize(term: string, subscripts: string): Token[] {
  const tokens: Token[] = [];
  let sawEllipsis = false;
  for (let i = 0; i < term.length; i++) {
    const ch = term.charAt(i);
    if (ch === ".") {
      if (term.startsWith(ELLIPSIS, i) && !sawEllipsis) {
        tokens.push(ELLIPSIS);
        sawEllipsis = true;
        i += 2;
        continue;
      }
      throw new InvalidParameterError(
        `einsum: '.' is only valid as a single '...' per term; received "${subscripts}"`,
        "subscripts",
        subscripts
      );
    }
    if (!isLetter(ch)) {
      throw new InvalidParameterError(
        `einsum: invalid character '${ch}' in subscripts "${subscripts}"; labels must be letters`,
        "subscripts",
        subscripts
      );
    }
    tokens.push(ch);
  }
  return tokens;
}

interface ParsedSubscripts {
  inputs: Token[][];
  /** Explicit output tokens, or null for implicit mode. */
  output: Token[] | null;
}

/**
 * Parse an einsum subscript string.
 * E.g. "ij,jk->ik" => inputs [["i","j"], ["j","k"]], output ["i","k"].
 */
function parseSubscripts(subscripts: string, nInputs: number): ParsedSubscripts {
  const trimmed = subscripts.replace(/\s/g, "");
  const arrowIdx = trimmed.indexOf("->");
  const inputPart = arrowIdx >= 0 ? trimmed.slice(0, arrowIdx) : trimmed;
  const outputPart = arrowIdx >= 0 ? trimmed.slice(arrowIdx + 2) : null;

  const inputStrs = inputPart.split(",");
  if (inputStrs.length !== nInputs) {
    throw new InvalidParameterError(
      `einsum: expected ${nInputs} input subscripts, got ${inputStrs.length}`,
      "subscripts",
      subscripts
    );
  }
  return {
    inputs: inputStrs.map((s) => tokenize(s, subscripts)),
    output: outputPart === null ? null : tokenize(outputPart, subscripts),
  };
}

/** Result dtype: float32 and int32 are kept when every operand has them, otherwise float64. */
function resultDtype(tensors: readonly Tensor[]): "float32" | "float64" | "int32" {
  const first = tensors[0]?.dtype;
  if (first !== undefined && tensors.every((t) => t.dtype === first)) {
    if (first === "float32" || first === "int32") return first;
  }
  return "float64";
}

/** Numeric view of a tensor's whole buffer (physical layout; strides/offset still apply). */
function numericBuffer(t: Tensor): ArrayLike<number> {
  const d = t.data;
  if (Array.isArray(d)) {
    throw new DTypeError("einsum is not defined for string dtype");
  }
  if (d instanceof BigInt64Array) {
    // int64 values are summed as doubles, exact up to 2^53.
    return Float64Array.from(d, (v) => Number(v));
  }
  return d;
}

/**
 * Einstein summation convention.
 *
 * Supports explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) subscripts, any
 * number of operands, repeated labels within one operand (diagonals and
 * traces), and `...` for broadcast dimensions. Dimensions of size 1 broadcast
 * against larger dimensions that share their label, as in NumPy.
 *
 * Common patterns:
 * - `"ij,jk->ik"` (matrix multiply)
 * - `"ii->i"` (diagonal)
 * - `"ii->"` (trace)
 * - `"ij->ji"` (transpose)
 * - `"i,i->"` (dot product)
 * - `"ij,j->i"` (matrix-vector multiply)
 * - `"...ij,...jk->...ik"` (batched matrix multiply)
 *
 * In implicit mode the output holds the labels that appear exactly once, in
 * alphabetical order (uppercase before lowercase), preceded by any ellipsis
 * dimensions.
 *
 * The result is float32 when every operand is float32, int32 when every
 * operand is int32, and float64 otherwise. Products are accumulated in
 * float64. int64 operands are converted to doubles (exact up to 2^53).
 *
 * @param subscripts - Subscript string in Einstein notation
 * @param tensors - Input tensors
 * @returns Result tensor
 * @throws {InvalidParameterError} If the subscripts are malformed or do not match the operand count
 * @throws {ShapeError} If an operand's rank does not match its subscript or label sizes conflict
 * @throws {DTypeError} If an operand has string dtype
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * einsum("ij,jk->ik", a, a);  // [[7, 10], [15, 22]]
 * einsum("ii->", a);          // 5
 * ```
 */
export function einsum(subscripts: string, ...tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("einsum requires at least one tensor", "tensors");
  }
  for (const t of tensors) {
    if (t.dtype === "string") {
      throw new DTypeError("einsum is not defined for string dtype");
    }
  }

  const parsed = parseSubscripts(subscripts, tensors.length);

  // ---- Expand ellipses into internal labels and match ranks ----
  const letterCounts = new Map<string, number>();
  const ellipsisRank: number[] = [];
  let maxEllipsis = 0;
  for (let t = 0; t < tensors.length; t++) {
    const tsr = tensors[t] as Tensor;
    const tokens = parsed.inputs[t] as Token[];
    const hasEllipsis = tokens.includes(ELLIPSIS);
    const explicit = tokens.length - (hasEllipsis ? 1 : 0);
    if (hasEllipsis ? tsr.ndim < explicit : tsr.ndim !== explicit) {
      throw new ShapeError(
        `einsum: operand ${t} has ${tsr.ndim} dimensions but subscript has ${explicit} indices` +
          (hasEllipsis ? " (plus '...')" : "")
      );
    }
    const k = hasEllipsis ? tsr.ndim - explicit : 0;
    ellipsisRank.push(k);
    maxEllipsis = Math.max(maxEllipsis, k);
    for (const tok of tokens) {
      if (tok !== ELLIPSIS) letterCounts.set(tok, (letterCounts.get(tok) ?? 0) + 1);
    }
  }
  const ellipsisLabels = Array.from({ length: maxEllipsis }, (_, i) => `${ELLIPSIS_LABEL}${i}`);

  const inputs: string[][] = parsed.inputs.map((tokens, t) => {
    const k = ellipsisRank[t] as number;
    const labels: string[] = [];
    for (const tok of tokens) {
      if (tok === ELLIPSIS) {
        for (let i = 0; i < k; i++) labels.push(ellipsisLabels[maxEllipsis - k + i] as string);
      } else {
        labels.push(tok);
      }
    }
    return labels;
  });

  let output: string[];
  if (parsed.output === null) {
    // Implicit output: ellipsis dimensions, then labels that appear exactly once, sorted.
    const once = [...letterCounts.entries()]
      .filter(([, c]) => c === 1)
      .map(([ch]) => ch)
      .sort();
    output = [...ellipsisLabels, ...once];
  } else {
    output = [];
    for (const tok of parsed.output) {
      if (tok === ELLIPSIS) output.push(...ellipsisLabels);
      else output.push(tok);
    }
    if (maxEllipsis > 0 && !parsed.output.includes(ELLIPSIS)) {
      throw new InvalidParameterError(
        "einsum: the operands contain '...' dimensions but the output subscript has no '...'",
        "subscripts",
        subscripts
      );
    }
  }
  if (new Set(output).size !== output.length) {
    throw new InvalidParameterError(
      "einsum: output subscript contains a repeated index label",
      "subscripts",
      subscripts
    );
  }
  for (const l of output) {
    if (!inputs.some((labels) => labels.includes(l))) {
      throw new InvalidParameterError(
        `einsum: output index '${l}' does not appear in any input subscript`,
        "subscripts",
        subscripts
      );
    }
  }

  // ---- Label sizes (size-1 dimensions broadcast) ----
  const dimSizes = new Map<string, number>();
  for (let t = 0; t < tensors.length; t++) {
    const tsr = tensors[t] as Tensor;
    const labels = inputs[t] as string[];
    const local = new Map<string, number>();
    for (let d = 0; d < labels.length; d++) {
      const label = labels[d] as string;
      const size = tsr.shape[d] as number;
      const seen = local.get(label);
      if (seen !== undefined && seen !== size) {
        throw new ShapeError(
          `einsum: dimension '${label}' has conflicting sizes ${seen} and ${size} within operand ${t}`
        );
      }
      local.set(label, size);
      const existing = dimSizes.get(label);
      if (existing === undefined || existing === 1) {
        dimSizes.set(label, size);
      } else if (size !== 1 && size !== existing) {
        throw new ShapeError(
          `einsum: dimension '${label}' has conflicting sizes ${existing} and ${size}`
        );
      }
    }
  }

  const outDtype = resultDtype(tensors);

  // ---- Fast path: 2-D x 2-D single-contraction (matmul-like) patterns ----
  // These route to the tuned dot kernel through transposed views; the generic
  // loop below is far slower for them.
  if (tensors.length === 2 && output.length === 2) {
    const [la, lb] = [inputs[0] as string[], inputs[1] as string[]];
    const t0 = tensors[0] as Tensor;
    const t1 = tensors[1] as Tensor;
    if (
      la.length === 2 &&
      lb.length === 2 &&
      la[0] !== la[1] &&
      lb[0] !== lb[1] &&
      t0.dtype === t1.dtype &&
      (t0.dtype === "float32" || t0.dtype === "float64")
    ) {
      const shared = la.filter((l) => lb.includes(l));
      const k = shared[0];
      if (shared.length === 1 && k !== undefined && !output.includes(k)) {
        const aRow = la.find((l) => l !== k) as string;
        const bCol = lb.find((l) => l !== k) as string;
        const kSizeA = t0.shape[la.indexOf(k)];
        const kSizeB = t1.shape[lb.indexOf(k)];
        if (output.includes(aRow) && output.includes(bCol) && kSizeA === kSizeB) {
          // Orient operands as [row, k] x [k, col].
          const A = la[1] === k ? t0 : transpose(t0);
          const B = lb[0] === k ? t1 : transpose(t1);
          const prod = dot(A, B);
          // Output label order may be [bCol, aRow] -> transpose the result.
          if (output[0] === aRow) return prod;
          const swapped = transpose(prod);
          // Host results are returned densely packed like the generic path.
          return swapped.isDeviceTensor ? swapped : contiguous(swapped);
        }
      }
    }
  }

  // ---- Generic path: strided loops over (output labels, summed labels) ----
  const sumLabels: string[] = [];
  for (const labels of inputs) {
    for (const l of labels) {
      if (!output.includes(l) && !sumLabels.includes(l)) sumLabels.push(l);
    }
  }
  const loopLabels = [...output, ...sumLabels];
  const nOut = output.length;
  const nSum = sumLabels.length;
  const nOps = tensors.length;
  const sizes = loopLabels.map((l) => dimSizes.get(l) ?? 1);

  // Stride of each operand along each loop label. Repeated labels add their
  // strides (that walks the diagonal); broadcast (size-1) dimensions get 0.
  const strides: number[][] = tensors.map((tsr, t) => {
    const labels = inputs[t] as string[];
    const out = new Array<number>(loopLabels.length).fill(0);
    for (let d = 0; d < labels.length; d++) {
      const li = loopLabels.indexOf(labels[d] as string);
      if ((tsr.shape[d] as number) === sizes[li]) {
        out[li] = (out[li] as number) + (tsr.strides[d] as number);
      }
    }
    return out;
  });

  const outShape = sizes.slice(0, nOut);
  let outputSize = 1;
  for (const s of outShape) outputSize *= s;
  let sumSize = 1;
  for (let j = nOut; j < loopLabels.length; j++) sumSize *= sizes[j] as number;

  const result = new Float64Array(outputSize);
  if (outputSize > 0 && sumSize > 0) {
    const buffers = tensors.map(numericBuffer);
    const cur = tensors.map((t) => t.offset);
    const outIdx = new Array<number>(nOut).fill(0);
    const sumIdx = new Array<number>(nSum).fill(0);
    const sumBase = new Array<number>(nOps).fill(0);

    for (let o = 0; o < outputSize; o++) {
      let total = 0;
      for (let t = 0; t < nOps; t++) sumBase[t] = cur[t] as number;
      sumIdx.fill(0);

      for (let s = 0; s < sumSize; s++) {
        let product = 1;
        for (let t = 0; t < nOps; t++) {
          product *= (buffers[t] as ArrayLike<number>)[sumBase[t] as number] as number;
        }
        total += product;
        for (let j = nSum - 1; j >= 0; j--) {
          const li = nOut + j;
          const next = (sumIdx[j] as number) + 1;
          if (next < (sizes[li] as number)) {
            sumIdx[j] = next;
            for (let t = 0; t < nOps; t++) {
              sumBase[t] = (sumBase[t] as number) + ((strides[t] as number[])[li] as number);
            }
            break;
          }
          sumIdx[j] = 0;
          for (let t = 0; t < nOps; t++) {
            sumBase[t] =
              (sumBase[t] as number) -
              ((strides[t] as number[])[li] as number) * ((sizes[li] as number) - 1);
          }
        }
      }
      result[o] = total;

      // Advance the output odometer.
      for (let j = nOut - 1; j >= 0; j--) {
        const next = (outIdx[j] as number) + 1;
        if (next < (sizes[j] as number)) {
          outIdx[j] = next;
          for (let t = 0; t < nOps; t++) {
            cur[t] = (cur[t] as number) + ((strides[t] as number[])[j] as number);
          }
          break;
        }
        outIdx[j] = 0;
        for (let t = 0; t < nOps; t++) {
          cur[t] =
            (cur[t] as number) -
            ((strides[t] as number[])[j] as number) * ((sizes[j] as number) - 1);
        }
      }
    }
  }

  const dtype: Exclude<DType, "string"> = outDtype;
  const data =
    outDtype === "float32"
      ? new Float32Array(result)
      : outDtype === "int32"
        ? new Int32Array(result)
        : result;
  return Tensor.fromTypedArray({ data, shape: outShape, dtype, device: "cpu" });
}

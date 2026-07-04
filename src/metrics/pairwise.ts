/**
 * Pairwise distance metrics and ranking metrics.
 *
 * @module metrics/pairwise
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Metrics}
 */

import { ShapeError } from "../core";
import { type Tensor, Tensor as TensorClass } from "../ndarray";

/**
 * Compute pairwise Euclidean distances between rows of X.
 *
 * @param X - 2-D tensor of shape [n, d]
 * @returns 2-D tensor of shape [n, n] containing pairwise distances
 */
/** Densify a 2-D tensor's rows into a contiguous Float64Array [n*d]. */
function densify2D(X: Tensor, n: number, d: number): Float64Array {
  const out = new Float64Array(n * d);
  const s0 = X.strides[0] ?? 0;
  const s1 = X.strides[1] ?? 0;
  const data = X.data;
  let w = 0;
  for (let i = 0; i < n; i++) {
    const base = X.offset + i * s0;
    for (let k = 0; k < d; k++) {
      out[w++] = Number(data[base + k * s1]);
    }
  }
  return out;
}

/** Wrap a distance matrix buffer as an [n, n] float64 tensor without copies. */
function wrapSquare(data: Float64Array, n: number): Tensor {
  return TensorClass.fromTypedArray({ data, shape: [n, n], dtype: "float64", device: "cpu" });
}

export function pairwiseEuclidean(X: Tensor): Tensor {
  if (X.ndim !== 2) {
    throw new ShapeError(`pairwiseEuclidean requires 2D input; got ndim=${X.ndim}`);
  }
  const n = X.shape[0] ?? 0;
  const d = X.shape[1] ?? 0;
  const data = new Float64Array(n * n);

  const dense = densify2D(X, n, d);
  for (let i = 0; i < n; i++) {
    const baseI = i * d;
    for (let j = i + 1; j < n; j++) {
      const baseJ = j * d;
      let sum = 0;
      for (let k = 0; k < d; k++) {
        const diff = (dense[baseI + k] as number) - (dense[baseJ + k] as number);
        sum += diff * diff;
      }
      const dist = Math.sqrt(sum);
      data[i * n + j] = dist;
      data[j * n + i] = dist;
    }
  }

  return wrapSquare(data, n);
}

/**
 * Compute pairwise Manhattan (L1) distances between rows of X.
 *
 * @param X - 2-D tensor of shape [n, d]
 * @returns 2-D tensor of shape [n, n] containing pairwise L1 distances
 */
export function pairwiseManhattan(X: Tensor): Tensor {
  if (X.ndim !== 2) {
    throw new ShapeError(`pairwiseManhattan requires 2D input; got ndim=${X.ndim}`);
  }
  const n = X.shape[0] ?? 0;
  const d = X.shape[1] ?? 0;
  const data = new Float64Array(n * n);

  const dense = densify2D(X, n, d);
  for (let i = 0; i < n; i++) {
    const baseI = i * d;
    for (let j = i; j < n; j++) {
      const baseJ = j * d;
      let sum = 0;
      for (let k = 0; k < d; k++) {
        sum += Math.abs(dense[baseI + k]! - dense[baseJ + k]!);
      }
      data[i * n + j] = sum;
      data[j * n + i] = sum;
    }
  }

  return wrapSquare(data, n);
}

/**
 * Compute pairwise cosine distances between rows of X.
 *
 * cosine_distance = 1 - cosine_similarity
 *
 * @param X - 2-D tensor of shape [n, d]
 * @returns 2-D tensor of shape [n, n] containing pairwise cosine distances
 */
export function pairwiseCosine(X: Tensor): Tensor {
  if (X.ndim !== 2) {
    throw new ShapeError(`pairwiseCosine requires 2D input; got ndim=${X.ndim}`);
  }
  const n = X.shape[0] ?? 0;
  const d = X.shape[1] ?? 0;
  const data = new Float64Array(n * n);

  const dense = densify2D(X, n, d);
  // Normalize each row to unit length once (O(n·d)); cosine similarity is then
  // the plain dot product, so the O(n²·d) inner loop drops the two redundant
  // norm accumulations it previously recomputed for every pair.
  const zeroRow = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    let norm = 0;
    for (let k = 0; k < d; k++) {
      const v = dense[base + k] as number;
      norm += v * v;
    }
    if (norm === 0) {
      zeroRow[i] = 1;
    } else {
      const inv = 1 / Math.sqrt(norm);
      for (let k = 0; k < d; k++) dense[base + k] = (dense[base + k] as number) * inv;
    }
  }

  // Rows are unit-normalized, so cosine similarity is a plain dot product and
  // the inner loop drops the two per-pair norm accumulations of the old code.
  for (let i = 0; i < n; i++) {
    const baseI = i * d;
    // Diagonal cosine distance is 0 (row with itself); leave data[i*n+i] = 0.
    for (let j = i + 1; j < n; j++) {
      const baseJ = j * d;
      let dist: number;
      if (zeroRow[i] || zeroRow[j]) {
        dist = 0;
      } else {
        let dotp = 0;
        for (let k = 0; k < d; k++) {
          dotp += (dense[baseI + k] as number) * (dense[baseJ + k] as number);
        }
        dist = 1 - dotp;
      }
      data[i * n + j] = dist;
      data[j * n + i] = dist;
    }
  }

  return wrapSquare(data, n);
}

// ---- Ranking metrics ----

function extractArray1D(t: Tensor, name: string): number[] {
  if (t.ndim !== 1) {
    throw new ShapeError(`${name} requires 1D input; got ndim=${t.ndim}`);
  }
  const arr: number[] = [];
  for (let i = 0; i < t.size; i++) {
    arr.push(Number(t.data[t.offset + i]));
  }
  return arr;
}

/** Single-query NDCG over plain arrays. Shared by the 1-D and 2-D paths. */
function ndcgSingle(trueArr: number[], scoreArr: number[], k?: number): number {
  const n = trueArr.length;
  if (n === 0) return 0;
  const topK = k ?? n;

  // Sort by predicted scores descending
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => (scoreArr[b] ?? 0) - (scoreArr[a] ?? 0));

  // DCG with LINEAR gains (gain = relevance), matching scikit-learn's
  // ndcg_score default. Exponential gains (2^rel − 1) would diverge from the
  // documented sklearn parity for graded relevance.
  let dcg = 0;
  for (let i = 0; i < Math.min(topK, n); i++) {
    const rel = trueArr[indices[i]!] ?? 0;
    dcg += rel / Math.log2(i + 2);
  }

  // Ideal DCG (sort by true relevance descending)
  const idealIndices = Array.from({ length: n }, (_, i) => i);
  idealIndices.sort((a, b) => (trueArr[b] ?? 0) - (trueArr[a] ?? 0));

  let idcg = 0;
  for (let i = 0; i < Math.min(topK, n); i++) {
    const rel = trueArr[idealIndices[i]!] ?? 0;
    idcg += rel / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

/** Extract row `r` of a 2-D tensor as a plain array, respecting strides. */
function extractRow2D(t: Tensor, r: number): number[] {
  const cols = t.shape[1] ?? 0;
  const s0 = t.strides[0] ?? 0;
  const s1 = t.strides[1] ?? 0;
  const row: number[] = [];
  for (let c = 0; c < cols; c++) {
    row.push(Number(t.data[t.offset + r * s0 + c * s1]));
  }
  return row;
}

/**
 * Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.
 *
 * Accepts either a single query (1-D tensors) or a batch of queries
 * (2-D tensors of shape `[nSamples, nLabels]`), in which case the mean
 * NDCG across samples is returned — matching scikit-learn's `ndcg_score`.
 *
 * @param yTrue - True relevance scores (1-D or 2-D tensor)
 * @param yScore - Predicted scores (1-D or 2-D tensor, same shape as yTrue)
 * @param k - Number of top results to consider (default: all)
 * @returns NDCG score in [0, 1]
 */
export function ndcgScore(yTrue: Tensor, yScore: Tensor, k?: number): number {
  if (yTrue.ndim !== yScore.ndim) {
    throw new ShapeError(
      `ndcgScore: yTrue and yScore must have same ndim; got ${yTrue.ndim} vs ${yScore.ndim}`
    );
  }

  if (yTrue.ndim === 2) {
    const rows = yTrue.shape[0] ?? 0;
    if ((yScore.shape[0] ?? 0) !== rows || (yTrue.shape[1] ?? 0) !== (yScore.shape[1] ?? 0)) {
      throw new ShapeError(
        `ndcgScore: yTrue and yScore must have same shape; got [${yTrue.shape}] vs [${yScore.shape}]`
      );
    }
    if (rows === 0) return 0;
    let sum = 0;
    for (let r = 0; r < rows; r++) {
      sum += ndcgSingle(extractRow2D(yTrue, r), extractRow2D(yScore, r), k);
    }
    return sum / rows;
  }

  const trueArr = extractArray1D(yTrue, "ndcgScore");
  const scoreArr = extractArray1D(yScore, "ndcgScore");
  if (trueArr.length !== scoreArr.length) {
    throw new ShapeError(
      `ndcgScore: yTrue and yScore must have same length; got ${trueArr.length} vs ${scoreArr.length}`
    );
  }
  return ndcgSingle(trueArr, scoreArr, k);
}

/**
 * Compute Mean Reciprocal Rank (MRR).
 *
 * @param yTrue - True binary relevance labels (1-D tensor, 0 or 1)
 * @param yScore - Predicted scores (1-D tensor)
 * @returns Reciprocal rank (1/rank of first relevant item)
 */
export function reciprocalRank(yTrue: Tensor, yScore: Tensor): number {
  const trueArr = extractArray1D(yTrue, "reciprocalRank");
  const scoreArr = extractArray1D(yScore, "reciprocalRank");
  if (trueArr.length !== scoreArr.length) {
    throw new ShapeError(
      `reciprocalRank: yTrue and yScore must have same length; got ${trueArr.length} vs ${scoreArr.length}`
    );
  }
  const n = trueArr.length;
  if (n === 0) return 0;

  // Sort by predicted scores descending
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => (scoreArr[b] ?? 0) - (scoreArr[a] ?? 0));

  for (let i = 0; i < n; i++) {
    if ((trueArr[indices[i]!] ?? 0) > 0) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

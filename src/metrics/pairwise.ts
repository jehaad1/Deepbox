/**
 * Pairwise distance metrics and ranking metrics.
 *
 * @module metrics/pairwise
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Metrics}
 */

import { InvalidParameterError, ShapeError } from "../core/errors";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { euclideanDistance, readFiniteFloat64 } from "./_internal";

function manhattanDistance(
  a: Float64Array,
  ia: number,
  b: Float64Array,
  ib: number,
  d: number
): number {
  let sum = 0;
  for (let f = 0; f < d; f++) sum += Math.abs((a[ia + f] as number) - (b[ib + f] as number));
  return sum;
}

type Matrix = { readonly data: Float64Array; readonly rows: number; readonly cols: number };

/** Read a finite 2D tensor into a dense row-major matrix. */
function readMatrix(t: Tensor, name: string, fn: string): Matrix {
  if (t.ndim !== 2) {
    throw new ShapeError(`${fn} requires 2D input; got ndim=${t.ndim}`);
  }
  return { data: readFiniteFloat64(t, name), rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0 };
}

/** Read X (and Y when given), checking that both have the same number of features. */
function readOperands(X: Tensor, Y: Tensor | undefined, fn: string): { x: Matrix; y: Matrix } {
  const x = readMatrix(X, "X", fn);
  if (Y === undefined) return { x, y: x };
  const y = readMatrix(Y, "Y", fn);
  if (x.cols !== y.cols) {
    throw new ShapeError(
      `${fn}: X and Y must have the same number of features; got ${x.cols} vs ${y.cols}`
    );
  }
  return { x, y };
}

/** Wrap a distance matrix buffer as a float64 tensor without copies. */
function wrapMatrix(data: Float64Array, rows: number, cols: number): Tensor {
  return TensorClass.fromTypedArray({ data, shape: [rows, cols], dtype: "float64", device: "cpu" });
}

/**
 * Fill a [n, m] distance matrix. Without Y the matrix is symmetric with a zero
 * diagonal, so only the upper triangle is evaluated.
 */
function distanceMatrix(
  x: Matrix,
  y: Matrix,
  symmetric: boolean,
  distance: (a: Float64Array, ia: number, b: Float64Array, ib: number, d: number) => number
): Tensor {
  const n = x.rows;
  const m = y.rows;
  const d = x.cols;
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const baseI = i * d;
    for (let j = symmetric ? i + 1 : 0; j < m; j++) {
      const dist = distance(x.data, baseI, y.data, j * d, d);
      out[i * m + j] = dist;
      if (symmetric) out[j * m + i] = dist;
    }
  }
  return wrapMatrix(out, n, m);
}

/**
 * Compute pairwise Euclidean distances between rows.
 *
 * Without `Y` the result is the symmetric [n, n] matrix of distances between the
 * rows of `X`; with `Y` it is the [n, m] matrix of distances from rows of `X` to
 * rows of `Y`.
 *
 * @param X - 2-D tensor of shape [n, d]
 * @param Y - Optional 2-D tensor of shape [m, d]
 * @returns Float64 tensor of shape [n, n] (or [n, m] with `Y`) containing distances
 *
 * @throws {ShapeError} If an input is not 2D or X and Y have different feature counts
 * @throws {DTypeError} If an input is a string tensor
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { pairwiseEuclidean } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * pairwiseEuclidean(tensor([[0, 0], [3, 4]])).toArray(); // [[0, 5], [5, 0]]
 * ```
 */
export function pairwiseEuclidean(X: Tensor, Y?: Tensor): Tensor {
  const { x, y } = readOperands(X, Y, "pairwiseEuclidean");
  return distanceMatrix(x, y, Y === undefined, euclideanDistance);
}

/**
 * Compute pairwise Manhattan (L1) distances between rows.
 *
 * Without `Y` the result is the symmetric [n, n] matrix of distances between the
 * rows of `X`; with `Y` it is the [n, m] matrix of distances from rows of `X` to
 * rows of `Y`.
 *
 * @param X - 2-D tensor of shape [n, d]
 * @param Y - Optional 2-D tensor of shape [m, d]
 * @returns Float64 tensor of shape [n, n] (or [n, m] with `Y`) containing L1 distances
 *
 * @throws {ShapeError} If an input is not 2D or X and Y have different feature counts
 * @throws {DTypeError} If an input is a string tensor
 * @throws {DataValidationError} If an input contains NaN or infinite values
 */
export function pairwiseManhattan(X: Tensor, Y?: Tensor): Tensor {
  const { x, y } = readOperands(X, Y, "pairwiseManhattan");
  return distanceMatrix(x, y, Y === undefined, manhattanDistance);
}

/** Scale each row to unit length in place; returns which rows had zero norm. */
function normalizeRows(m: Matrix): Uint8Array {
  const zeroRow = new Uint8Array(m.rows);
  const d = m.cols;
  for (let i = 0; i < m.rows; i++) {
    const base = i * d;
    // Scale by the largest magnitude first so squaring cannot overflow or underflow.
    let scale = 0;
    for (let k = 0; k < d; k++) {
      const a = Math.abs(m.data[base + k] as number);
      if (a > scale) scale = a;
    }
    if (scale === 0) {
      zeroRow[i] = 1;
      continue;
    }
    let sq = 0;
    for (let k = 0; k < d; k++) {
      const v = (m.data[base + k] as number) / scale;
      sq += v * v;
    }
    const inv = 1 / (scale * Math.sqrt(sq));
    for (let k = 0; k < d; k++) m.data[base + k] = (m.data[base + k] as number) * inv;
  }
  return zeroRow;
}

/**
 * Compute pairwise cosine distances between rows.
 *
 * cosine_distance = 1 - cosine_similarity, clipped to [0, 2].
 *
 * A row of all zeros has no direction: its similarity to every other row is taken
 * as 0, so its distance to them is 1 (as in scikit-learn). Without `Y` the diagonal
 * is exactly 0.
 *
 * @param X - 2-D tensor of shape [n, d]
 * @param Y - Optional 2-D tensor of shape [m, d]
 * @returns Float64 tensor of shape [n, n] (or [n, m] with `Y`) containing cosine distances
 *
 * @throws {ShapeError} If an input is not 2D or X and Y have different feature counts
 * @throws {DTypeError} If an input is a string tensor
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { pairwiseCosine } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * pairwiseCosine(tensor([[1, 0], [0, 1]])).toArray(); // [[0, 1], [1, 0]]
 * ```
 */
export function pairwiseCosine(X: Tensor, Y?: Tensor): Tensor {
  const { x: rawX, y: rawY } = readOperands(X, Y, "pairwiseCosine");
  const symmetric = Y === undefined;

  // readFiniteFloat64 returns fresh buffers, so normalizing in place is safe.
  const x: Matrix = rawX;
  const zeroX = normalizeRows(x);
  const y: Matrix = symmetric ? x : rawY;
  const zeroY = symmetric ? zeroX : normalizeRows(y);

  const n = x.rows;
  const m = y.rows;
  const d = x.cols;
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const baseI = i * d;
    for (let j = symmetric ? i + 1 : 0; j < m; j++) {
      const baseJ = j * d;
      let dist = 1;
      if (!zeroX[i] && !zeroY[j]) {
        let dot = 0;
        for (let k = 0; k < d; k++) {
          dot += (x.data[baseI + k] as number) * (y.data[baseJ + k] as number);
        }
        dist = Math.min(2, Math.max(0, 1 - dot));
      }
      out[i * m + j] = dist;
      if (symmetric) out[j * m + i] = dist;
    }
  }

  return wrapMatrix(out, n, m);
}

// ---- Ranking metrics ----

/** Discounted cumulative gain of the top `topK` positions of a ranking. */
function idealDcg(relevance: Float64Array, topK: number): number {
  const sorted = Float64Array.from(relevance).sort().reverse();
  let dcg = 0;
  for (let i = 0; i < topK; i++) dcg += (sorted[i] as number) * (1 / Math.log2(i + 2));
  return dcg;
}

/**
 * DCG of the ranking induced by `scores`. Documents with equal scores share the
 * average of their gains over the positions the tie group occupies, so the result
 * does not depend on the order of tied documents (scikit-learn's default).
 */
function tieAveragedDcg(relevance: Float64Array, scores: Float64Array, topK: number): number {
  const n = relevance.length;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => (scores[b] as number) - (scores[a] as number) || a - b);

  let dcg = 0;
  let start = 0;
  while (start < n) {
    const groupScore = scores[order[start] as number] as number;
    let end = start;
    let gainSum = 0;
    while (end < n && scores[order[end] as number] === groupScore) {
      gainSum += relevance[order[end] as number] as number;
      end++;
    }
    // Only positions below topK carry a discount.
    let discountSum = 0;
    for (let pos = start; pos < Math.min(end, topK); pos++) discountSum += 1 / Math.log2(pos + 2);
    dcg += (gainSum / (end - start)) * discountSum;
    start = end;
  }
  return dcg;
}

/** Single-query NDCG over dense arrays. Shared by the 1-D and 2-D paths. */
function ndcgSingle(relevance: Float64Array, scores: Float64Array, k: number | undefined): number {
  const n = relevance.length;
  if (n === 0) return 0;
  const topK = Math.min(k ?? n, n);

  const ideal = idealDcg(relevance, topK);
  if (ideal === 0) return 0;
  return tieAveragedDcg(relevance, scores, topK) / ideal;
}

/**
 * Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.
 *
 * Accepts either a single query (1-D tensors) or a batch of queries
 * (2-D tensors of shape `[nSamples, nLabels]`), in which case the mean
 * NDCG across samples is returned, matching scikit-learn's `ndcg_score`.
 * Gains are linear (the relevance itself). Documents with equal predicted
 * scores share their gain, so ties do not depend on input order. Queries
 * without any relevant document score 0.
 *
 * @param yTrue - True non-negative relevance scores (1-D or 2-D tensor)
 * @param yScore - Predicted scores (1-D or 2-D tensor, same shape as yTrue)
 * @param k - Number of top results to consider (default: all)
 * @returns NDCG score in [0, 1]
 *
 * @throws {ShapeError} If the inputs are not both 1D or both 2D, or their shapes differ
 * @throws {InvalidParameterError} If k is not a positive integer or yTrue has a negative value
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { ndcgScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * ndcgScore(tensor([[0, 1, 0, 2]]), tensor([[0.5, 0.5, 0.2, 0.5]]), 2); // 0.6199062332840657
 * ```
 */
export function ndcgScore(yTrue: Tensor, yScore: Tensor, k?: number): number {
  if (yTrue.ndim !== yScore.ndim) {
    throw new ShapeError(
      `ndcgScore: yTrue and yScore must have same ndim; got ${yTrue.ndim} vs ${yScore.ndim}`
    );
  }
  if (yTrue.ndim !== 1 && yTrue.ndim !== 2) {
    throw new ShapeError(`ndcgScore requires 1D or 2D input; got ndim=${yTrue.ndim}`);
  }
  if (k !== undefined && (!Number.isInteger(k) || k < 1)) {
    throw new InvalidParameterError("k must be a positive integer", "k", k);
  }

  if (yTrue.ndim === 2) {
    const rows = yTrue.shape[0] ?? 0;
    const cols = yTrue.shape[1] ?? 0;
    if ((yScore.shape[0] ?? 0) !== rows || cols !== (yScore.shape[1] ?? 0)) {
      throw new ShapeError(
        `ndcgScore: yTrue and yScore must have same shape; got [${yTrue.shape}] vs [${yScore.shape}]`
      );
    }
  } else if (yTrue.size !== yScore.size) {
    throw new ShapeError(
      `ndcgScore: yTrue and yScore must have same length; got ${yTrue.size} vs ${yScore.size}`
    );
  }

  const relevance = readFiniteFloat64(yTrue, "yTrue");
  const scores = readFiniteFloat64(yScore, "yScore");
  for (let i = 0; i < relevance.length; i++) {
    const v = relevance[i] as number;
    if (v < 0) {
      throw new InvalidParameterError("ndcgScore requires non-negative yTrue values", "yTrue", v);
    }
  }

  if (yTrue.ndim === 1) return ndcgSingle(relevance, scores, k);

  const rows = yTrue.shape[0] ?? 0;
  const cols = yTrue.shape[1] ?? 0;
  if (rows === 0) return 0;
  let sum = 0;
  for (let r = 0; r < rows; r++) {
    sum += ndcgSingle(
      relevance.subarray(r * cols, (r + 1) * cols),
      scores.subarray(r * cols, (r + 1) * cols),
      k
    );
  }
  return sum / rows;
}

/** Reciprocal rank of the best-ranked relevant item of one query. */
function reciprocalRankSingle(relevance: Float64Array, scores: Float64Array): number {
  const n = relevance.length;

  // The best relevant item is the one with the highest score; the earliest wins a tie.
  let best = -1;
  for (let i = 0; i < n; i++) {
    if (
      (relevance[i] as number) > 0 &&
      (best < 0 || (scores[i] as number) > (scores[best] as number))
    ) {
      best = i;
    }
  }
  if (best < 0) return 0;

  const bestScore = scores[best] as number;
  let rank = 1;
  for (let i = 0; i < n; i++) {
    const s = scores[i] as number;
    if (s > bestScore || (s === bestScore && i < best)) rank++;
  }
  return 1 / rank;
}

/**
 * Compute the reciprocal rank (RR) of the first relevant item.
 *
 * Items are ranked by decreasing score; equal scores keep their input order. For a
 * batch of queries (2-D tensors of shape `[nQueries, nItems]`) the mean over queries
 * is returned, which is the Mean Reciprocal Rank (MRR). Queries without a relevant
 * item score 0.
 *
 * @param yTrue - Relevance labels (1-D or 2-D tensor); any positive value is relevant
 * @param yScore - Predicted scores (same shape as yTrue)
 * @returns Reciprocal rank in [0, 1] (mean reciprocal rank for 2-D input)
 *
 * @throws {ShapeError} If the inputs are not both 1D or both 2D, or their shapes differ
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { reciprocalRank } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * reciprocalRank(tensor([0, 0, 1]), tensor([0.3, 0.2, 0.1])); // 1 / 3
 * ```
 */
export function reciprocalRank(yTrue: Tensor, yScore: Tensor): number {
  if (yTrue.ndim !== yScore.ndim) {
    throw new ShapeError(
      `reciprocalRank: yTrue and yScore must have same ndim; got ${yTrue.ndim} vs ${yScore.ndim}`
    );
  }
  if (yTrue.ndim !== 1 && yTrue.ndim !== 2) {
    throw new ShapeError(`reciprocalRank requires 1D or 2D input; got ndim=${yTrue.ndim}`);
  }
  if (yTrue.ndim === 1) {
    if (yTrue.size !== yScore.size) {
      throw new ShapeError(
        `reciprocalRank: yTrue and yScore must have same length; got ${yTrue.size} vs ${yScore.size}`
      );
    }
  } else if (
    (yTrue.shape[0] ?? 0) !== (yScore.shape[0] ?? 0) ||
    (yTrue.shape[1] ?? 0) !== (yScore.shape[1] ?? 0)
  ) {
    throw new ShapeError(
      `reciprocalRank: yTrue and yScore must have same shape; got [${yTrue.shape}] vs [${yScore.shape}]`
    );
  }

  const relevance = readFiniteFloat64(yTrue, "yTrue");
  const scores = readFiniteFloat64(yScore, "yScore");
  if (yTrue.ndim === 1) return reciprocalRankSingle(relevance, scores);

  const rows = yTrue.shape[0] ?? 0;
  const cols = yTrue.shape[1] ?? 0;
  if (rows === 0) return 0;
  let sum = 0;
  for (let r = 0; r < rows; r++) {
    sum += reciprocalRankSingle(
      relevance.subarray(r * cols, (r + 1) * cols),
      scores.subarray(r * cols, (r + 1) * cols)
    );
  }
  return sum / rows;
}

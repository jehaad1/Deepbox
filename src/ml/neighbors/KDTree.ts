/**
 * KD-Tree for exact nearest-neighbor and radius queries in Euclidean space.
 *
 * Building the tree costs O(n log^2 n); a query on low-dimensional data takes
 * roughly O(log n) instead of the O(n) of a brute-force scan.
 *
 * @module ml/neighbors/KDTree
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";

interface KDNode {
  /** Row index (into the data array) of the point stored at this node. */
  readonly index: number;
  readonly splitDim: number;
  readonly left: KDNode | null;
  readonly right: KDNode | null;
}

/** Bounded list of the best candidates so far, ordered by (distance, index). */
interface Candidates {
  readonly k: number;
  readonly dist: number[];
  readonly index: number[];
}

/** Offer a candidate to the bounded list, keeping it ordered by (distance, index). */
function offer(best: Candidates, dist: number, index: number): void {
  const { dist: bd, index: bi, k } = best;
  const n = bd.length;
  if (n === k) {
    const worstDist = bd[k - 1] as number;
    if (dist > worstDist || (dist === worstDist && index > (bi[k - 1] as number))) return;
  }
  let pos = n;
  while (pos > 0) {
    const prev = bd[pos - 1] as number;
    if (prev < dist || (prev === dist && (bi[pos - 1] as number) < index)) break;
    pos--;
  }
  bd.splice(pos, 0, dist);
  bi.splice(pos, 0, index);
  if (bd.length > k) {
    bd.pop();
    bi.pop();
  }
}

/**
 * KD-Tree over a row-major matrix of points.
 *
 * The tree keeps a reference to `data` (it is not copied), so the array must
 * not be modified while the tree is in use. All distances are Euclidean.
 * Ties in distance are ordered by ascending point index, so results are
 * deterministic.
 *
 * @example
 * ```ts
 * import { KDTree } from 'deepbox/ml';
 *
 * // Three 2-D points stored row by row
 * const data = new Float64Array([0, 0, 1, 0, 5, 5]);
 * const tree = new KDTree(data, 3, 2);
 * const { distances, indices } = tree.query(new Float64Array([0.9, 0.1]), 2);
 * // indices -> [1, 0]
 * ```
 */
export class KDTree {
  /** Number of indexed points. */
  readonly nSamples: number;
  /** Number of coordinates per point. */
  readonly nDims: number;

  private readonly data: Float64Array;
  private readonly root: KDNode | null;

  /**
   * @param data - Row-major points, at least `nSamples * nDims` values
   * @param nSamples - Number of points (rows) to index
   * @param nDims - Number of coordinates per point (>= 1)
   * @throws {InvalidParameterError} If `nSamples` or `nDims` is not a valid integer
   * @throws {ShapeError} If `data` holds fewer than `nSamples * nDims` values
   * @throws {DataValidationError} If an indexed value is NaN or infinite
   */
  constructor(data: Float64Array, nSamples: number, nDims: number) {
    if (!Number.isInteger(nSamples) || nSamples < 0) {
      throw new InvalidParameterError(
        `nSamples must be a non-negative integer; received ${nSamples}`,
        "nSamples",
        nSamples
      );
    }
    if (!Number.isInteger(nDims) || nDims < 1) {
      throw new InvalidParameterError(
        `nDims must be an integer >= 1; received ${nDims}`,
        "nDims",
        nDims
      );
    }
    const total = nSamples * nDims;
    if (data.length < total) {
      throw new ShapeError(
        `data has ${data.length} values but nSamples * nDims = ${total} are required`
      );
    }
    for (let i = 0; i < total; i++) {
      if (!Number.isFinite(data[i])) {
        throw new DataValidationError("data contains non-finite values (NaN or Inf)");
      }
    }

    this.nSamples = nSamples;
    this.nDims = nDims;
    this.data = data;

    const order = new Uint32Array(nSamples);
    for (let i = 0; i < nSamples; i++) order[i] = i;
    this.root = this.buildTree(order, 0, nSamples);
  }

  /**
   * Find the `k` nearest neighbors of a query point.
   *
   * If the tree holds fewer than `k` points, all of them are returned.
   *
   * @param point - Query point with `nDims` coordinates
   * @param k - Number of neighbors (integer >= 1)
   * @returns Euclidean distances and point indices, nearest first
   * @throws {InvalidParameterError} If `k` is not an integer >= 1
   * @throws {ShapeError} If `point` does not have `nDims` coordinates
   * @throws {DataValidationError} If `point` contains NaN or Inf
   */
  query(point: Float64Array, k: number): { distances: number[]; indices: number[] } {
    this.checkK(k);
    this.checkPoint(point);
    const best: Candidates = { k, dist: [], index: [] };
    this.queryRecursive(this.root, point, best);
    return { distances: best.dist.map(Math.sqrt), indices: best.index };
  }

  /**
   * Find all points whose distance to `point` is at most `radius`.
   *
   * @param point - Query point with `nDims` coordinates
   * @param radius - Search radius (>= 0)
   * @returns Euclidean distances and point indices, nearest first
   * @throws {InvalidParameterError} If `radius` is negative or NaN
   * @throws {ShapeError} If `point` does not have `nDims` coordinates
   * @throws {DataValidationError} If `point` contains NaN or Inf
   */
  queryRadius(point: Float64Array, radius: number): { distances: number[]; indices: number[] } {
    if (typeof radius !== "number" || Number.isNaN(radius) || radius < 0) {
      throw new InvalidParameterError(
        `radius must be a number >= 0; received ${radius}`,
        "radius",
        radius
      );
    }
    this.checkPoint(point);
    const hits: Array<{ dist: number; index: number }> = [];
    this.radiusRecursive(this.root, point, radius * radius, hits);
    hits.sort((a, b) => a.dist - b.dist || a.index - b.index);
    return {
      distances: hits.map((h) => Math.sqrt(h.dist)),
      indices: hits.map((h) => h.index),
    };
  }

  /**
   * Find the `k` nearest neighbors for several query points.
   *
   * Row `i` of the result holds the neighbors of query `i`. If the tree holds
   * fewer than `k` points, the unused trailing slots contain index `-1` and
   * distance `Infinity`.
   *
   * @param points - Row-major query points, `nPoints * nDims` values
   * @param nPoints - Number of query points
   * @param k - Number of neighbors (integer >= 1)
   * @returns Flat `(nPoints, k)` distance and index arrays
   * @throws {InvalidParameterError} If `nPoints` or `k` is invalid
   * @throws {ShapeError} If `points` holds fewer than `nPoints * nDims` values
   * @throws {DataValidationError} If `points` contains NaN or Inf
   */
  queryBatch(
    points: Float64Array,
    nPoints: number,
    k: number
  ): { distances: Float64Array; indices: Int32Array } {
    this.checkK(k);
    if (!Number.isInteger(nPoints) || nPoints < 0) {
      throw new InvalidParameterError(
        `nPoints must be a non-negative integer; received ${nPoints}`,
        "nPoints",
        nPoints
      );
    }
    const nDims = this.nDims;
    if (points.length < nPoints * nDims) {
      throw new ShapeError(
        `points has ${points.length} values but nPoints * nDims = ${nPoints * nDims} are required`
      );
    }

    const distances = new Float64Array(nPoints * k).fill(Number.POSITIVE_INFINITY);
    const indices = new Int32Array(nPoints * k).fill(-1);

    for (let i = 0; i < nPoints; i++) {
      const point = points.subarray(i * nDims, (i + 1) * nDims);
      const result = this.query(point, k);
      for (let j = 0; j < result.indices.length; j++) {
        distances[i * k + j] = result.distances[j] as number;
        indices[i * k + j] = result.indices[j] as number;
      }
    }

    return { distances, indices };
  }

  private checkK(k: number): void {
    if (!Number.isInteger(k) || k < 1) {
      throw new InvalidParameterError(`k must be an integer >= 1; received ${k}`, "k", k);
    }
  }

  private checkPoint(point: Float64Array): void {
    if (point.length !== this.nDims) {
      throw new ShapeError(
        `query point must have ${this.nDims} coordinates; received ${point.length}`
      );
    }
    for (let d = 0; d < this.nDims; d++) {
      if (!Number.isFinite(point[d])) {
        throw new DataValidationError("query point contains non-finite values (NaN or Inf)");
      }
    }
  }

  /** Build the subtree over `order[lo, hi)`, splitting on the widest dimension. */
  private buildTree(order: Uint32Array, lo: number, hi: number): KDNode | null {
    if (hi <= lo) return null;
    const { data, nDims } = this;

    let dim = 0;
    if (hi - lo > 1) {
      let bestSpread = -1;
      for (let d = 0; d < nDims; d++) {
        let minVal = Number.POSITIVE_INFINITY;
        let maxVal = Number.NEGATIVE_INFINITY;
        for (let p = lo; p < hi; p++) {
          const v = data[(order[p] as number) * nDims + d] as number;
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
        const spread = maxVal - minVal;
        if (spread > bestSpread) {
          bestSpread = spread;
          dim = d;
        }
      }
    }

    order
      .subarray(lo, hi)
      .sort(
        (a, b) => (data[a * nDims + dim] as number) - (data[b * nDims + dim] as number) || a - b
      );

    const mid = lo + ((hi - lo) >> 1);
    return {
      index: order[mid] as number,
      splitDim: dim,
      left: this.buildTree(order, lo, mid),
      right: this.buildTree(order, mid + 1, hi),
    };
  }

  private squaredDist(query: Float64Array, row: number): number {
    const { data, nDims } = this;
    const base = row * nDims;
    let sum = 0;
    for (let d = 0; d < nDims; d++) {
      const diff = (query[d] as number) - (data[base + d] as number);
      sum += diff * diff;
    }
    return sum;
  }

  private queryRecursive(node: KDNode | null, query: Float64Array, best: Candidates): void {
    if (node === null) return;

    const dist = this.squaredDist(query, node.index);
    offer(best, dist, node.index);
    const { dist: bd, k } = best;

    const dim = node.splitDim;
    const diff = (query[dim] as number) - (this.data[node.index * this.nDims + dim] as number);
    const diff2 = diff * diff;

    const first = diff <= 0 ? node.left : node.right;
    const second = diff <= 0 ? node.right : node.left;

    this.queryRecursive(first, query, best);

    // The far side can only hold a closer (or equally close, lower-index) point
    // when the splitting plane is not farther than the current k-th distance.
    if (bd.length < k || diff2 <= (bd[k - 1] as number)) {
      this.queryRecursive(second, query, best);
    }
  }

  private radiusRecursive(
    node: KDNode | null,
    query: Float64Array,
    r2: number,
    hits: Array<{ dist: number; index: number }>
  ): void {
    if (node === null) return;

    const dist = this.squaredDist(query, node.index);
    if (dist <= r2) {
      hits.push({ dist, index: node.index });
    }

    const dim = node.splitDim;
    const diff = (query[dim] as number) - (this.data[node.index * this.nDims + dim] as number);
    const diff2 = diff * diff;

    const first = diff <= 0 ? node.left : node.right;
    const second = diff <= 0 ? node.right : node.left;

    this.radiusRecursive(first, query, r2, hits);

    if (diff2 <= r2) {
      this.radiusRecursive(second, query, r2, hits);
    }
  }
}

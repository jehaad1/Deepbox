/**
 * Ball Tree for exact nearest-neighbor and radius queries in Euclidean space.
 *
 * Partitions the data into nested hyper-spheres (balls). Pruning is based on
 * the distance to a ball rather than on axis-aligned splits, so it keeps
 * working in moderate to high dimensions where KD-Trees degrade.
 *
 * @module ml/neighbors/BallTree
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";

interface BallNode {
  readonly center: Float64Array;
  /** Largest distance from `center` to any point in the ball. */
  readonly radius: number;
  /** The ball covers `order[start, end)`. */
  readonly start: number;
  readonly end: number;
  readonly left: BallNode | null;
  readonly right: BallNode | null;
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

/** Relative slack that keeps floating-point rounding from pruning a ball that holds a hit. */
const PRUNE_SLACK = 1e-12;

/**
 * Ball Tree over a row-major matrix of points.
 *
 * The tree keeps a reference to `data` (it is not copied), so the array must
 * not be modified while the tree is in use. All distances are Euclidean.
 * Ties in distance are ordered by ascending point index, so results are
 * deterministic.
 *
 * @example
 * ```ts
 * import { BallTree } from 'deepbox/ml';
 *
 * // Three 2-D points stored row by row
 * const data = new Float64Array([0, 0, 1, 0, 5, 5]);
 * const tree = new BallTree(data, 3, 2);
 * const { distances, indices } = tree.query(new Float64Array([0.9, 0.1]), 2);
 * // indices -> [1, 0]
 * ```
 */
export class BallTree {
  /** Number of indexed points. */
  readonly nSamples: number;
  /** Number of coordinates per point. */
  readonly nDims: number;

  private readonly data: Float64Array;
  private readonly leafSize: number;
  /** Point indices, permuted so that every node covers a contiguous range. */
  private readonly order: Uint32Array;
  private readonly root: BallNode | null;

  /**
   * @param data - Row-major points, at least `nSamples * nDims` values
   * @param nSamples - Number of points (rows) to index
   * @param nDims - Number of coordinates per point (>= 1)
   * @param leafSize - Largest number of points stored in a leaf (integer >= 1, default 40)
   * @throws {InvalidParameterError} If `nSamples`, `nDims` or `leafSize` is not a valid integer
   * @throws {ShapeError} If `data` holds fewer than `nSamples * nDims` values
   * @throws {DataValidationError} If an indexed value is NaN or infinite
   */
  constructor(data: Float64Array, nSamples: number, nDims: number, leafSize = 40) {
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
    if (!Number.isInteger(leafSize) || leafSize < 1) {
      throw new InvalidParameterError(
        `leafSize must be an integer >= 1; received ${leafSize}`,
        "leafSize",
        leafSize
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
    this.leafSize = leafSize;

    this.order = new Uint32Array(nSamples);
    for (let i = 0; i < nSamples; i++) this.order[i] = i;
    this.root = this.buildTree(0, nSamples);
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
    this.radiusRecursive(this.root, point, radius, radius * radius, hits);
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

  /** Build the ball over `order[start, end)`, splitting at the median of the widest dimension. */
  private buildTree(start: number, end: number): BallNode | null {
    if (end <= start) return null;
    const { data, nDims, order } = this;

    const center = new Float64Array(nDims);
    for (let p = start; p < end; p++) {
      const base = (order[p] as number) * nDims;
      for (let d = 0; d < nDims; d++) {
        center[d] = (center[d] as number) + (data[base + d] as number);
      }
    }
    const count = end - start;
    for (let d = 0; d < nDims; d++) center[d] = (center[d] as number) / count;

    let maxDist2 = 0;
    for (let p = start; p < end; p++) {
      const base = (order[p] as number) * nDims;
      let dist2 = 0;
      for (let d = 0; d < nDims; d++) {
        const diff = (data[base + d] as number) - (center[d] as number);
        dist2 += diff * diff;
      }
      if (dist2 > maxDist2) maxDist2 = dist2;
    }
    const radius = Math.sqrt(maxDist2);

    if (count <= this.leafSize) {
      return { center, radius, start, end, left: null, right: null };
    }

    let bestDim = 0;
    let bestSpread = -1;
    for (let d = 0; d < nDims; d++) {
      let minVal = Number.POSITIVE_INFINITY;
      let maxVal = Number.NEGATIVE_INFINITY;
      for (let p = start; p < end; p++) {
        const v = data[(order[p] as number) * nDims + d] as number;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      const spread = maxVal - minVal;
      if (spread > bestSpread) {
        bestSpread = spread;
        bestDim = d;
      }
    }

    order
      .subarray(start, end)
      .sort(
        (a, b) =>
          (data[a * nDims + bestDim] as number) - (data[b * nDims + bestDim] as number) || a - b
      );

    const mid = start + (count >> 1);
    return {
      center,
      radius,
      start,
      end,
      left: this.buildTree(start, mid),
      right: this.buildTree(mid, end),
    };
  }

  private squaredDistToRow(query: Float64Array, row: number): number {
    const nDims = this.nDims;
    const base = row * nDims;
    let sum = 0;
    for (let d = 0; d < nDims; d++) {
      const diff = (query[d] as number) - (this.data[base + d] as number);
      sum += diff * diff;
    }
    return sum;
  }

  private squaredDistToCenter(query: Float64Array, center: Float64Array): number {
    let sum = 0;
    for (let d = 0; d < this.nDims; d++) {
      const diff = (query[d] as number) - (center[d] as number);
      sum += diff * diff;
    }
    return sum;
  }

  /** Lower bound on the distance from `query` to any point inside `node`. */
  private lowerBound(query: Float64Array, node: BallNode): number {
    const toCenter = Math.sqrt(this.squaredDistToCenter(query, node.center));
    return toCenter * (1 - PRUNE_SLACK) - node.radius;
  }

  private queryRecursive(node: BallNode | null, query: Float64Array, best: Candidates): void {
    if (node === null) return;

    // Skip the ball when even its closest possible point is farther than the
    // current k-th neighbor. A negative bound means the query is inside the ball.
    const bound = this.lowerBound(query, node);
    if (
      bound > 0 &&
      best.dist.length >= best.k &&
      bound * bound > (best.dist[best.k - 1] as number)
    ) {
      return;
    }

    if (node.left === null || node.right === null) {
      for (let p = node.start; p < node.end; p++) {
        const idx = this.order[p] as number;
        offer(best, this.squaredDistToRow(query, idx), idx);
      }
      return;
    }

    // Visit the child whose center is closer first.
    const leftDist = this.squaredDistToCenter(query, node.left.center);
    const rightDist = this.squaredDistToCenter(query, node.right.center);
    if (leftDist <= rightDist) {
      this.queryRecursive(node.left, query, best);
      this.queryRecursive(node.right, query, best);
    } else {
      this.queryRecursive(node.right, query, best);
      this.queryRecursive(node.left, query, best);
    }
  }

  private radiusRecursive(
    node: BallNode | null,
    query: Float64Array,
    radius: number,
    r2: number,
    hits: Array<{ dist: number; index: number }>
  ): void {
    if (node === null) return;

    // The slack in lowerBound is relative to the distance to the center, so the
    // comparison is done on distances here rather than on squares.
    if (this.lowerBound(query, node) > radius) return;

    if (node.left === null || node.right === null) {
      for (let p = node.start; p < node.end; p++) {
        const idx = this.order[p] as number;
        const dist = this.squaredDistToRow(query, idx);
        if (dist <= r2) hits.push({ dist, index: idx });
      }
      return;
    }

    this.radiusRecursive(node.left, query, radius, r2, hits);
    this.radiusRecursive(node.right, query, radius, r2, hits);
  }
}

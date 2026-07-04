/**
 * Ball Tree for efficient spatial queries.
 *
 * Partitions data into nested hyper-spheres (balls) for fast
 * nearest-neighbor and radius searches. Works well in moderate
 * to high dimensions where KD-Trees degrade.
 *
 * @module ml/neighbors/BallTree
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

interface BallNode {
  center: Float64Array;
  radius: number;
  indices: number[];
  left: BallNode | null;
  right: BallNode | null;
}

interface HeapItem {
  dist: number;
  index: number;
}

export class BallTree {
  private root: BallNode | null = null;
  private readonly nDims: number;
  private readonly data: Float64Array;
  private readonly leafSize: number;

  constructor(data: Float64Array, nSamples: number, nDims: number, leafSize = 40) {
    this.nDims = nDims;
    this.data = data;
    this.leafSize = leafSize;

    const indices: number[] = [];
    for (let i = 0; i < nSamples; i++) indices.push(i);
    this.root = this.buildTree(indices);
  }

  /**
   * Find k nearest neighbors for a query point.
   * Returns arrays of [distances, indices] sorted by distance.
   */
  query(point: Float64Array, k: number): { distances: number[]; indices: number[] } {
    const heap: HeapItem[] = [];
    this.queryRecursive(this.root, point, k, heap);

    heap.sort((a, b) => a.dist - b.dist);

    return {
      distances: heap.map((h) => Math.sqrt(h.dist)),
      indices: heap.map((h) => h.index),
    };
  }

  /**
   * Find all points within a given radius.
   * Returns arrays of [distances, indices].
   */
  queryRadius(point: Float64Array, radius: number): { distances: number[]; indices: number[] } {
    const results: HeapItem[] = [];
    const r2 = radius * radius;
    this.radiusRecursive(this.root, point, r2, results);

    results.sort((a, b) => a.dist - b.dist);

    return {
      distances: results.map((h) => Math.sqrt(h.dist)),
      indices: results.map((h) => h.index),
    };
  }

  /**
   * Batch query: find k nearest neighbors for multiple points.
   */
  queryBatch(
    points: Float64Array,
    nPoints: number,
    k: number
  ): { distances: Float64Array; indices: Int32Array } {
    const distances = new Float64Array(nPoints * k);
    const indices = new Int32Array(nPoints * k).fill(-1);

    for (let i = 0; i < nPoints; i++) {
      const point = new Float64Array(this.nDims);
      for (let d = 0; d < this.nDims; d++) {
        point[d] = points[i * this.nDims + d] ?? 0;
      }
      const result = this.query(point, k);
      for (let j = 0; j < Math.min(k, result.distances.length); j++) {
        distances[i * k + j] = result.distances[j]!;
        indices[i * k + j] = result.indices[j]!;
      }
    }

    return { distances, indices };
  }

  private buildTree(indices: number[]): BallNode | null {
    if (indices.length === 0) return null;

    const center = this.computeCenter(indices);
    const radius = this.computeRadius(indices, center);

    if (indices.length <= this.leafSize) {
      return { center, radius, indices, left: null, right: null };
    }

    // Find the dimension with greatest spread
    let bestDim = 0;
    let bestSpread = -1;
    for (let d = 0; d < this.nDims; d++) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const idx of indices) {
        const val = this.data[idx * this.nDims + d] ?? 0;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
      const spread = maxVal - minVal;
      if (spread > bestSpread) {
        bestSpread = spread;
        bestDim = d;
      }
    }

    // Sort by that dimension and split at median
    const sorted = [...indices].sort((a, b) => {
      const va = this.data[a * this.nDims + bestDim] ?? 0;
      const vb = this.data[b * this.nDims + bestDim] ?? 0;
      return va - vb;
    });

    const mid = Math.floor(sorted.length / 2);
    const leftIndices = sorted.slice(0, mid);
    const rightIndices = sorted.slice(mid);

    return {
      center,
      radius,
      indices,
      left: this.buildTree(leftIndices),
      right: this.buildTree(rightIndices),
    };
  }

  private computeCenter(indices: number[]): Float64Array {
    const center = new Float64Array(this.nDims);
    for (const idx of indices) {
      for (let d = 0; d < this.nDims; d++) {
        center[d] = (center[d] ?? 0) + (this.data[idx * this.nDims + d] ?? 0);
      }
    }
    for (let d = 0; d < this.nDims; d++) {
      center[d] = (center[d] ?? 0) / indices.length;
    }
    return center;
  }

  private computeRadius(indices: number[], center: Float64Array): number {
    let maxDist2 = 0;
    for (const idx of indices) {
      let dist2 = 0;
      for (let d = 0; d < this.nDims; d++) {
        const diff = (this.data[idx * this.nDims + d] ?? 0) - (center[d] ?? 0);
        dist2 += diff * diff;
      }
      if (dist2 > maxDist2) maxDist2 = dist2;
    }
    return Math.sqrt(maxDist2);
  }

  private queryRecursive(
    node: BallNode | null,
    query: Float64Array,
    k: number,
    heap: HeapItem[]
  ): void {
    if (node === null) return;

    // Prune: if the closest point in this ball is farther than
    // the worst neighbor we already have, skip
    const distToCenter = Math.sqrt(this.squaredDist(query, node.center));
    const closestPossible = distToCenter - node.radius;
    // Only prune when the query lies OUTSIDE the ball (closestPossible > 0).
    // If it's inside, closestPossible is negative and squaring it would
    // produce a large positive bound that wrongly prunes the containing ball.
    if (
      closestPossible > 0 &&
      heap.length >= k &&
      closestPossible * closestPossible >= heap[0]!.dist
    ) {
      return;
    }

    // Leaf node: check all points
    if (node.left === null && node.right === null) {
      for (const idx of node.indices) {
        const point = new Float64Array(this.nDims);
        for (let d = 0; d < this.nDims; d++) {
          point[d] = this.data[idx * this.nDims + d] ?? 0;
        }
        const dist = this.squaredDist(query, point);

        if (heap.length < k) {
          heap.push({ dist, index: idx });
          heap.sort((a, b) => b.dist - a.dist);
        } else if (dist < heap[0]!.dist) {
          heap[0] = { dist, index: idx };
          heap.sort((a, b) => b.dist - a.dist);
        }
      }
      return;
    }

    // Determine which child to visit first (closer center)
    const leftDist = node.left ? this.squaredDist(query, node.left.center) : Infinity;
    const rightDist = node.right ? this.squaredDist(query, node.right.center) : Infinity;

    const [first, second] =
      leftDist <= rightDist ? [node.left, node.right] : [node.right, node.left];

    this.queryRecursive(first, query, k, heap);
    this.queryRecursive(second, query, k, heap);
  }

  private radiusRecursive(
    node: BallNode | null,
    query: Float64Array,
    r2: number,
    results: HeapItem[]
  ): void {
    if (node === null) return;

    // Prune: if the closest point in this ball is outside the radius, skip
    const distToCenter = Math.sqrt(this.squaredDist(query, node.center));
    const closestPossible = distToCenter - node.radius;
    if (closestPossible > 0 && closestPossible * closestPossible > r2) {
      return;
    }

    // Leaf node: check all points
    if (node.left === null && node.right === null) {
      for (const idx of node.indices) {
        const point = new Float64Array(this.nDims);
        for (let d = 0; d < this.nDims; d++) {
          point[d] = this.data[idx * this.nDims + d] ?? 0;
        }
        const dist = this.squaredDist(query, point);
        if (dist <= r2) {
          results.push({ dist, index: idx });
        }
      }
      return;
    }

    this.radiusRecursive(node.left, query, r2, results);
    this.radiusRecursive(node.right, query, r2, results);
  }

  private squaredDist(a: Float64Array, b: Float64Array): number {
    let sum = 0;
    for (let d = 0; d < this.nDims; d++) {
      const diff = (a[d] ?? 0) - (b[d] ?? 0);
      sum += diff * diff;
    }
    return sum;
  }
}

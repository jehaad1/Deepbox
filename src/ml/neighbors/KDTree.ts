/**
 * KD-Tree for efficient spatial queries.
 *
 * Provides O(n log n) nearest neighbor search and radius queries
 * instead of brute-force O(n²).
 *
 * @module ml/neighbors/KDTree
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

interface KDNode {
  point: Float64Array;
  index: number;
  splitDim: number;
  left: KDNode | null;
  right: KDNode | null;
}

interface HeapItem {
  dist: number;
  index: number;
}

export class KDTree {
  private root: KDNode | null = null;
  private readonly nDims: number;
  private readonly data: Float64Array;

  constructor(data: Float64Array, nSamples: number, nDims: number) {
    this.nDims = nDims;
    this.data = data;

    // Build tree
    const indices: number[] = [];
    for (let i = 0; i < nSamples; i++) indices.push(i);
    this.root = this.buildTree(indices, 0);
  }

  /**
   * Find k nearest neighbors for a query point.
   * Returns arrays of [distances, indices] sorted by distance.
   */
  query(point: Float64Array, k: number): { distances: number[]; indices: number[] } {
    const heap: HeapItem[] = [];

    this.queryRecursive(this.root, point, k, heap);

    // Sort by distance ascending
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

  private buildTree(indices: number[], depth: number): KDNode | null {
    if (indices.length === 0) return null;

    const dim = depth % this.nDims;

    // Sort by the current dimension
    indices.sort((a, b) => {
      const va = this.data[a * this.nDims + dim] ?? 0;
      const vb = this.data[b * this.nDims + dim] ?? 0;
      return va - vb;
    });

    const mid = Math.floor(indices.length / 2);
    const midIdx = indices[mid]!;

    const point = new Float64Array(this.nDims);
    for (let d = 0; d < this.nDims; d++) {
      point[d] = this.data[midIdx * this.nDims + d] ?? 0;
    }

    return {
      point,
      index: midIdx,
      splitDim: dim,
      left: this.buildTree(indices.slice(0, mid), depth + 1),
      right: this.buildTree(indices.slice(mid + 1), depth + 1),
    };
  }

  private queryRecursive(
    node: KDNode | null,
    query: Float64Array,
    k: number,
    heap: HeapItem[]
  ): void {
    if (node === null) return;

    const dist = this.squaredDist(query, node.point);

    if (heap.length < k) {
      heap.push({ dist, index: node.index });
      // Keep max-heap property (largest distance first for easy replacement)
      heap.sort((a, b) => b.dist - a.dist);
    } else if (dist < heap[0]!.dist) {
      heap[0] = { dist, index: node.index };
      heap.sort((a, b) => b.dist - a.dist);
    }

    const dim = node.splitDim;
    const diff = (query[dim] ?? 0) - (node.point[dim] ?? 0);
    const diff2 = diff * diff;

    // Visit the closer subtree first
    const first = diff <= 0 ? node.left : node.right;
    const second = diff <= 0 ? node.right : node.left;

    this.queryRecursive(first, query, k, heap);

    // Check if we need to visit the other subtree
    if (heap.length < k || diff2 < heap[0]!.dist) {
      this.queryRecursive(second, query, k, heap);
    }
  }

  private radiusRecursive(
    node: KDNode | null,
    query: Float64Array,
    r2: number,
    results: HeapItem[]
  ): void {
    if (node === null) return;

    const dist = this.squaredDist(query, node.point);
    if (dist <= r2) {
      results.push({ dist, index: node.index });
    }

    const dim = node.splitDim;
    const diff = (query[dim] ?? 0) - (node.point[dim] ?? 0);
    const diff2 = diff * diff;

    const first = diff <= 0 ? node.left : node.right;
    const second = diff <= 0 ? node.right : node.left;

    this.radiusRecursive(first, query, r2, results);

    if (diff2 <= r2) {
      this.radiusRecursive(second, query, r2, results);
    }
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

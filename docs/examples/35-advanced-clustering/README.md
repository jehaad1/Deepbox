# Advanced Clustering

> **View online:** https://deepbox.dev/examples/35-advanced-clustering

Runs eight clustering algorithms on synthetic data and scores them with the silhouette score and the adjusted Rand index. Seven run on well-separated blobs. Spectral clustering runs on two interleaved moons, a non-convex shape.

## Deepbox Modules Used

| Module             | Features Used                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ml`       | AgglomerativeClustering, GaussianMixture, SpectralClustering, OPTICS, MiniBatchKMeans, MeanShift, Birch, AffinityPropagation |
| `deepbox/metrics`  | silhouetteScore, adjustedRandScore                                                                                           |
| `deepbox/datasets` | makeBlobs, makeMoons                                                                                                         |

## Usage

```bash
npm run example:35
```

## Output

- Console output only: a silhouette score and ARI per algorithm, and a comparison table for the blobs dataset.

## Files

```
35-advanced-clustering/
├── index.ts     # Example script
└── README.md    # This file
```

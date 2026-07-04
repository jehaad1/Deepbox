# Advanced Clustering

> **View online:** https://deepbox.dev/examples/35-advanced-clustering

Advanced clustering algorithms new in v1.0.0: Agglomerative, GaussianMixture, SpectralClustering, OPTICS, MiniBatchKMeans, MeanShift, Birch, and AffinityPropagation.

## Deepbox Modules Used

| Module            | Features Used                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| `deepbox/ml`      | AgglomerativeClustering, GaussianMixture, SpectralClustering, OPTICS, MiniBatchKMeans, MeanShift, Birch, AffinityPropagation |
| `deepbox/metrics` | silhouetteScore, adjustedRandScore                                                                |
| `deepbox/datasets`| makeBlobs, makeMoons                                                                              |

## Usage

```bash
npm run example:35
```

## Architecture

```
35-advanced-clustering/
├── index.ts     # Main entry point
└── README.md    # This file
```

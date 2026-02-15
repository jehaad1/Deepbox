# Movie Recommendation Engine

> **View online:** https://deepbox.dev/projects/05-recommendation-engine

A collaborative filtering recommendation system using user-item ratings, clustering, and dimensionality reduction.

## Features

- **User-Item Matrix**: Dense ratings matrix with missing values encoded as zeros
- **Collaborative Filtering**: User-based and item-based similarity
- **Clustering**: K-means for user segmentation
- **Dimensionality Reduction**: PCA for visualization

## Deepbox Modules Used

| Module              | Features Used                  |
| ------------------- | ------------------------------ |
| `deepbox/ndarray`   | Tensor operations              |
| `deepbox/ml`        | KMeans, PCA                    |
| `deepbox/metrics`   | silhouetteScore                |
| `deepbox/dataframe` | Result tabulation              |
| `deepbox/plot`      | Cluster and distribution plots |

## Usage

```bash
npm run project:05
```

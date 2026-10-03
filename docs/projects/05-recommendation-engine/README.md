# Movie Recommendation Engine

> **View online:** https://deepbox.dev/projects/05-recommendation-engine

Collaborative filtering on a synthetic rating matrix of 200 users and 50 movies, with user clustering and a 2D projection.

## Features

- User-item matrix with missing ratings stored as zeros
- User-based filtering: cosine similarity between users, then a similarity-weighted average of the neighbours' ratings
- Item-based similarity with adjusted cosine
- `KMeans` clustering of users, with k picked by silhouette score
- `PCA` projection of users to two components
- Leave-one-out check of the rating error, compared with a baseline that always predicts the global mean rating

## Reading the error

The script prints the mean absolute error of the user-based predictions next to the global-mean baseline. The filter beats the baseline on this data (0.89 vs 0.96), but by a modest margin. Treat the baseline as the number to beat, not as a measure of how hard the task is.

## Deepbox Modules Used

| Module              | Features Used                                   |
| ------------------- | ----------------------------------------------- |
| `deepbox/ndarray`   | `tensor`, `toArray`                             |
| `deepbox/ml`        | `KMeans`, `PCA`                                 |
| `deepbox/metrics`   | `silhouetteScore`                               |
| `deepbox/dataframe` | `DataFrame` for the recommendation table        |
| `deepbox/plot`      | `Figure`, PCA scatter plot and rating histogram |

## Usage

```bash
npm run project:05
```

## Output

- Rating statistics, cluster sizes, recommendations for one user and the similarity matrix on the console
- `output/user-clusters.svg`
- `output/rating-distribution.svg`

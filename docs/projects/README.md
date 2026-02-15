# Deepbox Real-World Production Projects

> **Browse online:** https://deepbox.dev/projects · **Docs:** https://deepbox.dev/docs

Enterprise-level projects built with Deepbox. Each project demonstrates comprehensive use of Deepbox's modules in real-world scenarios.

## Projects Overview

| #   | Project                                                            | Modules Used             | Description                                                       |
| --- | ------------------------------------------------------------------ | ------------------------ | ----------------------------------------------------------------- |
| 1   | [Financial Portfolio Risk Analysis](./01-financial-risk-analysis/) | stats, linalg, dataframe | Portfolio optimization, VaR calculation, correlation analysis     |
| 2   | [Neural Network Image Classifier](./02-neural-image-classifier/)   | nn, ndarray, metrics     | MLP-based digit classification with evaluation visualizations     |
| 3   | [Customer Churn Prediction](./03-customer-churn-prediction/)       | ml, preprocess, metrics  | End-to-end ML pipeline with model comparison                      |
| 4   | [Time Series Stock Forecasting](./04-stock-price-forecasting/)     | ml, stats, dataframe     | Regression-based price forecasting with technical indicators      |
| 5   | [Movie Recommendation Engine](./05-recommendation-engine/)         | ml, metrics, dataframe   | Collaborative filtering with K-means clustering and PCA           |
| 6   | [Sentiment Analysis System](./06-sentiment-analysis/)              | ml, preprocess, metrics  | Sentiment classification with logistic regression and Naive Bayes |

## Features Coverage

Each project demonstrates specific Deepbox modules:

### Core Modules

- **deepbox/ndarray**: Tensor operations, autograd, broadcasting
- **deepbox/linalg**: SVD, eigenvalues, matrix decomposition
- **deepbox/dataframe**: Data manipulation, grouping, filtering
- **deepbox/stats**: Statistical tests, correlations, descriptive stats

### Machine Learning

- **deepbox/ml**: Classical ML (RF, GBM, SVM, KNN, Trees)
- **deepbox/preprocess**: Scaling, encoding, train/test split
- **deepbox/metrics**: Classification, regression, clustering metrics

### Deep Learning

- **deepbox/nn**: Neural network layers and losses for the image classifier
- **deepbox/datasets**: Built-in datasets used by ML and NN projects

### Utilities

- **deepbox/random**: Random distributions, sampling
- **deepbox/plot**: Visualization (SVG/PNG output)

## Running the Projects

```bash
# From the Deepbox root directory
npm run build

# Run individual projects
npm run project:01
npm run project:02
npm run project:03
npm run project:04
npm run project:05
npm run project:06
```

## Project Structure

Each project follows a consistent structure:

```
project-name/
├── index.ts           # Main entry point
├── README.md          # Project documentation
├── src/               # Source modules (if applicable)
│   ├── models.ts      # Model definitions
│   ├── data.ts        # Data processing
│   └── utils.ts       # Utility functions
└── output/            # Generated outputs (plots, reports)
```

## License

These projects are part of the Deepbox library and follow the same MIT license.

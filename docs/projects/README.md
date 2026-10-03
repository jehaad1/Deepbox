# Deepbox Projects

> **Browse online:** https://deepbox.dev/projects · **Docs:** https://deepbox.dev/docs

Nine end-to-end projects that use several Deepbox modules together on a larger scale than the example snippets. Each project generates its own synthetic data with a fixed seed, so every run prints the same numbers, apart from timings. All nine run on Deepbox 1.5.0.

## Project Catalog

| #   | Project                                                            | Modules Used                | Description                                                          |
| --- | ------------------------------------------------------------------ | --------------------------- | -------------------------------------------------------------------- |
| 01  | [Financial Portfolio Risk Analysis](./01-financial-risk-analysis/) | stats, linalg, dataframe    | Portfolio optimization, VaR/CVaR, correlations, Monte Carlo          |
| 02  | [Neural Network Image Classifier](./02-neural-image-classifier/)   | nn, optim, ndarray, metrics | Digits classification trained on plain tensors, with training curves |
| 03  | [Customer Churn Prediction](./03-customer-churn-prediction/)       | ml, preprocess, metrics     | Six classifiers, pipeline cross-validation and feature importances   |
| 04  | [Time Series Stock Forecasting](./04-stock-price-forecasting/)     | ml, stats, dataframe        | Synthetic market forecasting with technical indicators               |
| 05  | [Movie Recommendation Engine](./05-recommendation-engine/)         | ml, metrics, dataframe      | Collaborative filtering, clustering, PCA                             |
| 06  | [Sentiment Analysis](./06-sentiment-analysis/)                     | ml, preprocess, metrics     | Text classification with TF-IDF features and Naive Bayes             |
| 07  | [Fraud Detection Platform](./07-fraud-detection-platform/)         | ml, dataframe, plot         | Calibrated fraud scoring, anomaly detection, feature inspection      |
| 08  | [Support Ticket Triage](./08-support-ticket-triage/)               | preprocess, ml, plot        | Text vectorization, model selection, routing diagnostics             |
| 09  | [Experimentation Platform](./09-experimentation-platform/)         | stats, dataframe, plot      | Variant scorecards, uplift inference, KDE plots, power planning      |

## Coverage Summary

- `deepbox/dataframe`: grouping, querying, console tables and JSON export
- `deepbox/preprocess`: scaling, train/test splits and text vectorization
- `deepbox/ml`: supervised learning, anomaly detection, calibration, pipelines, cross-validation and model search
- `deepbox/nn` and `deepbox/optim`: a multi-layer perceptron, loss functions and the Adam optimizer
- `deepbox/stats` and `deepbox/linalg`: finance, experimentation and time-series analysis
- `deepbox/plot`: SVG charts that are written to each project's `output/` folder

## Reading the results

The data in every project is synthetic and often easy to separate, so scores such as 100% accuracy in project 08 say little about real data. Each project README says what to expect. Project 07 explains in detail how to read the anomaly detector numbers, including why the `OneClassSVM` recall differs from the 0.97 that 1.0.0 printed.

## Running Projects

```bash
npm run project:01
npm run project:02
npm run project:03
npm run project:04
npm run project:05
npm run project:06
npm run project:07
npm run project:08
npm run project:09
```

To run all projects in order:

```bash
npm run projects:all
```

To run one project without npm:

```bash
npx tsx --tsconfig docs/projects/tsconfig.json docs/projects/01-financial-risk-analysis/index.ts
```

Each run overwrites the files in that project's `output/` folder.

## License

These projects are part of Deepbox and follow the repository MIT license.

# Deepbox Real-World Projects

> **Browse online:** https://deepbox.dev/projects · **Docs:** https://deepbox.dev/docs

This directory contains **9 production-style projects** that exercise the Deepbox stack at a larger scale than the example snippets.

## Project Catalog

| #   | Project                                                            | Modules Used             | Description                                                         |
| --- | ------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------------- |
| 01  | [Financial Portfolio Risk Analysis](./01-financial-risk-analysis/) | stats, linalg, dataframe | Portfolio optimization, VaR/CVaR, correlations, Monte Carlo         |
| 02  | [Neural Network Image Classifier](./02-neural-image-classifier/)   | nn, ndarray, metrics     | Digits classification with training curves and evaluation           |
| 03  | [Customer Churn Prediction](./03-customer-churn-prediction/)       | ml, preprocess, metrics  | Multi-model churn workflow with CV and feature analysis             |
| 04  | [Time Series Stock Forecasting](./04-stock-price-forecasting/)     | ml, stats, dataframe     | Synthetic market forecasting with technical indicators              |
| 05  | [Movie Recommendation Engine](./05-recommendation-engine/)         | ml, metrics, dataframe   | Collaborative filtering, clustering, PCA                            |
| 06  | [Sentiment Analysis System](./06-sentiment-analysis/)              | ml, preprocess, metrics  | Text classification with bag-of-words and Naive Bayes               |
| 07  | [Fraud Detection Platform](./07-fraud-detection-platform/)         | ml, dataframe, plot      | Calibrated fraud scoring, anomaly detection, and feature inspection |
| 08  | [Support Ticket Triage](./08-support-ticket-triage/)               | preprocess, ml, plot     | Text vectorization, model selection, and routing diagnostics        |
| 09  | [Experimentation Platform](./09-experimentation-platform/)         | stats, dataframe, plot   | Rollout scorecards, uplift inference, KDE diagnostics, power plans  |

## Coverage Summary

- `deepbox/dataframe`: reporting, grouping, export, and operational summaries
- `deepbox/preprocess`: scaling, text vectorization, and dataset preparation
- `deepbox/ml`: supervised learning, anomaly detection, calibration, inspection, and model search
- `deepbox/nn`: neural network architecture and loss usage
- `deepbox/stats` + `deepbox/linalg`: finance, experimentation, and time-series analytical workflows
- `deepbox/plot`: SVG-based artifacts suitable for docs and dashboards

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

To execute all projects sequentially:

```bash
npm run projects:all
```

## License

These projects are part of Deepbox and follow the repository MIT license.

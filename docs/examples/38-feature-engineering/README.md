# Feature Engineering & Preprocessing

> **View online:** https://deepbox.dev/examples/38-feature-engineering

Advanced preprocessing tools new in v1.0.0: imputation, feature selection, text vectorizers, spline transformers, and advanced splitters.

## Deepbox Modules Used

| Module              | Features Used                                                                      |
| ------------------- | ---------------------------------------------------------------------------------- |
| `deepbox/preprocess`| SimpleImputer, KNNImputer, SelectKBest, VarianceThreshold, TfidfVectorizer, CountVectorizer, SplineTransformer, PolynomialFeatures, RobustScaler, PowerTransformer, KFold, StratifiedKFold, TimeSeriesSplit |
| `deepbox/ndarray`   | tensor                                                                             |

## Usage

```bash
npm run example:38
```

## Architecture

```
38-feature-engineering/
├── index.ts     # Main entry point
└── README.md    # This file
```

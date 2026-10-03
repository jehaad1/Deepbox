# Feature Engineering & Preprocessing

> **View online:** https://deepbox.dev/examples/38-feature-engineering

Nine short parts on preprocessing: imputation, feature selection, text vectorizers, polynomial and spline features, median-based and power scalers, and cross-validation splitters. `PowerTransformer` has `standardize` set to `false` by default, while scikit-learn uses `true`. The example shows both settings.

## Deepbox Modules Used

| Module               | Features Used                                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/preprocess` | SimpleImputer, KNNImputer, SelectKBest, fClassif, VarianceThreshold, TfidfVectorizer, CountVectorizer, SplineTransformer, PolynomialFeatures, RobustScaler, PowerTransformer, KFold, StratifiedKFold, TimeSeriesSplit |
| `deepbox/ndarray`    | tensor                                                                                                                                                                                                                |

## Usage

```bash
npm run example:38
```

## Output

- Console output only: each transformer's input and result, and the train and test sizes of every fold.
- The older name `f_classif` still works. Use `fClassif`.

## Files

```
38-feature-engineering/
├── index.ts     # Example script
└── README.md    # This file
```

# Ensemble & Advanced ML Models

> **View online:** https://deepbox.dev/examples/34-ensemble-advanced-ml

Trains AdaBoost, Bagging, Voting, Stacking and ExtraTrees classifiers, a Gaussian process regressor and Linear Discriminant Analysis on synthetic data, then prints a comparison table. It also shows `classWeight` and `clone()`.

## Deepbox Modules Used

| Module               | Features Used                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ml`         | AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier, GaussianProcessRegressor, LinearDiscriminantAnalysis |
| `deepbox/metrics`    | accuracy, r2Score, f1Score                                                                                                                              |
| `deepbox/datasets`   | makeClassification, makeRegression                                                                                                                      |
| `deepbox/preprocess` | trainTestSplit                                                                                                                                          |

## Usage

```bash
npm run example:34
```

## Output

- Console output only: accuracy and F1 for each classifier, the R² of the Gaussian process regressor, and a comparison table.
- Results are seeded with `randomState`, so they repeat between runs.

## Files

```
34-ensemble-advanced-ml/
├── index.ts     # Example script
└── README.md    # This file
```

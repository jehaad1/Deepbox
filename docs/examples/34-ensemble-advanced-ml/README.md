# Ensemble & Advanced ML Models

> **View online:** https://deepbox.dev/examples/34-ensemble-advanced-ml

Advanced ensemble methods and ML models new in v1.0.0: AdaBoost, Bagging, Voting, Stacking, ExtraTrees, Gaussian Processes, Discriminant Analysis, and Semi-supervised Learning.

## Deepbox Modules Used

| Module              | Features Used                                                                          |
| ------------------- | -------------------------------------------------------------------------------------- |
| `deepbox/ml`        | AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier, GaussianProcessRegressor, LinearDiscriminantAnalysis, LabelPropagation |
| `deepbox/metrics`   | accuracy, r2Score, f1Score                                                             |
| `deepbox/datasets`  | makeClassification, makeRegression                                                     |
| `deepbox/preprocess`| trainTestSplit                                                                         |

## Usage

```bash
npm run example:34
```

## Architecture

```
34-ensemble-advanced-ml/
├── index.ts     # Main entry point
└── README.md    # This file
```

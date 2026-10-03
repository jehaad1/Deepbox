# Customer Churn Prediction

> **View online:** https://deepbox.dev/projects/03-customer-churn-prediction

Predicts churn on 1,000 synthetic customers. Six classifiers are trained and compared, then the best one by F1 is cross-validated and inspected.

## Features

- Models: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `GaussianNB`
- Synthetic customer data with ten features and a seeded generator
- Five-fold cross-validation with `crossValScore`. The `StandardScaler` sits inside a `Pipeline`, so each fold fits it on its own training rows only. Folds are stratified by class for classifiers
- Confusion matrix, detection rate and false alarm rate for the best model
- Random forest feature importances

## What to expect

The generator draws churn from a probability that depends on a few features plus random noise, so the best possible accuracy is low. Every model lands near 60% accuracy. Differences of one or two points between models are within noise on a 200-row test set. Read the table as a demonstration of the workflow, not as a ranking of the algorithms.

## Deepbox Modules Used

| Module               | Features Used                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ml`         | `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `GaussianNB`, `Pipeline`, `crossValScore` |
| `deepbox/preprocess` | `StandardScaler`, `trainTestSplit`                                                                                                                                        |
| `deepbox/metrics`    | `accuracy`, `precision`, `recall`, `f1Score`, `confusionMatrix`                                                                                                           |
| `deepbox/dataframe`  | `DataFrame` for the console tables                                                                                                                                        |
| `deepbox/plot`       | `Figure`, model comparison and cross-validation bar charts                                                                                                                |

## Usage

```bash
npm run project:03
```

## Output

- Model comparison table
- Cross-validation scores per fold
- Confusion matrix and detection metrics
- `output/model-comparison.svg`
- `output/cv-scores.svg`

## Architecture

```text
03-customer-churn-prediction/
├── index.ts              # Main entry: data generation, training, CV, plots
├── README.md             # This file
└── output/               # Generated SVGs (model comparison, CV scores)
```

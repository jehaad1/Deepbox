# Model Selection & Pipeline

> **View online:** https://deepbox.dev/examples/37-model-selection-pipeline

Chains a scaler and a model with `Pipeline`, scores models with `crossValidate`, and tunes hyperparameters with `GridSearchCV` and `RandomizedSearchCV`. A pipeline step's parameters are tuned as `stepName__parameterName`, for example `classifier__nNeighbors`.

## Deepbox Modules Used

| Module               | Features Used                                             |
| -------------------- | --------------------------------------------------------- |
| `deepbox/ml`         | Pipeline, GridSearchCV, RandomizedSearchCV, crossValidate |
| `deepbox/preprocess` | StandardScaler, trainTestSplit                            |
| `deepbox/datasets`   | makeClassification                                        |

## Usage

```bash
npm run example:37
```

## Output

- Console output only: pipeline accuracy, cross-validation scores for four models, the best parameters found by grid and randomized search, and test accuracy of the best model.
- The older name `cross_validate` still works. Use `crossValidate`.

## Files

```
37-model-selection-pipeline/
├── index.ts     # Example script
└── README.md    # This file
```

# Tree-Based & Ensemble Models

Decision Trees, Random Forests, Gradient Boosting, and Linear SVM. Covers both classification and regression variants.

## Deepbox Modules Used

| Module               | Features Used                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `deepbox/datasets`   | loadIris                                                                                    |
| `deepbox/ml`         | DecisionTree, RandomForest, GradientBoosting (Classifier + Regressor), LinearSVC, LinearSVR |
| `deepbox/ndarray`    | tensor, slice                                                                               |
| `deepbox/metrics`    | accuracy, mse, r2Score                                                                      |
| `deepbox/preprocess` | trainTestSplit                                                                              |

## Usage

```bash
npm run example:11
```

## Output

- Console output showing accuracy and regression metrics for 8 different tree-based and ensemble models

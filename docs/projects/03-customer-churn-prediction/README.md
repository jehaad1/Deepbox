# Customer Churn Prediction System

A production-grade customer churn prediction system demonstrating classical machine learning with Deepbox.

## Features

- **Multiple Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Gaussian Naive Bayes
- **Feature Engineering**: Synthetic customer data generation
- **Cross-Validation**: K-Fold validation for robust evaluation
- **Model Comparison**: Comprehensive metrics comparison
- **Feature Importance**: Analysis of predictive features

## Deepbox Modules Used

| Module               | Features Used                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ml`         | LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, GaussianNB |
| `deepbox/preprocess` | StandardScaler, trainTestSplit, KFold                                                                                            |
| `deepbox/metrics`    | accuracy, precision, recall, f1Score, confusionMatrix                                                                            |
| `deepbox/dataframe`  | DataFrame for data manipulation                                                                                                  |
| `deepbox/plot`       | Model comparison and cross-validation visualization                                                                              |

## Usage

```bash
npm run project:03
```

## Output

- Model comparison table
- Confusion matrix summaries
- Feature importance analysis

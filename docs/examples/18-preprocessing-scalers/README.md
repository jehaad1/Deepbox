# Preprocessing — Scalers

> **View online:** https://deepbox.dev/examples/18-preprocessing-scalers

Feature scaling is essential before many ML algorithms. Deepbox provides 7 feature scalers.

## Deepbox Modules Used

| Module               | Features Used                                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `deepbox/ndarray`    | tensor                                                                                                      |
| `deepbox/preprocess` | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer |

## Usage

```bash
npm run example:18
```

## Output

- Console output comparing all 7 scaler outputs on the same dataset

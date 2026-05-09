# Model Summary

## Machine Learning Model

- Model Type: Random Forest Regression
- Purpose: MVC and fatigue-related prediction
- Application: Exercise optimization and endurance estimation

## Mathematical Model

An exponential fatigue model was integrated to estimate:

- Fatigue progression
- Safe endurance limits
- Optimal contraction duration
- Recovery intervals

| Parameter | Formula | Description |
|---|---|---|
| Predicted MVC | `MVC = a1(Age) + a2(Gender) + a3(Activity) + a4(Force) + C` | Maximum voluntary contraction estimated via weighted linear regression of physiological inputs |
| Fatigue Index (FI) | `FI = (Initial Force - Final Force) / Initial Force` | Normalized measure of force decline (0 to 1 range) |
| Fatigue Rate Model | `F(t) = MVC × e^(-kt)` | Exponential decay of force capacity over time `t`, with `k` as fatigue coefficient |
| Safe Exercise Time | `SET = (1/k) × ln(MVC / F_threshold)` | Safe duration before force drops to minimum safe level |

## Evaluation

The framework was evaluated using statistical analysis, feature importance assessment, and prediction visualization.

An important factor to considering the right metrics,apart from the objective function, is to understand the significance of the rare observations.
If the rare observations are important then the metric should penalize the underperformance of those observations.

Note: If the dataset is highly imbalanced, accuracy_score is not preferred choice of metrics. Confusion matrix, f1_score, recall are used for imbalanced datasets.

1. $R_{2}$ (R^2)
2. Residual Sum of Squares
3. Mean Squared Error
4. Mean Absolute Error
5. Median Absolute Error
6. Root Mean Square Error
7. Explained Variance Score
8. Log-Loss

**1. R^2 - Coefficient of Determination**
A measure to know how well the model predict for the future samples.

R^2(y, \hat{y}) = 1 - \frac{\sum_0^{n-1} (y_i - \hat{y}_i)^2}{\sum_0^{n-1}(y_i - \bar{y})^2}

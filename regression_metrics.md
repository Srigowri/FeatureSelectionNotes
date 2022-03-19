An important factor to considering the right metrics,apart from the objective function, is to understand the significance of the rare observations. If the rare observations are important then the metric should penalize the underperformance of those observations.

Note: If the dataset is highly imbalanced, accuracy_score is not preferred choice of metrics. Confusion matrix, f1_score, recall are used for imbalanced datasets.

_**Regression Metrics**_

1. R^2 - coefficient of Determination  [Widely used]
2. Residual Sum of Squares
3. Mean Squared Error / l2 norm loss
4. Mean Absolute Error / l1 norm loss  [Robust to outliers]
5. Median Absolute Error   [Most Robust to outliers]
6. Root Mean Square Error
7. Explained Variance Score

**1. R^2 - Coefficient of Determination**
- Used in regression
- A measure to know how well the model predict for the future samples.
- y is the vector of true values, y' is the predicted values, y_avg is the average, R^2 is given by

R2(y,y') = 1 - (sum(yi - y'i)^2) / (sum(yi- y_avg))^2)

The best value is 1, a constant model results in 0, a model worse that the constant model has negative values.

```
#sklearn
from sklearn.metrics import r2_score

#Scratch
def r2_score(y_true, y_pred):
  n = len(y)
  y_avg = np.sum(y_true) / n
  return 1 - (np.sum((y_true - y_pred)**2)) / np.sum((y_true - y_avg)**2)))
```

**2. Residual Sum of Squares**
- Used in least square regression to minimize this value
- Take the square of sum of differences between the observed and the predicted values
- Does not give much intuition as a metric/ not interpretable
- Rarely used as a metric on its own (used in Least Square Regression)

```
#no scikit-learn implementation

#Scratch
def rss(y_true, y_pred):
  return np.sum((y_true - y_pred)**2)
```

**3. Mean Squared Error (MSE)**
- Used in regression
- Interpretable version of RSS (on an average how far behind is the model wrt error)
- Take the average of square of sum of differences between the observed and the predicted values
- Not robust to outliers because of the squaring term 

```
#scikit-learn
from sklearn.metrics import mean_squared_error


#Scratch
def mean_squared_error(y_true, y_pred):
  return np.sum((y_true - y_pred)**2) * (1/len(y))
```

**4. Mean Absolute Error (MAE)**
- l1 error, robust to outliers

```
#scikit-learn
from sklearn.metrics import mean_absolute_error

#Scratch
def mean_absolute_error(y_true, y_pred):
  return np.sum(np.abs(y_true - y_pred)) * (1/len(y))
```

**5. Median Absolute Error (MAE)**
- l1 error, robust to outliers of all the metrics

```
#scikit-learn
from sklearn.metrics import median_absolute_error

#Scratch
def median_absolute_error(y_true, y_pred):
  return np.median(np.abs(y_true - y_pred))
```

**6. Root Mean Squared Error (RMSE)**
- Not robust to outliers

**7. Explained variance**
explained_variance(y_true, y_pred) = 1 - (Var(y_true) - Var(y_pred))/Var(y_true)
```
#scikit-learn
from sklearn.metrics import explained_variance_score

#Scratch
def explained_variance_score(y_true, y_pred):
  return 1 - (np.var(y_true) - np.var(y_pred))/ np.var(y_true)
```



# Notes on Feature Selection

1. Filter Methods
2. Wrapper Methods
3. Embedded Methods


# Filter Methods
1a. **Basic**
```
from sklearn.feature_selection import VarianceThreshold
```

Remove constant features
```
  selection = VarianceThreshold(threshold=0)
```
Remove quasi-constant features

(Constant over the majority of the dataset)
```
  selection = VarianceThreshold(threshold= 0.01)  #99% of the data to show some variance
```
```
  selection.fit(X_train)  #X_train does not include the Target variable
  selected_features = selection.get_support())  # a boolean vector
  print("Number of non constant features", sum(selected_features))  
  print(X_train.columns[selected_features])
  X_train = selection.transform(X_train)
```
1b. **Univariate Feature Selection**

Assumption: Variables have gaussian distribution, linear relation between the feature and the target. Uses univariate statistical test like ANOVA, F-test, chi squared test to check the degree of linear dependency between two random variables

https://scikit-learn.org/stable/modules/feature_selection.html

Note!!! The input to these methods are scoring functions that return univariate scores and p-values.

Classification : chi2, f_classif, mutual_info_classif
Regression: f_regression, mutual_info_regression

f_classif and f_regression estimate the degree of linear dependence between two random variables and they are based on F-score.
mutual_information can estimate any statistical dependency, but requires a large input as they are non-parameteric


SelectKBest: Select features with k highest scores
```
from sklearn.feature_selection import SelectKBest, chi2
num_features = 2
X_train = SelectKBest(chi, k = num_features).fit_transform(X_train, y_train)
print(X_train.shape)
```

SelectPercentile: Select features based on the percentile of the highest scores

```
from sklearn.feature_selection import SelectPercentile
X_train = SelectPercentile(percentile = 10).fit_transform(X_train, y_train)
print(X_train.shape)

#number of features that are in top 10 percentile
```

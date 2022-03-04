# Notes on Feature Selection

1. Filter Methods
2. Wrapper Methods
3. Embedded Methods


# Filter Methods
1a. **Basic**
```
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=0)  #Remove constant features

#(Constant over the majority of the dataset)
selection = VarianceThreshold(threshold= 0.01)  #99% of the data to show some variance
#Remove quasi-constant features
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

Classification : chi2 (Fisher score implementation, based on p-value), f_classif (ANOVA-F score implementation), mutual_info_classif (non-parameteric mutual infomation)
Regression: f_regression (ANOVA-F score implementation), mutual_info_regression (non-parameteric mutual information)

SelectKBest: Select features with k highest scores
```
from sklearn.feature_selection import SelectKBest, chi2
num_features = 2
X_train = SelectKBest(chi, k = num_features).fit_transform(X_train, y_train)  #Fisher score 
print(X_train.shape)
```
The chi squared test has a null hypothesis that the feature and the target are independent and alternative that they're are dependent.
p is the probability that they are independent. Choose a small value of p to reject the null hypothesis.

SelectPercentile: Select features based on the percentile of the highest scores

```
from sklearn.feature_selection import SelectPercentile
X_train = SelectPercentile(percentile = 10).fit_transform(X_train, y_train)
print(X_train.shape)

#number of features that are in top 10 percentile
```

1c. **Mutual Information**

Measure how much each independent varible (feature) depends on the dependent varible(target) and select the ones with the maximum independent gain. If the feature and the target are independent, the information gain is 0, otherwise is a positive value.

```
from sklearn.feature_selection import mutual_info_classif #discrete target
from sklearn.feature_selection import mutual_info_regression #continous target
scores = mutual_info_classif(feature_df, target_df)
```
1d. **Correlation**
Find the correlation among all the features and the target. Feature and target must be highly correlated and the features must be uncorrelated.
```
df = pd.DataFrame(feature_with_target)
corr_matrix = df.corr()
plt.figure(figsize=(8,6))
plt.title('Correlation Heatmap of Iris Dataset')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()    
```


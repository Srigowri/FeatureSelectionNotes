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
SelectPercentile: Select features based on the percentile of the highest scores

```
from sklearn.feature_selection import SelectKBest, chi2
num_features = 2
X_train = SelectKBest(chi2, k = num_features).fit_transform(X_train, y_train)  #Fisher score 
print(X_train.shape)

from sklearn.feature_selection import SelectPercentile
X_train = SelectPercentile(percentile = 10).fit_transform(X_train, y_train)
print(X_train.shape)

#number of features that are in top 10 percentile
```
The chi squared test has a null hypothesis that the feature and the target are independent and alternative that they're are dependent.
p is the probability that they are independent. Choose a small value of p to reject the null hypothesis.


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
df = pd.DataFrame(X)
corr_matrix = df.corr()
plt.figure(figsize=(8,6))
plt.title('Correlation Heatmap of Iris Dataset')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()    
```
Drop the features (columns) that have high correlation (indicating that feature is redundant)

# Wrapper Methods
This a computationally expensive search method, find the subset of the features that are important by training the model and checking the performance

- Forward Selection  (Starts empty, iteratively adds up greedily- until no improvement in performance. Use mlxtend.feature_selection, usually roc or r2 is used as metrics)
- Backward elimination (Starts with entire set and removes least significant feature, until no improvement in performance. Use mlxtend.feature_selection, usually roc or r2 is used as metrics)
- Exhaustive feature selection - Enumerated over all subsets except empty set
- Recursive feature selection [https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py](url)
- Recursive feature elimination (uses cross validation) [https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py](url)


```
Forward Selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#perform any data preprocessing (converting categorical to numerical, removing few of highly correlated feature, removing highly independent feature wrt target) etc

#model corresponds to either a classifier or a regressor
Forward Selection
sfs = SFS(model, k_features = 10, forward= True, floating = False, verbose=2, scoring='r2',cv=3)
sfs.fit(X_train,y_train)
print(sfs.k_feature_idx)
print(X_train.columns[list(sfs.k_feature_idx)])


#model corresponds to either a classifier or a regressor
Backward Elimination
sfs = SFS(model, k_features = 10, forward= False, floating = False, verbose=2, scoring='r2',cv=3)
sfs.fit(X_train,y_train)
print(sfs.k_feature_idx)
print(X_train.columns[list(sfs.k_feature_idx)])

```


# Embedded Methods
The regularization methods are embedded methods, in the sense that they penalize the features given a threshold for its coefficient.
LASSO, RIDGE have in-built penalization functions and reduce overfitting. 

Regularization is adding penalty to the parameters of ML model to restrict its freedom.


**Lasso Regession**
- Add L1 regularization i.e the value of the penalty added is the absolute magnitude of the coefficients
- L1 regularization introduces sparsity by shrinking the coeffient to zero for certain features

```
Note!!!: Linear Regression module of sklearn does not have regularization, hence import Lasso from linear_model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
penalty = 100
selection = SelectFromModel(Lasso(alpha = penalty)) #be careful about the penalty, you don't want to remove necessary features
selection.fit(X_train)   #perform any necessary data preprocessing steps.
selected_features = X_train.columns[selection.get_support()] #get support() returns a vector of boolean values
print(np.sum(selection.estimator_.coef_ == 0))
```
**Importance from Random Forest**

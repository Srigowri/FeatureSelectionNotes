Data is missed either randomly or almost-randomly or because of loss of data

**1. Missing completely at Random [MCAR]**

When the probability of a variable missing the data is same for all the observations, then the variable is MCAR.
There is no relationship to the missing data and other values in the sample as it is not systematic.

**2. Missing at Random [MAR]**

There is a systematic relationship between the missing value and the nature of the sample. 
Eg: Men are more likely to disclose their age and weight. This might lead to women having missing values in their data

**3. Missing Not at Random [MNAR]**

Here, the data is not recorded.

Techniques for data imputation (replace missing data with statistical estimates of the missing data)
1. Complete Case Analysis
Make analysis of the data samples with complete data, in other words no missing value for tthe same.
Remove all the samples with missing values. Useful when there are fewer variables with missing data and fewer samples with missing values.

2. Imputation with mean/mode/median
Mean - numerical data with gaussian distribution
Median - numerical data with skewed distribution
Mode - categorical data distribution (frequent category)
Problems with this type of imputation is that when the number of NA is large enough, it will lead to distorting the data distribution, underestimating the variance. This will distort the correlation and the covariance as the intrinsic relations are not preserved.

Note: We need ensure the imputation of the test set also happens from the train set to avoid overfitting.

3. Random Sample Imputation
Replace missing values by randomly selected values from a set of random distribution, this technique will preserve the underlying distribution. The assumption is that the data is Missing Completely at Random.
Note IMP!!!: Make sure to set the random state so that on each run the same results can be reproduced.

```
random_sample = df[variable].dropna().sample(n = df[variable].isnull().sum(), random_state = 0)
random_sample.index = df[df[variable].isnull()].index
df.loc[df[variable].isnull(), variable+'_random'] = random_sample
df[variable+'_random'].fillna(random_sample)

```
4. Replacement with arbitrary value
Here the data is not missing at random, we want to indicate that the data is missing, -1 or 999 or -999 or a value of the range is used in the numerical data(not suitable for linear regression). For categorical ,'missing' is used. 
Not suitable when using with linear models, works best with tree-based models.

5. End of distribution imputation
Replace missing value with the far end / tail end of the variable distribution.
```
X_train[variable].mean() + 3*X_train[variable].std()
```
6. Add Missing value indicator
Add a binary variable to indicate if the data is missing or not, along with using some imputation technique. This is useful even for linear models.


If the missing values contibute to less than 5% of the overall data, don't give them special preference. Instead impute them either with mean or random values. (or as suited)
if they are more than 5%, do mean/median imputation+adding an additional binary variable to capture missingness add a 'Missing' label in categorical variables.

If the imbalance is too high then treat the missing values specially.

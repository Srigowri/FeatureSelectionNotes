Categorical Encoding: Transform strings and labels of category into numerical values
1. One Hot Encoding
2. Label Encoding
3. Ordinal Encoding
4. Helmert Encoding (not clear on this)
5. Binary Encoding
6. Target Mean Encoding

**1. One hot Encoding**

Creates a dummy boolean variable for each unique value. This is suitable when the unique categories are few in number.
#Note: don't drop last or first column for tree based algorithms, for linear regression you may drop it.
```
pandas.get_dummies(df, columns=['Gender'], prefix = ['gender'])  #creates two dummy features

If there are k unique values, we need k-1 features
pandas.get_dummies(data['Gender'], drop_first=True)  #gets merged in one variable
```
Sklearn has OneHotEncoder in the preprocessing module.
```
One hot encoder:

x = data.select_dtypes('object')
encoder =OneHotEncoder()
encoder.fit(x)
onehotlabels = encoder.transform(x).toarray()
print(onehotlabels.shape)
print(onehotlabels)
```
**2. Ordinal Encoding**
Give a meaningful order to the labels. Eg: Grades, Days of the week, Months etc.
Label Encoder and Ordinal Encoder have the similar functionality except that label encoder is used for 1D such as the target.
Ordinal Encoder is used for 2D (n_samples, d_features)
```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
Label Encoder: This will not care about the ordering of the data
le = LabelEncoder()
y = le.fit_transform(y)  #works even for nan

Ordinal Encoder:
oe = OrdinalEncoder()
encoded_x = oe.fit_transform(x)

Ordinal Encoding by custom ordering of data
temp_map = {'Very Cold':0, 'Cold':1,'Warm':2,'Hot':3, 'Very Hot':4}
data['temp_ordinal'] = data['temp'].map(temp_map)
#data['temp_ordinal'] = data['temp'].apply(lambda x: temp_map[x])
```
**3. Helmert Encoding**
Have not understood this yet.

**4. Binary Encoding**
A variable with n unique values can be encoded with log2(n) binary values.
```
from category_encoders import BinaryEncoder
be = BinaryEncoder()
be.fit_transform(data['temp'])
```

**5. Count/Frequency encoding**
Replace each category of label with the count of the occurance or the percentage. 
```
counts_group  = data.groupby(['temp'])['temp'].count()
data.loc[:,'temp_count1'] = data['temp'].map(counts_group)
data

counts_value  = data['temp'].value_counts().to_dict()
data.loc[:,'temp_count2'] = data['temp'].map(counts_value)
data
```
**6. Target Mean Encodingg** 
Find the mean of the categorical variable with respect to the target label.
Group by the categorical variable, find mean of the target and map it back to the original dataframe
```
mapping = data.groupby(by=[<categorical_variable>])[target_variable].mean()
data[<categorical_variable>] = data[<categorical_variable>].map(mapping)
```
A variant to mean encoding is to use smoothed mean values

```
global_mean = data[<target_variable>].mean()
cat_agg = data.groupby(by=<categorical_variable>)[<target_variable>].agg(['count','mean'])
count = cat_agg['count']
mean = cat_agg['mean']
smoothed = (count * mean + weight * global_mean)/(count + weight)
data[<categorical_variable>] = data[<categorical_variable>].map(smoothed)
```
**7. Weight of evidence encoding and Probability encoding**
WoE = ln (P(Good)/P(Bad)) * 100
PE = P(Good)/P(Bad) 

Both of these create a monotonic relation between the categorical variable and the target.





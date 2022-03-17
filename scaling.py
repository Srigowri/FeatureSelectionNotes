#Feature scaling is essential for algorithms whose objective functions rely on distance based metrics
#Eg: K Nearest neighbour, K Means, PCA (it can be skewed towards features with larger magnitudes) and Neural Networks (faster convergence of gradient descent algorithm)
#Scaling is not useful for rule based algorithms such as CART, Random forest, Decision Tree Classifiers
#Linear Discriminant Analysis and Naive Bayes don't benefit much from feature scaling as they adjust the weights accordingly

"""
Normalization: Bounding the data values between [-1,1] or [0,1]
Standardization: Make data values to be of zero mean and unit variance

Mean shifting does not affect the covariance matrix, whereas standardization and scaling changes the matrix.
"""

```
1. Min Max Scaler
2. Max Abs Scaler
3. Standard Scaler
4. Robust Scaler
5. Quantile Transformer Scaler
6. Power Transformer Scaler
7. Unit Vector Scaler
```


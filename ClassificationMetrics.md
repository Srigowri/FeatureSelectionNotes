An important factor to considering the right metrics,apart from the objective function, is to understand the significance of the rare observations. If the rare observations are important then the metric should penalize the underperformance of those observations.
Note: If the dataset is highly imbalanced, accuracy_score is not preferred choice of metrics. Confusion matrix, f1_score, recall are used for imbalanced datasets.


_**Classification Metrics**_
1. Accuracy Score
2. Confusion Matrix
3. Hamming loss
4. Precision and Recall
5. F1-score
6. Receiver Operator Characteristic (ROC)
7. Area Under Curve (AUC)

**1. Accuracy Score**
- Total correct predictions
- Does not give an idea as to where the model is making error

```
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

**2. Confusion Matrix**
- Can see where the model is making mistakes, True Positive, False Positive, True Negative, False Negative
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm/len(y_true), cmap='Blues', annot=True)
```

**3. Hamming Loss**
Information Theory: Number of bits needed to correct the information sent across.
In classification, it is the observations that are incorrectly predicted. Basically, 1- Accuracy.
```
from sklearn.metrics import hamming_loss
hamming_loss(y_true, y_pred)
```

**4. Precision and Recall**
- Your web search engine returns 30 search page results, 20 of which is relevant (TP) and the 10 are irrelevant (FP).
- Precision = True Positive/ (True Positive + False  Positive)

- Your web search engine fails to recall other 40 relevant pages (FN- it is told as not relevant by model which is false)
- Recall = True Positive/ (True Positive + False Negative)

![image](https://user-images.githubusercontent.com/11163109/159114442-318fb2b2-a682-4eb9-9acc-7ddea63b5025.png)

**5. f1 score**
- A value between 0 and 1
- Weighted harmonic mean of precision and recall
- sklearn provides f1beta_score to find a balance between precision and recall. 
- beta = 0, the metric considers only precision
- beta = 1, the metric considers f1 score
- beta > 1, the metric considers recall
- This metric is specific to the model end user

**6. Receiver Operator Characteristic (ROC)**
The slope and area of the ROC plot signifies the following things:
- How hard it is for the model to correctly pick the correct responses
- How confidently the model will pick the true positives    
- Best model sticks to the top of the curve
- We can analyze the slope to understand when the model picks up and falls off.
- 
```
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_hat)
fig = plt.figure( figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', label='Model Performace')
plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

```

**7. Area Under Curve (AUC)**
- Area under the ROC curve, higher the better.
- Classification metric of choice
```
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y, y_hat)
roc_auc_score(y, y_pred_proba)
```


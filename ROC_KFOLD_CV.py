#Reference: https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv/notebook

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
%matplotlib inline


cv = StratifiedKFold(n_splits = 5, shuffle=False)
clf = RandomForestClassifier(n_estimators = 100, random_state = 0)

tprs = []
aucs = []
mean_fpr = np.linspace(0,1, 100)
fig= plt.figure(figsize = (10,5))
i = 0
for train, test in cv.split(X,y):
    X_train, x_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(x_test)[:,1]   
    fpr, tpr, thres = roc_curve(y_test, prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
    
mean_tpr = np.mean(tprs, axis = 0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')  #random classifier
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

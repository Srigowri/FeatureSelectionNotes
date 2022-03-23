from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import numpy as np
from collections import Counter 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import subprocess
from IPython.display import Image

seed = 42
X,y = make_classification(n_samples = 1000, n_features = 20, n_informative=8, n_redundant=3, n_repeated=2,random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=seed)

####################################################################
#Decision Tree Classifer
decision_tree = DecisionTreeClassifier(random_state = seed)
decision_tree.fit(X_train, y_train)
decision_tree_y_pred = decision_tree.predict(X_test)
decision_tree_y_proba = decision_tree.predict_proba(X_test)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
decision_tree_logloss = log_loss(y_test, decision_tree_y_proba)
print("====DecisionTree===")
print("Accuracy = ",decision_tree_accuracy)
print("Log loss = ", decision_tree_logloss)
print("Number of nodes = ", decision_tree.tree_.node_count)

dt_viz_file = './dt.dot'
dt_png_file = './dt.png'

export_graphviz(decision_tree, out_file=dt_viz_file)
cmd = ["dot", "-Tpng", dt_viz_file, "-o", dt_png_file]
subprocess.check_call(cmd)
Image(filename=dt_png_file)

#####################################################################
#Adaboost Classifer
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
               algorithm='SAMME', n_estimators =1000,random_state = seed)
adaboost.fit(X_train, y_train)
adaboost_y_pred = adaboost.predict(X_test)
adaboost_y_proba = adaboost.predict_proba(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_y_pred)
adaboost_logloss = log_loss(y_test, adaboost_y_proba)

print("====DecisionTree===")
print("Accuracy = ",adaboost_accuracy)
print("Log loss = ", adaboost_logloss)
dt_viz_file = './dt.dot'
dt_png_file = './dt.png'
export_graphviz(adaboost.estimators_[0], out_file=dt_viz_file)
cmd = ["dot", "-Tpng", dt_viz_file, "-o", dt_png_file]
subprocess.check_call(cmd)
Image(filename=dt_png_file)
#####################################################################
gbc = GradientBoostingClassifier(max_depth=1, n_estimators =1000,random_state = seed)
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)
gbc_y_proba = gbc.predict_proba(X_test)
gbc_accuracy = accuracy_score(y_test, gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_proba)
print("====DecisionTree===")
print("Accuracy = ",gbc_accuracy)
print("Log loss = ", gbc_logloss)
dt_viz_file = './dt.dot'
dt_png_file = './dt.png'
export_graphviz(gbc.estimators_[2][0], out_file=dt_viz_file)
cmd = ["dot", "-Tpng", dt_viz_file, "-o", dt_png_file]
subprocess.check_call(cmd)
Image(filename=dt_png_file)
#####################################################################
#XGB classifer

from xgboost.sklearn import XGBClassifier

params = {'objective':'binary:logistic','max_depth':2,'silent':1,'learning_rate':1,'n_estimators':5}
num_rounds = 5
bst = XGBClassifier(**params).fit( X_train,y_train)
preds = bst.predict(X_test)

corrects = 0
for i,y in enumerate(preds):
  if y_test[i] == y:
    corrects+=1
acc=accuracy_score(y_test, preds)
print(acc, corrects/len(y_test))
#####################################################################

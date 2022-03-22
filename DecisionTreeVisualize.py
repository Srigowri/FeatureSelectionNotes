import subprocess
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


dt_viz_file = './dt.dot'
dt_png_file = './dt.png'

seed = 42
X,y = make_classification(n_samples = 1000, n_features = 20, n_informative=8, n_redundant=3, n_repeated=2,random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=seed)

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

export_graphviz(decision_tree, out_file=dt_viz_file)
cmd = ["dot", "-Tpng", dt_viz_file, "-o", dt_png_file]
subprocess.check_call(cmd)
Image(filename=dt_png_file)

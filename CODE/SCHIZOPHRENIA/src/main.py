import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from lightgbm import LGBMClassifier
import pickle

FNC: pd.DataFrame = pd.read_csv("Data/Train/train_FNC.csv")
print(FNC.head())

SBM: pd.DataFrame = pd.read_csv("Data/Train/train_SBM.csv")
print(SBM.head())

labels: pd.DataFrame = pd.read_csv("Data/Train/train_labels.csv")
print(labels.head())

# Sets of Features Source Based Morph and Functional Connection
SBM = SBM.drop(columns=['Id'])
FNC = FNC.drop(columns=['Id'])

y = labels["Class"]
X = FNC

# features and labels
print(X.head())
print(y.head())

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Light GBM
clf = LGBMClassifier(random_state=8, num_leaves=2, n_estimators=40)
clf.fit(X_train, y_train)
print("LGBM Classifier: ", clf.score(X_test, y_test))

# Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Random Forest: ", clf.score(X_test, y_test))

# Gradient Boost Machine
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print("Gradient Boost Machine: ", clf.score(X_test, y_test))

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Decision Tree: ", clf.score(X_test, y_test))

# AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
print("AdaBoost: ", clf.score(X_test, y_test))

# Guassian Process
clf = GaussianProcessClassifier()
clf.fit(X_train, y_train)
print("Guassian Process: ", clf.score(X_test, y_test))

# Xgboost
clf = xgb.XGBClassifier(use_label_encoder=False, tree_method="exact")
clf.fit(X_train, y_train)
print("XGBoost: ", clf.score(X_test, y_test))

pickle.dump(clf, open("xgb_model.pkl", "wb"))

pred = clf.predict(X_test.iloc[[2]])
actual = y_test.iloc[[2]]
print("predicted: {0} actual: {1}".format(pred, actual))

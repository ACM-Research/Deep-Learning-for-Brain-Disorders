from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

df = pd.read_csv("DATA/BRAIN.csv")
print(df.head())

labels   = df["Class"]
features = df.drop(["Class", "Image"], axis=1)

clf = xgb.XGBClassifier(max_depth=15, alpha=10,
                        n_estimators=10)

X_train, X_test, y_train, y_test = train_test_split(features, labels)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

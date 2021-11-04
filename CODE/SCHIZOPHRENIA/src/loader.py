import pickle
import pandas as pd

xgb_model = pickle.load(open("xgb_model.pkl", "rb"))

FNC = pd.read_csv("Data/Test/test_FNC.csv")
FNC = FNC.drop(columns=["Id"])

print(xgb_model.predict(FNC.iloc[[1]]))


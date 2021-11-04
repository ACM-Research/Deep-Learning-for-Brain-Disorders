from model import model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.image import imread

HEIGHT, WIDTH = 256, 256

def main():
    df = pd.read_csv("DATA/train.csv")

    y = df["Label"]
    ims = []
    for i in range(len(df)):
        brn = np.resize(imread(df["Paths"][i]), (HEIGHT, WIDTH, 3))
        ims.append(brn)
    X = np.array(ims)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # TRAIN THE MODEL
    model.fit(X_train, y_train, epochs=15)
    model.evaluate(X_test, y_test)
    model.save("alz-model")

if __name__ == "__main__":
    main()

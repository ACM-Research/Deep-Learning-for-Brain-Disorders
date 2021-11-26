import pandas as pd
import numpy as np
from model import model
from matplotlib.image import imread
from sklearn.model_selection import train_test_split

HEIGHT, WIDTH = 256, 256

def main():
    df = pd.read_csv("DATA/train.csv")
    y = df["Class"]
    ims = []
    for i in range(len(df)):
        brn = np.resize(imread(df["Paths"][i]), (HEIGHT, WIDTH, 3))
        ims.append(brn)
    X = np.array(ims)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model.fit(X_train, y_train, epochs=1)
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()

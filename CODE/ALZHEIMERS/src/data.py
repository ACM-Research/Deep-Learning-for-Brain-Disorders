import os
import pandas as pd
from matplotlib.image import imread
import numpy as np
from sklearn.model_selection import train_test_split

HEIGHT, WIDTH = 256, 256
ROOT_DIR = "DATA/train/"

def construct_df(PATH_DIR: str, i: int) -> pd.DataFrame:
    df = pd.DataFrame()
    dfp = [ROOT_DIR + PATH_DIR + x for x in os.listdir(ROOT_DIR + PATH_DIR)]
    ls = np.tile(i, len(dfp))
    df["Paths"] = dfp
    df["Label"] = ls
    #df["Pixel"] = df["Paths"].map(lambda x: np.resize(imread(x), (HEIGHT, WIDTH, 3)))

    return df

mild = construct_df("MildDemented/", 0)
nond = construct_df("NonDemented/", 1)
modd = construct_df("ModerateDemented/", 2)
vmid = construct_df("VeryMildDemented/", 3)

df = pd.DataFrame()
df = df.append([mild, nond, modd, vmid], ignore_index=True)
df = df.sample(frac=1)
df.to_csv("DATA/train.csv")
print(df.head())

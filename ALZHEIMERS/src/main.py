# deps
import os
import pandas as pd
import matplotlib.image as mpimg
import numpy as np

direc = "./DATA/train/"
dirs_list = os.listdir(direc)

images = []
labels = []
for i in range(len(dirs_list)):
    y = [mpimg.imread(direc + dirs_list[i] + "/" + x) for x in os.listdir(direc + dirs_list[i])]
    t = np.tile(i, len(y))
    images.append(y)
    labels.append(t)

images = pd.Series(images, name="Images")
labels = pd.Series(labels, name="Labels")

df = pd.concat([images, labels], axis=1, join='outer')
df = df.sample(frac=1)

df.to_csv("train.csv")

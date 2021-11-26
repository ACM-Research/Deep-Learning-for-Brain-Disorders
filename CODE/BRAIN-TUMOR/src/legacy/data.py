import os
import pandas as pd
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split

# global options
ROOT = "DATA/"
HEIGHT, WIDTH = 256, 256

# select yes paths
yes_paths = [ROOT + "yes/" + x for x in os.listdir(ROOT + "yes/")]
yes_class = np.tile(1, len(yes_paths))

# select no paths
no_paths = [ROOT + "no/" + x for x in os.listdir(ROOT + "no/")]
no_class = np.tile(0, len(no_paths))

# labels and paths
labels = pd.Series(np.concatenate([yes_class, no_class]))
paths  = pd.Series(np.concatenate([yes_paths, no_paths]))

# make data frame
df = pd.DataFrame()

# create columns
df["Class"] = labels
df["Paths"] = paths

# shuffle
df = df.sample(frac=1)
df.to_csv("DATA/train.csv")

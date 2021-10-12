import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# example dir
CUR_DIR = "DATA/train/MildDemented/"

# read in an entire folder of files
def gen_case(direc):
    ims_list = os.listdir(direc)
    ims_tensor = [mpimg.imread(direc + x) for x in ims_list]
    return ims_tensor

# shape of one img is (208, 176)
imgs = gen_case(CUR_DIR)

labels = np.tile(1, 717)

# create two series
labels = pd.Series(labels)
imgs = pd.Series(imgs)

print("\nLABELS:\n")
print(labels.head())

print("\nIMGS:\n")
print(imgs.head())

df = pd.concat([imgs, labels], axis=1, join='outer')
print(df.head())

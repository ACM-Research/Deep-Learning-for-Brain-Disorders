# deps
import os
import pandas as pd
import matplotlib.image as mpimg
import numpy as np

df = pd.DataFrame()

paths = []
label = {}

i = 0

for root, dirs, files in os.walk("./DATA/train/"):
    for file in files:
        if file.endswith(".jpg"):
            if root not in label:
                label[root] = i
            if root in label:
                label[root] += 1
            paths.append(root + "/" + file)

images = [mpimg.imread(x) for x in paths]
images = pd.Series(images)

print(label)

# hardcoded time
l1 = pd.Series(np.tile(1, label['./DATA/train/MildDemented']))
l2 = pd.Series(np.tile(2, label['./DATA/train/ModerateDemented']))
l3 = pd.Series(np.tile(3, label['./DATA/train/NonDemented']))
l4 = pd.Series(np.tile(4, label['./DATA/train/VeryMildDemented']))

labels = pd.concat([l1, l2, l3, l4], axis=0)

# I just realized we don't need to put these into a dataframe
print(labels.shape)
print(images.shape)

"""DATA PROCESSING COMPLETE"""

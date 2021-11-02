# deps
import os
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split

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
l1 = np.tile(1, label['./DATA/train/MildDemented'])
l2 = np.tile(2, label['./DATA/train/ModerateDemented'])
l3 = np.tile(3, label['./DATA/train/NonDemented'])
l4 = np.tile(4, label['./DATA/train/VeryMildDemented'])

# labels
labels = pd.Series(np.concatenate([l1, l2, l3, l4]))

# finally
data = {"Images": images, "Labels": labels}
df = pd.concat(data, axis=1)
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())

print(df["Images"][0].shape)

"""DATA PROCESSING COMPLETE"""
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras

# Initialize network

X_train, X_test, y_train, y_test = train_test_split(df["Images"], df["Labels"])

X_train = np.array([x[None, :, :] for x in X_train])

print(model.forward(torch.Tensor(X_train)))

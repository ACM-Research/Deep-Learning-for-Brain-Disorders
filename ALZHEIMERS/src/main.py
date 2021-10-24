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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

# Hyperparameters
learning_rate = 1e-3

"""
# Chen CNN (RIP)
class CNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(64),
            nn.Conv2d(64, 192, kernel_size=(2, 2),
            nn.ReLU(),
            nn.MaxPool2d(192),
            nn.Conv2d(192, 384, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(256),
            # nn.AdaptiveAvgPool2d((6,6)),
            # nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.main(x)
"""

class CNN(nn.Module):
    def __init__(self, out_size:int = 4):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=1),
            nn.MaxPool2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(960, 480),
            nn.ReLU(),
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Initialize network
model = CNN()
print(model)

X_train, X_test, y_train, y_test = train_test_split(df["Images"], df["Labels"])

X_train = np.array([x[None, :, :] for x in X_train])

print(model.forward(torch.Tensor(X_train)))

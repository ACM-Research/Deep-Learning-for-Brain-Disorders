from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data import df

HEIGHT = 256
WIDTH  = 256
EPOCHS = 10
NUM_CLASSES = 1

model = Sequential([
    # input layer
    layers.InputLayer((HEIGHT, WIDTH, 3)),

    # encoder
    keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal"),
    keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

    # decoder
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="relu"),
])

print(model.summary())

model.compile( optimizer=keras.optimizers.Adam(learning_rate=1e-3)
             , loss=keras.losses.BinaryCrossentropy(from_logits=True)
             , metrics=['accuracy'])

# labels
y = df["Class"]

# features
features = []
for i in range(len(df)):
    brain_img = df["Pixel"][i].astype(np.float32)
    brain_img /= 255
    features.append(brain_img)
X = np.array(features)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(X_train.shape)

# training
model.fit(x=X_train, y=y_train, epochs=EPOCHS, batch_size=10)

# testing
print(model.evaluate(X_test, y_test))

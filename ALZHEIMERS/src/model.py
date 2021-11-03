from tensorflow.keras import layers
from tensorflow import keras

HEIGHT, WIDTH = 256, 256
NUM_CLASSES = 4

model = keras.models.Sequential([
    # input layer
    layers.InputLayer((HEIGHT, WIDTH, 3)),

    # encoder
    keras.layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
    keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
    keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

    # decoder
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="relu")
])

print(model.summary())
model.compile( optimizer=keras.optimizers.Adam(learning_rate=1e-3)
             , loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
             , metrics=['accuracy'])

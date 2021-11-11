import sys
import pickle
import tensorflow as tf
from preprocess import dPipeline
import numpy as np

def main():
    alz = tf.keras.models.load_model("models/alzheimers")
    print(alz.summary())

    brn = tf.keras.models.load_model("models/brain_tumor")
    print(brn.summary())

    alzSlice, brnSlice, connDat, signDat = dPipeline(sys.argv[1])

    pred = alz.predict(alzSlice)
    print(pred)

    pred = brn.predict(brnSlice)
    print(pred)

    # xgb_model_loaded = pickle.load(open("./models/xgb_model.pkl", "rb"))
    # pred = xgb_model_loaded.predict(signDat)
    # print(pred)

if __name__ == "__main__":
    main()

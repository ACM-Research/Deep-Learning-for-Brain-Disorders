import tensorflow as tf

def main():
    alz = tf.models.load_model("models/alzheimers")
    print(alz.summary())

if __name__ == "__main__":
    main()

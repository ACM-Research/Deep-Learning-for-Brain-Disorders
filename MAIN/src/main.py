import os
import sys
import random
import torch
import numpy as np
from fastai.vision.all import *
from typing import List

# mapping for loading pickel file
def is_yes(fname):
    return "y" in fname.lower()

# dict for mapping
lbl_dict = dict(
      MildDemented = "Mild",
      ModerateDemented = "Moderate",
      NonDemented = "Normal",
      VeryMildDemented = "VeryMild"
  )

# mapping for loading pickel file
def label_func(fname):
  return lbl_dict[parent_label(fname)]

# combining the outputs of our model
def AugmentedOr(probs: List[np.float32], labels: List[str]) -> str:
    """
    probs: List[np.float32]
    labels: List[str]

    Takes in the two arguements, and weights values based on
    confidence in the case of a false prediction from a binary
    network drop the network output then combine the remaining
    valid outputs using Noisy Or.
    """

    # in the case that it's not brain cancer
    if ("False" in labels):
        # output alzheimers prediction
        return f"confidence: {np.max(probs[0])}\nlabel: {labels[0]}"

    # if it might be brain cancer combine with noisy or to discriminate outputs
    weights = [np.max(p)*np.ceil(np.average(p)) for p in probs]

    # output prediction
    return f"confidence: {np.max(probs)} \nlabel: {labels[np.argmax(weights)]}"

def BrnTest(models, path, size):
    """
    Please ignore this code this was used
    for testing pourposes to get numbers
    for the project this code is really
    bad and should not be used.
    """

    paths = [path + str(random.choice(os.listdir(path))) for _ in range(size)]
    images = [PILImage.create(x) for x in paths]
    labels = [is_yes(x) for x in paths]

    predictions = []
    for image in images:
        a = []
        b = []
        for model in models:
            name, _, confidence = model.predict(image)
            a.append(confidence.numpy())
            b.append(name)
        predictions.append(AugmentedOr(a, b))

    count = 0
    for i in range(size):
        if (labels[i] == False):
            if ((predictions[i] == "Normal") or (predictions[i] == "VeryMild")):
                count += 1
        elif (str(labels[i]) == predictions[i]):
            count += 1

    print("Accuracy: ", (count / size))

def main():
    # load our two models in with pytorch
    alz = torch.load("models/alzheimers")
    brn = torch.load("models/brain_tumor")

    # read image from commandline arguement
    img = PILImage.create(sys.argv[1])

    # get preds
    name, _, confidence = alz.predict(img)
    bname, _, bconfidence = brn.predict(img)

    # combine the outputs and print prediction
    print(AugmentedOr([confidence.numpy(), bconfidence.numpy()], [name, bname]))

if __name__ == "__main__":
    main()

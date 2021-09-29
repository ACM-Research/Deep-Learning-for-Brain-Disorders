# we can import all kinds of models from torch
from torchvision.models import resnet152

from PIL.Image import open
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
plot an image

plt.imshow(img)
plt.savefig("mild.png")
"""

img = mpimg.imread("DATA/test/MildDemented/26 (19).jpg")

# img.shape all images are the same 208x176 shape
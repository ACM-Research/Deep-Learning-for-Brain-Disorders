import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# read in data and get image part
obj = nib.load("DATA/ABIDE_50054.nii")
img = obj.get_fdata()

"""
This is like the mri volume data
it's a 3 tensor so we want 1 image
some frame and images are empty
"""

plt.imshow(img[110][:][:])
plt.savefig("brain.png")
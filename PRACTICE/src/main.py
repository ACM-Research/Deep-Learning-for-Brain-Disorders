import nibabel as nib
import matplotlib.pyplot as plt

# load object and read in image tensor
obj = nib.load("data/ABIDE_50102_MRI_MP-RAGE_br_raw_20120830181319367_S164732_I328742.nii")
img = obj.get_fdata()

""" Particular Solution

img = img[110][:][:]

plt.imshow(img)
plt.savefig("brain.png")

"""

def getCrossSection(img, index):
    cS = img[index][:][:]
    plt.imshow(cS)
    plt.savefig("brain.png")

getCrossSection(img, 111)

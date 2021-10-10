# deps
import os
import matplotlib.image as mpimg
import numpy as np

# example dir
CUR_DIR = "DATA/train/MildDemented/"

# read in an entire folder of files
def gen_case(direc):
    ims_list = os.listdir(direc)
    ims_tensor = [mpimg.imread(direc + x) for x in ims_list]
    return ims_tensor

# shape of one img is (208, 176)
imgs = gen_case(CUR_DIR)
print(imgs[0].shape) # 208, 176
print(np.array(imgs).shape) # large tensor shape 717, 208, 176

# deps
import matplotlib.image as mpimg
import os

# example dir
CUR_DIR = "DATA/train/MildDemented/"

# read in an entire folder of files
def gen_case(direc):
    ims_list = os.listdir(direc)
    ims_tensor = [mpimg.imread(direc + x) for x in ims_list]
    return ims_tensor

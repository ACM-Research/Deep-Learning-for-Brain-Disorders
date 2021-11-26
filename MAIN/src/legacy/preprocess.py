from nilearn import image
from nilearn import datasets
from nilearn import input_data
import numpy as np
import nibabel as nib
from sklearn.covariance import GraphicalLassoCV
import matplotlib.pyplot as plt

# so there's various data that we need for prediction
# therefore there's different pipelines we need for data processing

def dPipeline(path: str):
   """
   1: Autism
      - Model Takes in connectome data
   2: Alzheimers
      - Model takes in image data
   3: Schizophrenia
      - Model takes in signal data
   4: Brain Tumor
      - Model takes in image data

   This method returns:
   1: alz
      - Image slice for Alz. and Brn Tumor
   2: brn
      - Connection data for Autism
   3: conn
      - Signal data for Schizophrenia
   4: signal
      - Signal data for Schizophrenia
   """

   # load image
   sliceIm = nib.load(path)
   sliceIm = sliceIm.get_fdata()
   # index the first brain in the 4D set of brains
   sliceIm = sliceIm[:, :, :, 0]
   # get the number of slices in the volume data and take the halfway point
   mid = sliceIm.shape[2] // 2
   # index the halfway point that's our image :)
   sliceIm = sliceIm[:, :, mid]
   sliceIm = np.resize(sliceIm, (256, 256, 3))
   alzSlice = sliceIm[None, :, :, :]

   sliceIm = np.resize(sliceIm, (224, 224, 3))
   brnSlice = sliceIm[None, :, :, :]

   # fetch the atlas
   atlas = datasets.fetch_atlas_msdl()
   # create a maps masker
   masker = input_data.NiftiMapsMasker(maps_img=atlas.maps)
   # extract time series information from 4D brain data
   time_series = masker.fit_transform(path)
   # calculate sparse inverse covariance
   estim = GraphicalLassoCV()
   # calculate covariance with graphical CV
   estim.fit(time_series)
   # connection data for autism
   connDat = estim.covariance_

   # Schizophrenia Hypothenically
   atlas = datasets.fetch_atlas_aal()
   masker = input_data.NiftiMapsMasker(maps_img=atlas.maps)
   time_series = masker.fit_transform(path)
   signDat = time_series

   return alzSlice, brnSlice, connDat, signDat


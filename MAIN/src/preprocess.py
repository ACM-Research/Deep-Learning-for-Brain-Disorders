from nilearn import image
from nilearn import datasets
from nilearn import input_data
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
   1: sliceIm
      - Image slice for Alz. and Brn Tumor
   2: connDat
      - Connection data for Autism
   3: signDat
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

   # fetch the atlas
   atlas = datasets.fetch_atlas_msdl()
   # create a maps masker
   masker = input_data.NiftiMapsMasker(maps_img=atlas.maps)
   # extract time series information from 4D brain data
   time_series = masker.fit_transform(path)
   # calculate sparse inverse covariance
   estim = GraphicalLassoCV()
   estim.fit(time_series)

   # connection data for autism
   connDat = estim.covariance_
   
   # signal data for schizophrenia
   signDat = time_series[mid]

   return sliceIm, connDat, signDat

# test case
sliceIm, connDat, signDat = dPipeline("rest.nii.gz")

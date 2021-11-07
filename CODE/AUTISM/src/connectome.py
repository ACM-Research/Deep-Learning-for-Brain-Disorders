import sys

from nilearn import datasets
from nilearn import input_data
from sklearn.covariance import GraphicalLassoCV

atlas = datasets.fetch_atlas_msdl()

# extract time series information
masker = input_data.NiftiMapsMasker(maps_img=atlas.maps, standardize=True, verbose=5)

time_series = masker.fit_transform(sys.argv[0])
estim = GraphicalLassoCV()
estim.fit(time_series)

# connectome information
print(estim.covariance_)

from nilearn.regions import Parcellations

kmeans = Parcellations(method='kmeans', n_parcels=50,
                       standardize=False, smoothing_fwhm=1.,
                       verbose=1)

kmeans.fit("DATA/ABIDE_50054.nii")
kmeans_labels_img = kmeans.labels_img_
kmeans_labels_img.to_filename('kmeans_parcellation.nii.gz')

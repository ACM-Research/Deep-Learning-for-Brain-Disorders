# first line: 62
def filter_and_mask(imgs, mask_img_, parameters,
                    memory_level=0, memory=Memory(location=None),
                    verbose=0,
                    confounds=None,
                    sample_mask=None,
                    copy=True,
                    dtype=None):
    """Extract representative time series using given mask.

    Parameters
    ----------
    imgs : 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    For all other parameters refer to NiftiMasker documentation.

    Returns
    -------
    signals : 2D numpy array
        Signals extracted using the provided mask. It is a scikit-learn
        friendly 2D array with shape n_sample x n_features.

    """
    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)

    # Check whether resampling is truly necessary. If so, crop mask
    # as small as possible in order to speed up the process

    if not _check_same_fov(imgs, mask_img_):
        parameters = copy_object(parameters)
        # now we can crop
        mask_img_ = image.crop_img(mask_img_, copy=False)
        parameters['target_shape'] = mask_img_.shape
        parameters['target_affine'] = mask_img_.affine

    data, affine = filter_and_extract(imgs, _ExtractionFunctor(mask_img_),
                                      parameters,
                                      memory_level=memory_level,
                                      memory=memory,
                                      verbose=verbose,
                                      confounds=confounds,
                                      sample_mask=sample_mask,
                                      copy=copy,
                                      dtype=dtype)

    # For _later_: missing value removal or imputing of missing data
    # (i.e. we want to get rid of NaNs, if smoothing must be done
    # earlier)
    # Optionally: 'doctor_nan', remove voxels with NaNs, other option
    # for later: some form of imputation
    return data

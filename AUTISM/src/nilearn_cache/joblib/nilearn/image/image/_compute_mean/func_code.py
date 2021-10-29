# first line: 470
def _compute_mean(imgs, target_affine=None,
                  target_shape=None, smooth=False):
    from . import resampling
    input_repr = _repr_niimgs(imgs, shorten=True)

    imgs = check_niimg(imgs)
    mean_data = _safe_get_data(imgs)
    affine = imgs.affine
    # Free memory ASAP
    del imgs
    if mean_data.ndim not in (3, 4):
        raise ValueError('Computation expects 3D or 4D '
                         'images, but %i dimensions were given (%s)'
                         % (mean_data.ndim, input_repr))
    if mean_data.ndim == 4:
        mean_data = mean_data.mean(axis=-1)
    else:
        mean_data = mean_data.copy()
    mean_data = resampling.resample_img(
        nibabel.Nifti1Image(mean_data, affine),
        target_affine=target_affine, target_shape=target_shape,
        copy=False)
    affine = mean_data.affine
    mean_data = get_data(mean_data)

    if smooth:
        nan_mask = np.isnan(mean_data)
        mean_data = _smooth_array(mean_data, affine=np.eye(4), fwhm=smooth,
                                  ensure_finite=True, copy=False)
        mean_data[nan_mask] = np.nan

    return mean_data, affine

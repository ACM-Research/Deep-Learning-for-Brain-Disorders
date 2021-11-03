# first line: 297
def compute_multi_epi_mask(epi_imgs, lower_cutoff=0.2, upper_cutoff=0.85,
                           connected=True, opening=2, threshold=0.5,
                           target_affine=None, target_shape=None,
                           exclude_zeros=False, n_jobs=1,
                           memory=None, verbose=0):
    """ Compute a common mask for several sessions or subjects of fMRI data.

    Uses the mask-finding algorithms to extract masks for each session
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    epi_imgs: list of Niimg-like objects
        See http://nilearn.github.io/manipulating_images/input_output.html
        A list of arrays, each item being a subject or a session.
        3D and 4D images are accepted.
        If 3D images is given, we suggest to use the mean image of each
        session

    threshold: float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: bool, optional
        if connected is True, only the largest connect component is kept.

    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    memory: instance of joblib.Memory or string
        Used to cache the function call.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    Returns
    -------
    mask : 3D nibabel.Nifti1Image
        The brain mask.
    """
    if len(epi_imgs) == 0:
        raise TypeError('An empty object - %r - was passed instead of an '
                        'image or a list of images' % epi_imgs)
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_epi_mask)(epi_img,
                                  lower_cutoff=lower_cutoff,
                                  upper_cutoff=upper_cutoff,
                                  connected=connected,
                                  opening=opening,
                                  exclude_zeros=exclude_zeros,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  memory=memory)
        for epi_img in epi_imgs)

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask

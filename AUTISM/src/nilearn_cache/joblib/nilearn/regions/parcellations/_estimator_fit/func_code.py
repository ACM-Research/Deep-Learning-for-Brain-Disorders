# first line: 19
def _estimator_fit(data, estimator, method=None):
    """Estimator to fit on the data matrix

    Parameters
    ----------
    data : numpy array
        Data matrix.

    estimator : instance of estimator from sklearn
        MiniBatchKMeans or AgglomerativeClustering.

    method : str, {'kmeans', 'ward', 'complete', 'average', 'rena'}, optional
        A method to choose between for brain parcellations.

    Returns
    -------
    labels_ : numpy.ndarray
        labels_ estimated from estimator.

    """
    if method == 'rena':
        rena = ReNA(mask_img=estimator.mask_img,
                    n_clusters=estimator.n_clusters,
                    scaling=estimator.scaling,
                    n_iter=estimator.n_iter,
                    threshold=estimator.threshold,
                    memory=estimator.memory,
                    memory_level=estimator.memory_level,
                    verbose=estimator.verbose)
        rena.fit(data)
        labels_ = rena.labels_

    else:
        estimator = clone(estimator)
        estimator.fit(data.T)
        labels_ = estimator.labels_

    return labels_

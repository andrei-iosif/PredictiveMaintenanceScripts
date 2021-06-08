import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def get_principal_components(X, explained_variance=0.99, debug=False, seed=0):
    pca = PCA(explained_variance, svd_solver='full', random_state=seed)
    pca = pca.fit(X)

    if debug:
        explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
        pca_dims = np.argmax(explained_variance_cumsum >= explained_variance) + 1
        print(
            f"Can reduce from {X.shape[1]} to {pca_dims} dimensions while retaining {explained_variance}% of variance.")

        plt.plot(explained_variance_cumsum)
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')

        plt.figure()
        plt.bar(range(pca.components_.shape[0]), pca.explained_variance_ratio_)
        plt.xlabel('PC')
        plt.ylabel('Explained variance')
    return pca


def dimensionality_reduction(X, pca):
    return pca.transform(X)

import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def get_principal_components(X, explained_variance=0.99, debug=False, seed=0, output_path_plots=None):
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
        if output_path_plots:
            plt.savefig(os.path.join(output_path_plots, 'pca_0.png'), format='png', dpi=300)

        plt.figure()
        plt.bar(range(pca.components_.shape[0]), pca.explained_variance_ratio_)
        plt.xlabel('PC')
        plt.ylabel('Explained variance')
        if output_path_plots:
            plt.savefig(os.path.join(output_path_plots, 'pca_1.png'), format='png', dpi=300)

    return pca


def dimensionality_reduction(X, pca):
    return pca.transform(X)

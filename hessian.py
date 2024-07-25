from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import convolve
from scipy.ndimage import rotate
from utils import *

### IMPORTANT ###
# EN COMMENTAIRE SE TROUVE(NT) LA OU LES MAUTRES METHODES, DES PARAMETRES PEUVENT ETRE A MODIFIER ENTRE LES METHODES

'''
def compute_hessian_matrix(nd_array, sigma=1, scale=True):
    """
    Computes the hessian matrix for an nd_array.
    This can be used to detect vesselness as well as other features.

    In 3D the first derivative will contain three directional gradients at each index:
    [ gx,  gy,  gz ]

    The Hessian matrix at each index will then be equal to the second derivative:
    [ gxx, gxy, gxz]
    [ gyx, gyy, gyz]
    [ gzx, gzy, gzz]

    The Hessian matrix is symmetrical, so gyx == gxy, gzx == gxz, and gyz == gzy.

    :param nd_array: n-dimensional array from which to compute the hessian matrix.
    :param sigma: gaussian smoothing to perform on the array.
    :param scale: if True, the hessian elements will be scaled by sigma squared.
    :return: hessian array of shape (..., ndim, ndim)
    """
    ndim = nd_array.ndim

    # smooth the nd_array
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma)

    # compute the first order gradients
    gradient_list = np.gradient(smoothed)

    # compute the hessian elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]

    if sigma > 0 and scale:
        # scale the elements of the hessian matrix
        hessian_elements = [(sigma ** 2) * element for element in hessian_elements]

    # create hessian matrix from hessian elements
    hessian_full = [[None] * ndim] * ndim

    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian_full[ax0][ax1] = element
        if ax0 != ax1:
            hessian_full[ax1][ax0] = element

    hessian_rows = list()
    for row in hessian_full:
        hessian_rows.append(np.stack(row, axis=-1))

    hessian = np.stack(hessian_rows, axis=-2)
    return hessian
'''


def compute_hessian_spline_3d(nd_array, sigma=1):
    #On applique un filtre gaussien en premier lieu afin de rendre l'image moins bruitée
    nd_array = ndi.gaussian_filter(nd_array, sigma=sigma)

    
    #On défini la matrice B comme dans l'article
    B = np.array([
        [-1, 1, -1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [8, 4, 2, 1]
    ])
        
    #On inverse B, on "aplati" les tableaux avec reshape puis on calcul les produits extérieurs
    #u, k, d etc sont les éléments nécessaires à cette méthode présentés dans l'article
    M = np.linalg.inv(B)
    u = np.array([0.125, 0.25, 0.5, 1]).reshape(-1, 1)
    up = np.array([0.75, 1, 1, 0]).reshape(-1, 1)
    d = np.dot(up.T, M).flatten()
    k = np.dot(u.T, M).flatten()
    
    def conv3d(nd_array, kernel1, kernel2):
        return convolve(convolve(nd_array, kernel1[:, None, None], mode='same'), kernel2[None, :, None], mode='same')

    #On fait une rotation de l'image afin de récupérer l'information contenue dans l'axe z <
    #Optimisation possible (à mon avis), plutôt que faire tourner l'image entière, faire tourner les coupes nécéssaires seulement
    image_rotated = rotate(nd_array, 90, axes=(0, 2), reshape=False)

    Hxx_rotated = -conv3d(image_rotated, d, k)
    Hxy_rotated = -conv3d(image_rotated, k, d)
    Hxz_rotated = -conv3d(image_rotated, k, d)

    Hxx = rotate(Hxx_rotated, -90, axes=(0, 2), reshape=False)
    Hxy = rotate(Hxy_rotated, -90, axes=(0, 2), reshape=False)
    Hxz = rotate(Hxz_rotated, -90, axes=(0, 2), reshape=False)

    Hyy = -conv3d(nd_array, d, k)
    Hyz = -conv3d(nd_array, k, d)
    Hzz = -conv3d(nd_array, d, k)
    
    #Initialisation et remplissage de la hessienne
    hessian = np.zeros((nd_array.shape[0], nd_array.shape[1], nd_array.shape[2], 3, 3))
    hessian[..., 0, 0] = Hxx
    hessian[..., 0, 1] = Hxy
    hessian[..., 0, 2] = Hxz
    hessian[..., 1, 0] = Hxy
    hessian[..., 1, 1] = Hyy
    hessian[..., 1, 2] = Hyz
    hessian[..., 2, 0] = Hxz
    hessian[..., 2, 1] = Hyz
    hessian[..., 2, 2] = Hzz

    return hessian


def absolute_hessian_eigenvalues(nd_array, sigma=2, scale=True):
    #Ici remplacer compute_hessian_spline_3d par la fonction qui correspond à la méthode souhaitée
    return absolute_eigenvaluesh(compute_hessian_spline_3d(nd_array, sigma=2))

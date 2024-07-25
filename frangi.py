import numpy as np
import nibabel as nib
from utils import divide_nonzero
from hessian import absolute_hessian_eigenvalues

### IMPORTANT ###
#Base de code trouvée ici: https://github.com/ellisdg/frangi3d

#Paramètres alpha et beta fixés à 0.5 par Frangi
#Voir https://www.researchgate.net/publication/2388170_Multiscale_Vessel_Enhancement_Filtering
def frangi(nd_array, scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=100, black_vessels=True):
    if not nd_array.ndim == 3:
        raise(ValueError("Only 3 dimensions is currently supported"))

    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(sigmas.shape + nd_array.shape)

    for i, sigma in enumerate(sigmas):
        eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=True)
        filtered_array[i] = compute_vesselness(*eigenvalues, alpha=alpha, beta=beta, c=frangi_c, black_white=black_vessels)

    return np.max(filtered_array, axis=0)

#Les fonctions suivantes utilisent les formules de calcul de vesselness présentées dans le livre 
#Voir https://www.researchgate.net/publication/2388170_Multiscale_Vessel_Enhancement_Filtering
def compute_measures(eigen1, eigen2, eigen3):
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S

def compute_plate_like_factor(Ra, alpha):
    return 1 - np.exp(np.negative(np.square(Ra)) / (2 * np.square(alpha)))


def compute_blob_like_factor(Rb, beta):
    return np.exp(np.negative(np.square(Rb) / (2 * np.square(beta))))


def compute_background_factor(S, c):
    return 1 - np.exp(np.negative(np.square(S)) / (2 * np.square(c)))


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)
    plate = compute_plate_like_factor(Ra, alpha)
    blob = compute_blob_like_factor(Rb, beta)
    background = compute_background_factor(S, c)
    return filter_out_background(plate * blob * background, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    if black_white:
        voxel_data[eigen2 < 0] = 0
        voxel_data[eigen3 < 0] = 0
    else:
        voxel_data[eigen2 > 0] = 0
        voxel_data[eigen3 > 0] = 0
    voxel_data[np.isnan(voxel_data)] = 0
    return voxel_data


if __name__ == "__main__":
    #Charger le fichier .nii
    nii_file = 'NewData_B.nii'  #Remplace par le chemin du fichier à filtrer
    nii_img = nib.load(nii_file)
    nii_data = nii_img.get_fdata()
    nii_final_data = np.squeeze(nii_data, 3) #Cette ligne peut, dans certains cas, poser problème et peut être commentée pour le résoudre
    #frangi_img2 = nib.Nifti1Image(nii_final_data, nii_img.affine)
    #nib.save(frangi_img2, 'substrack2.nii')
    
    # Appliquer la fonction Frangi
    frangi_result = frangi(nii_final_data, scale_range=(1, 10), scale_step=1, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True)

    # Sauvegarder le résultat dans un nouveau fichier .nii
    frangi_img = nib.Nifti1Image(frangi_result, nii_img.affine)
    nib.save(frangi_img, 'NewData_B_Hast_TESTNOGAUSS.nii')
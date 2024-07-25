import nibabel as nib
import numpy as np
from skimage.metrics import mean_squared_error

def load_image(file_path):
    nifti_image = nib.load(file_path)
    return nifti_image.get_fdata()

def toBinaire(image, seuil=0):
    #Seuil sert à savoir de quelle couleur mettre le pixel
    return (image > seuil).astype(np.uint8) 

def evaluation_perf(ground_truth, prediction):
    #Evaluation des perf
    mse = mean_squared_error(ground_truth, prediction)

    return mse

def main():
    verite_terrain = load_image('NewData_GT_B.nii')    
    verite_terrain_final = np.squeeze(verite_terrain, 3)
    image_filtree = load_image('NewData_B_ClassiqueSigma.nii')
    
    print(verite_terrain_final.shape, image_filtree.shape)

    #Passage au binaire
    verite_terrain_binaire = toBinaire(verite_terrain_final)
    image_filtree_binaire = toBinaire(image_filtree)

    #Verification des shapes
    if verite_terrain_binaire.shape != image_filtree_binaire.shape:
        raise ValueError("The shape of the ground truth and prediction images must match.")

    #Evaluation de la performance du filtre
    mse = evaluation_perf(verite_terrain_binaire, image_filtree_binaire)

    #Print the results
    print(f'Accuracy: {mse:.4f}')

main()
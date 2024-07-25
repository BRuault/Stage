import numpy as np

def divide_nonzero(array1, array2):
    #Divise 2 tableaux, retourne 0 en cas de division par 0
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def absolute_eigenvaluesh(nd_array):
    #Calcul les valeurs propres de la hessienne
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    return [np.squeeze(eigenvalue, axis=-1)
            for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)]


def sortbyabs(a, axis=0):
    # Création des indices pour chaque dimension du tableau
    index = np.ix_(*[np.arange(i) for i in a.shape])
    
    # Tri des indices le long de l'axe spécifié
    sorted_indices = np.argsort(a, axis=axis)
    
    # Remplacement des indices de l'axe spécifié par ceux triés
    index = list(index)
    index[axis] = sorted_indices
    
    # Conversion en tuple car les indices doivent être un tuple pour l'indexation avancée
    index = tuple(index)
    
    # Retourne le tableau trié en utilisant les indices triés
    return a[index]

import os
import numpy as np
from tqdm import tqdm
from utils_new import create_directory, load_data_df, cosine_similarity
from datetime import datetime
from apply_vector import init_decoder
from utils_test import ang_to_str, find_matrix, tranform_to_img, read_txt
from utils_pca import get_pca_vectors, kabsch

def test_eigenvalues(persons, df):
    max_value_d = []

    for person1_idx in range(len(persons)):
        for person2_str in persons[person1_idx + 1:]:

            person1 = [persons[person1_idx]]
            person2 = [person2_str]
            eigenvectors1, eigenvalues1 = get_pca_vectors(df, person1)
            eigenvectors2, eigenvalues2 = get_pca_vectors(df, person2)
            max_value = 0
            max_value1 = 0
            max_value2 = 0
            max_idx = 0
            length = len(eigenvalues1)
            for j in range(length):
                diff = np.abs((eigenvalues1[j]-eigenvalues2[j])) #/eigenvalues[j]
                if diff > max_value:
                    max_idx = j
                    max_value1 = eigenvalues1[j]
                    max_value2 = eigenvalues2[j]
                    max_value = diff
            max_value_d.append((max_value, max_value1, max_value2, max_idx))

            print(f"Ended idx{person1_idx}")
    print(max_value_d)



def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


'''def kabsch(P, Q):
    """Izračuna optimalno rotacijsko matriko med dvema nizoma vektorjev P in Q."""
    # Zagon SVD na A
    A = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(A)
    return np.dot(Wt.T, V.T)'''

# 1. Ustvari dva naključna sistema baznih vektorjev in jih normaliziraj
def generate_orthonormal_vectors(n):
    """Generira n ortonormalnih vektorjev"""
    return np.linalg.qr(np.random.rand(n, n))[0]

def rotation_matrix_z(theta):
    """Ustvari rotacijsko matrico za rotacijo okoli osi Z za kot theta (v radianih)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotation_matrix_x(theta):
    """Ustvari rotacijsko matrico za rotacijo okoli osi X za kot theta (v radianih)."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def main():
    A = np.array([
    [1, 1, 1],
    [5, 2, 4],
    [-1, 1, 3]
    ])
    b = np.array([2, 13, 6])

    # Rešite sistem enačb Ax = b
    x = np.linalg.solve(A, b)
    print(x)
    vector1 = np.array([1, 1, 0])
    
    eigenvectors1 = generate_orthonormal_vectors(3)
    R_test = rotation_matrix_z(np.pi/4)
    R_test_x = rotation_matrix_x(np.pi/4)
    Transformation = np.dot(R_test, R_test_x)
    eigenvectors2 = np.dot(eigenvectors1, Transformation.T)

    # 2. Pridobi rotacijsko matriko R
    R = kabsch(eigenvectors1, eigenvectors2)
    if np.allclose(R,Transformation):
        print("Ujemata se!")
    else:
        print(f" Transformation: {Transformation} \n R: {R}")
        raise ValueError(f"Ne ujemata se! ")

    vector2 = np.dot(R_test, vector1)

    print(f"Cosine similarity: {cosine_similarity(vector1, vector2)}")



    '''
    cos_sim = []
    for i in range(len(eigenvectors1)):
        cos_sim.append(cosine_similarity(eigenvectors2[i], eigenvectors1[i]))
    print(f"Cos sim avg: {np.mean(cos_sim)}")'''

    
    '''P = np.random.rand(10, 3)  # 10 točk v 3D prostoru

    # Izberi naključni kot in ustvari rotacijsko matriko
    theta = np.radians(45)  # Rotacija za 45 stopinj
    R_true = rotation_matrix_3d(theta)

    # Rotiraj točke P z rotacijsko matriko R_true, da dobiš Q
    Q = np.dot(P, R_true.T)

    # Izračunaj rotacijsko matriko z Kabschovim algoritmom
    R_estimated = kabsch(P, Q)

    # Preveri, ali se izračunana rotacijska matrika ujema s pravo rotacijo
    print("Prava rotacijska matrika:\n", R_true)
    print("Izračunana rotacijska matrika:\n", R_estimated)

    # Preveri natančnost
    print("Ali je izračunana rotacijska matrika pravilna?", np.allclose(R_true, R_estimated, atol=1e-6))'''
    



if __name__ == '__main__':
  main()
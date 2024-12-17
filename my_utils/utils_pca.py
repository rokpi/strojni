import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from my_utils.utils_new import get_all_angles, ang_to_str

def calculate_tempCentre(df):
    #Keeps all the angles and img_dirs while grouping
    grouped = df.groupby('person').apply(lambda group: pd.Series({
        'tempCentre': np.mean(np.vstack(group['embedding']), axis=0).reshape(1, -1),
        'angles': list(group['angle']),  
        'img_dirs': list(group['img_dir']),
        'embeddings': np.vstack(group['embedding'])
    }))
    return grouped

def centre_and_norm_embedds(df, grouped):
    # Groups tempCentree with original Dataframe
    df = df.merge(grouped[['tempCentre']], left_on='person', right_index=True)
    
    # subtract
    df['centred_embedding'] = df.apply(lambda row: row['embedding'] - row['tempCentre'], axis=1)

    std_devs = grouped['embeddings'].apply(lambda embeddings: np.std(embeddings, axis=0))
    df = df.merge(std_devs.rename('std_dev'), left_on='person', right_index=True)
    df['normalized_embedding'] = df.apply(lambda row: row['centred_embedding']/row['std_dev'], axis = 1)
    
    return np.array(df['normalized_embedding'].tolist())


def save_centroids(centroids, fileDirectory):


    angles = get_all_angles
    for i, neut_cent in tqdm(enumerate(centroids['centred_embedding']), total = len(centroids)):
        for j, pos_cent in enumerate(centroids['centred_embedding']):
            direction_vector = pos_cent - neut_cent
            filename = f"{ang_to_str(angles[i])}_to_{ang_to_str(angles[j])}"
            subfile_path = os.path.join(fileDirectory, filename)
            np.save(subfile_path, direction_vector)
    
    print(f"Centroids saved in direction: {fileDirectory}")
    return fileDirectory

def solve_eigenproblem(C):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx= np.argsort(eigenvalues, axis=0)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def get_pca_vectors(df, people = None):
    #df from extract embedds
    df_copy = df.copy()
    if people is not None:
        df_copy = df_copy[df_copy['person'].isin(people)]
        
    grouped = calculate_tempCentre(df_copy)
    norm_embedds = centre_and_norm_embedds(df_copy, grouped)

    N = norm_embedds.shape[0]
    norm_embedds = norm_embedds.reshape(-1, norm_embedds.shape[2])
    covarianceMatrix = 1/(N * (N-1) - 1) * np.dot(norm_embedds.T, norm_embedds)
    eigenvalues, eigenvectors = solve_eigenproblem(covarianceMatrix)
    return eigenvectors, eigenvalues

def define_true_vector(vector, pca_vector):
    vector = np.subtract(vector, np.dot(np.dot(vector.T, pca_vector), pca_vector))
    return vector


def kabsch(A, B):

    A_mean = np.mean(A, axis=0)
    B_mean = np.mean(B, axis=0)

    A_centered = A - A_mean
    B_centered = B - B_mean

    H = np.dot(A_centered.T, B_centered)

    U, _, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R
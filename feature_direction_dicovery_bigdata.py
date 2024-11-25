import os
import argparse
from utils import get_all_filepaths
from keras_vggface.vggface import VGGFace
import numpy as np
from pathlib import Path
from datetime import datetime


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=r'C:\Users\rokpi\Downloads\big_data\20240823_082057_vgg-vgg', type=str, help='Path to the directory of .npz files.')
    parser.add_argument('--file', default= 'npy', type= str, help='Type of file you want to save as.')
    parser.add_argument('--vector_num', default=10, type= int, help='Number of relevant vectors to export.')
    parser.add_argument('--out_dir', default=r'C:\Users\rokpi\Downloads\feature_direction_discovery', type=str, help='Output directory where the embeddings will be saved.')
    args = parser.parse_args()
    return args

def centre_respect_to_donor(personsEmbedd):
    tempCentre = np.mean(personsEmbedd, axis=0).reshape(1, -1)
    centreDonor = personsEmbedd - tempCentre  # Broadcasting
    return centreDonor 

def solve_eigenproblem(C):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx= np.argsort(eigenvalues, axis=0)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def main(args):
    in_directory = args.input_dir
    all_embeddings_centred = []
    for subdir in os.listdir(in_directory):
        subdir_path = os.path.join(in_directory, subdir)
        data = np.load(subdir_path)
        #print("Available keys in the .npz file: ", data.keys())

        if 'embeddings' in data:
            personsEmbedd = data['embeddings']
        else:
            raise ValueError("Key 'embeddings' not found in the .npz file!")


        #print(f"Starting direction discovery for file {subdir} ...")
        print(f"personEmbed shape {personsEmbedd.shape}")
        centreDonor = centre_respect_to_donor(personsEmbedd=personsEmbedd)
        #print(centreDonor.shape)
        all_embeddings_centred.append(centreDonor)

    # converts list into matrix
    all_embeddings_matrix = np.vstack(all_embeddings_centred)
    print(f"Shape of all_embeddings_matrix: {all_embeddings_matrix.shape}")
    all_embeddings_matrix = all_embeddings_matrix - np.mean(all_embeddings_matrix, axis=0)
    N = all_embeddings_matrix.shape[0]
    covarianceMatrix = 1/(N * (N-1) - 1) * np.dot(all_embeddings_matrix.T, all_embeddings_matrix)
    print("Covariance calculated...")
    eigenvalues, eigenvectors = solve_eigenproblem(covarianceMatrix)

    selected_vectors = eigenvectors[:, :args.vector_num]

    #save the vectors into given directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.file == 'csv':
        filename = f"{current_date}_sel_vectors.csv"
        out_file = os.path.join(args.out_dir, filename)
        np.savetxt(out_file, selected_vectors, delimiter=',', fmt='%f', header='Vector1,Vector2,...')
    elif args.file == 'npy':
        filename = f"{current_date}_sel_vectors.npy"
        out_file = os.path.join(args.out_dir, filename)
        np.save(out_file, selected_vectors)

    print(f"Selected vectors saved to {out_file}")


if __name__ == '__main__':
  args = argparser()
  main(args)



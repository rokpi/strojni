import numpy as np
from scipy.optimize import minimize
import argparse
from utils_new import load_data_df, divide_dataframe_one
import pandas as pd
from tqdm import tqdm

def cosine_similarity(a, b):
    """Izračuna kosinusno podobnost med vektorjema a in b."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def loss_function(R_vector, embedding, target, dims):
    """Izračuna izgubo (kotno razdaljo) med rotiranim vektorjem in ciljnim vektorjem."""
    R = R_vector.reshape((dims, dims))
    rotated_embedding = np.dot(R, embedding)
    return 1 - cosine_similarity(rotated_embedding, target)

def train(data,dim, num_iterations=100):
    """Uči rotacijsko matriko na podanih podatkih."""
    # Inicializacija rotacijske matrike
    R = np.random.randn(dim, dim)
    R = R / np.linalg.norm(R, axis=0)

    for _ in tqdm(range(num_iterations), total = num_iterations):
        for embedding, target in data:
            embedding = embedding[:dim]
            target = target [:dim]

            R_vector = R.ravel()
            result = minimize(loss_function, R_vector, args=(embedding, target, dim), method='L-BFGS-B')
            R = result.x.reshape((dim, dim))

    return R

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedds_dir', default=r'/home/rokp/test/models/mtcnn/resnet-resnet/resnet-resnetmtcnn_imgs_HR_128.npz', type=str, help='Path to the models')
    args = parser.parse_args()
    return args

def main(args):
    # Filtriramo samo kote 0 in 30
    angles = [0, 30]
    df = load_data_df('/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnetconv4_3_3x3_imgs_HR_128.npz')
    df = divide_dataframe_one(df, 10)
    df = df[df['angle'].isin(angles)]


    # Skupinimo po osebi in ustvarimo tuple (embedding_angle_0, embedding_angle_30)
    grouped = df.groupby('person')
    print(grouped)
    print("začenjam..")
    R = train(grouped, 100)
    print(R)
    np.save('/home/rokp/test/arcface/loli.npy', R)

import matplotlib.pyplot as plt



if __name__ == '__main__':
  #args = argparser()
  #main(args)     
    plt.plot([0, 1], [0, 1])
    plt.title("Test graf")

    # Shranite graf kot .jpg datoteko
    plt.savefig("test/ROC/test_graf.jpg", format="jpg")
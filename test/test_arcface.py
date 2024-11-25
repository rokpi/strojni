import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_new import load_data_df, divide_dataframe_one
from tqdm import tqdm
import pandas as pd

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedds_dir', default=r'/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnetconv4_3_3x3_imgs_images_mtcnn_data.npz', type=str, help='Path to the models')
    parser.add_argument('--base', default=0, type=int, help='Viewpoint of images to transform to.')
    parser.add_argument('--goal', default='neut', type=str, help='Whether the base is the neut or pos centroid.')
    parser.add_argument('--txt', default='/home/rokp/test/launch_train_arcface.txt', type=str, help='Directory to txt file with people to use for training.')    
    parser.add_argument('--txt_test', default='/home/rokp/test/launch_test_arcface.txt', type=str, help='Directory to txt file with people to use for testing.')
    args = parser.parse_args()
    return args

def l2_normalize(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def kabsch_algorithm(A, B):

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

def align_embeddings_and_calculate_rotation(embeddings_person, neut, positive):
    embeddings1 = [item['embedding'] for item in embeddings_person if item['angle'] == neut]
    embeddings2 = [item['embedding'] for item in embeddings_person if item['angle'] == positive]


    if len(embeddings1) < 3 or len(embeddings2) < 3:
        raise ValueError("For Kabsch algorithm, we need at least three embeddings for each angle!")

    A = np.array(embeddings1)
    B = np.array(embeddings2)

    # Flatten and transpose
    A = A.reshape(A.shape[0], -1)
    B = B.reshape(B.shape[0], -1)

    # Apply Kabsch algorithm to find the optimal rotation matrix
    R = kabsch_algorithm(A, B)

    return R

def read_txt(txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()
    people = [line.strip() for line in lines if line.strip()]
    print(f"Persons: {people}.")
    return people

def cosine_similarity(embedding1, embedding2):
    
    # Pretvori matrike v vektorje
    embedding1 = np.squeeze(embedding1).flatten()
    embedding2 = np.squeeze(embedding2).flatten()

    #print(f"Flattened embedding 1: {embedding1}")
    #print(f"Flattened embedding 2: {embedding2}")
    
    # Izračunaj kozinsko podobnost med dvema vektorjema
    cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cos_sim


def check_rotation_matrix(R):
    should_be_identity = np.dot(R.T, R)
    identity = np.eye(R.shape[0])
    orthogonality_check = np.allclose(should_be_identity, identity)

    determinant = np.linalg.det(R)
    det_check = np.isclose(determinant, 1.0)

    return orthogonality_check, det_check

def get_rotation(R, embedd_person_test, neut, pos):
    embeddings1 = [item['embedding'] for item in embedd_person_test if item['angle'] == neut]
    embeddings2 = [item['embedding'] for item in embedd_person_test if item['angle'] == pos]

    transformed_embeddings1 = [np.dot(embedding, R) for embedding in embeddings1]

    cosine_similarities = [cosine_similarity(transformed_embedding, embedding2)
                           for transformed_embedding, embedding2 in zip(transformed_embeddings1, embeddings2)]

    average_cosine_similarity = np.mean(cosine_similarities)
    print(f"Average cosine similarity between transformed embeddings1 and embeddings2: {average_cosine_similarity}")

    orthogonal, determinant_correct = check_rotation_matrix(R)
    print(f"Is orthogonal: {orthogonal}, Determinant = 1: {determinant_correct}")


def get_embeddings_with_angles(df):
    embeddings_with_angles = df[['embedding', 'angle']].apply(
        lambda row: {'embedding': l2_normalize(np.array(row['embedding']).astype(float)), 'angle': row['angle']}, axis=1)
    return embeddings_with_angles.tolist()

def optimal_rotation_matrix(p1, p2):
    """
    Izračunaj optimalno rotacijsko matriko, ki preslika p1 v p2 v n-dimenzionalnem prostoru.
    Metoda uporablja SVD, da zagotovi optimalno rotacijsko transformacijo brez strižnih ali razteznih operacij.
    """

    p1 = p1.reshape(-1, 1).T
    p2 = p2.reshape(-1, 1).T

    # Najprej normaliziraj oba vektorja
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)

    p1_norm = np.squeeze(p1_norm).flatten()
    p2_norm = np.squeeze(p2_norm).flatten()
    # Izračunaj matriko korelacije
    H = np.dot(p1_norm[:, np.newaxis], p2_norm[np.newaxis, :])
    
    # Singularni vrednostni razcep (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Izračun optimalne rotacijske matrike (brez strižnih transformacij)
    R = np.dot(Vt.T, U.T)
    
    # Preveri, ali je rotacijska matrika ustrezna (determinanta mora biti 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R


def rotate_data(df):

    df_base = df[df['angle'] == 0].copy()
    
    # Vzamemo prvi embedding kot referenčni bazni vektor
    base_vector = np.array(df_base.iloc[0]['embedding'])

    # Izračunamo rotacijsko matriko za vsak embedding v df_base
    df_base['rot'] = df_base['embedding'].apply(lambda x: optimal_rotation_matrix(base_vector, np.array(x)))

    # Sedaj uporabimo te rotacije na celotni df
    for person in df_base['person'].unique():
        # Pridobimo rotacijsko matriko za osebo
        base_rot = df_base[df_base['person'] == person].iloc[0]['rot']
        
        # Preprečimo SettingWithCopyWarning z .copy()
        df.loc[df['person'] == person, 'embedding'] = df.loc[df['person'] == person, 'embedding'].copy().apply(
            lambda emb: np.dot(np.array(emb), base_rot.T)
        )

    return df

    

def main(args):
    in_directory = args.embedds_dir
    df = load_data_df(in_directory)
    #people = read_txt(args.txt)
    #df_train = df[df['person'].isin(people)]
    #embedd_person_train = get_embeddings_with_angles(df_train)

    people_test = read_txt(args.txt_test)
    df_test = df[df['person'].isin(people_test)]

    embedd_person_test = get_embeddings_with_angles(df_test)
 
    num_people = [3]#, 5, 10, 20, 100, 150, 200]

    for num in num_people:
        print(f"Začenjam s število ljudi je enako: {str(num)}")
        embedd_person_train = divide_dataframe_one(df, num)
        #embedd_person_train = rotate_data(embedd_person_train)
        embedd_person_train = get_embeddings_with_angles(embedd_person_train)
        R = align_embeddings_and_calculate_rotation(embedd_person_train, 0, 30)
        get_rotation(R, embedd_person_test, 0, 30)

    '''angles = [-30, -15, 0, 15, 30]
    for ang in tqdm(angles, total=len(angles)):
        if args.goal == 'neut':
            get_rotation(embedd_person_test, embedd_person_train, args.base, ang)
        elif args.goal == 'base':
            get_rotation(embedd_person_test, embedd_person_train, ang, args.base)'''


if __name__ == '__main__':
    args = argparser()
    main(args)

import argparse
import os
import numpy as np
import argparse
from tqdm import tqdm
from my_utils.utils_new import create_directory, load_data_df, ang_to_str, divide_dataframe_one, get_all_angles
from my_utils.utils_pca import get_pca_vectors
from datetime import datetime
import pandas as pd

def calculate_tempCentre(df, what):
    #Keeps all the angles and img_dirs while grouping
    grouped = df.groupby('person').apply(lambda group: pd.Series({
        'tempCentre': np.mean(np.vstack(group['embedding']), axis=0).reshape(1, -1),
        what: list(group[what]),  
        'img_dirs': list(group['img_dir']) 
    }))
    return grouped

def centre_embeddings(df, grouped):
    # Groups tempCentree with original Dataframe
    df = df.merge(grouped[['tempCentre']], left_on='person', right_index=True)
    
    # subtract
    df['centred_embedding'] = df.apply(lambda row: row['embedding'] - row['tempCentre'], axis=1)
    
    return df

def calculate_centroids(df, what):
    # Group by angle and calculate centroid
    centroids_by_angle = df.groupby(what)['centred_embedding'].apply(
        lambda embeddings: np.mean(np.vstack(embeddings), axis=0)
    )
    # Converts to dataframe
    centroids_df = pd.DataFrame(centroids_by_angle).reset_index()
    
    return centroids_df

def centre_data_people(df, what):
    grouped = calculate_tempCentre(df, what)
    df = centre_embeddings(df, grouped)
    centroids = calculate_centroids(df,what)
    
    return  centroids

def save_centroids(centroids, fileDirectory, angles):
    for i, neut_cent in tqdm(enumerate(centroids['centred_embedding']), total = len(centroids)):
        for j, pos_cent in enumerate(centroids['centred_embedding']):
            direction_vector = pos_cent - neut_cent
            filename = f"{ang_to_str(angles[i])}_to_{ang_to_str(angles[j])}"
            subfile_path = os.path.join(fileDirectory, filename)
            np.save(subfile_path, direction_vector)
    
    print(f"Centroids saved in direction: {fileDirectory}")
    return fileDirectory

def save_bulk(centroids, fileDirectory, what):
    for index, item in tqdm(centroids.iterrows(), total = len(centroids)):
        direction_vector = item['centred_embedding']
        filename = f"Centroid_{ang_to_str(int(item[what]))}"
        subfile_path = os.path.join(fileDirectory, filename)
        np.save(subfile_path, direction_vector)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default=r'/home/rokp/test/models/mtcnn/vgg-vgg/vgg-vggmtcnn_images_mtcnn.npz', type=str, help='Path to the embeddings')
    parser.add_argument('--out_dir', default=r'/home/rokp/test/bulk', type=str, help='Output directory where the embeddings will be saved.')
    parser.add_argument('--what', default='angle', type=str, help='Which centroid to create.')
    args = parser.parse_args()
    return args

def main(args):
    what = args.what
    in_directory = args.inputs
    df = load_data_df(in_directory)
    print(len(df))
    df = df.iloc[:47000]
    print(len(df))
    #df[what] = df[what].apply(lambda x: x.item())
    #unique_values = df[what].unique()
    #print(unique_values)

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileDirectory = os.path.join(args.out_dir, f"{current_date}_{what}")
    create_directory(fileDirectory)
    eigenvectors, eigenvalues = get_pca_vectors(df)
    np.save(os.path.join(fileDirectory, 'eigenvectors.npy'), eigenvectors)
    np.save(os.path.join(fileDirectory, 'eigenvalues.npy'), eigenvalues)
    
    global_mean = np.mean(np.vstack(df['embedding']), axis=0).reshape(1, -1)
    centroids = centre_data_people(df, what)
    centroids['centred_embedding'] = centroids['centred_embedding'].apply(lambda x: np.array(x))
    #print(centroids)

    #save_centroids(centroids, fileDirectory, unique_angles)
    save_bulk(centroids, fileDirectory, what)
    filename = "global"
    subfile_path = os.path.join(fileDirectory, filename)
    np.save(subfile_path, global_mean)
    return fileDirectory

if __name__ == '__main__':
  args = argparser()
  main(args)
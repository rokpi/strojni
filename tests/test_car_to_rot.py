import argparse
import numpy as np
from utils_new import load_data_df, divide_dataframe_one, cosine_similarity, rotational_to_cartesian ,cartesian_to_rotational, save_rot_npz
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedds_dir', default=r'/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnetconv4_3_3x3_imgs_images_mtcnn_data.npz', type=str, help='Path to the models')
    parser.add_argument('--out_dir', default=r'/home/rokp/test/centroid', type=str, help='Path to the models')

    args = parser.parse_args()
    return args

def read_txt(txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()
    people = [line.strip() for line in lines if line.strip()]
    print(f"Persons: {people}.")
    return people



def main(args):
    save = True
    in_directory = args.embedds_dir

    df = load_data_df(in_directory)
    df = divide_dataframe_one(df, 249)
    similarities = []
    for index, item in tqdm(df.iterrows(), total = len(df)):
        embedding = item['embedding']

        rot_embedd = cartesian_to_rotational(embedding)
        rot_embedd = rotational_to_cartesian(rot_embedd)
        cos_sim = cosine_similarity(embedding, rot_embedd)
        similarities.append(cos_sim)
    print(f"Absolute mean cos sim: {np.mean(np.abs(similarities))}")
    
    if save:
        save_rot_npz(df, args.out_dir, in_directory)

if __name__ == '__main__':
    args = argparser()
    main(args)

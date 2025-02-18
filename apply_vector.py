import tensorflow as tf
import argparse
import os
import numpy as np
import argparse
from tqdm import tqdm
from my_utils.utils import restore_original_image_from_array
from my_utils.utils_new import create_directory, model_to_str, load_data_df, rotational_to_cartesian
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from pathlib import Path
from datetime import datetime
from my_utils.utils_exceptions import decoder_layer, define_tranform, exception_transform



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', default= 30, type=int, help='Viewpoint of images to tranform to.')
    parser.add_argument('--neut', default= -30, type=int, help='Viewpoint of images to tranform from.')
    parser.add_argument('--inputs', default= '/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnetconv4_3_3x3_imgs_images_mtcnn_data.npz', type=str, help='Path to the hdf5 file')
    parser.add_argument('--model', default= '/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnet.conv4_3_3x3.20231029-120710.hdf5', type=str, help='Path to the hdf5 file')
    
    parser.add_argument('--restore', default= False, type=bool, help= 'Dont convert just restore the embedding to img.')    
    parser.add_argument('--method', default= 'centroid', type=str, help='Whether the vector is rotation or centroid.')
    parser.add_argument('--vector', default= '/home/rokp/test/centroid/20240911_085735_00_to_00', type=str, help='Path to .npy file that contains the vector.')
    parser.add_argument('--txt', default='/home/rokp/test/launch_test_arcface.txt', type=str, help='File with people numbers.')
    parser.add_argument('--out_dir', default='/home/rokp/test/transformed_embeddings', type=str, help='Output directory.')
    parser.add_argument('--alpha', default= 2, type=int, help='Value of alpha.')
    args = parser.parse_args()
    return args

def read_txt(txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()  
    people = []
    for line in lines:
        line = line.strip()
        if line:
            people.append(line)
    return people

def ang_to_str(angle):
    string = str(abs(angle)).zfill(2)    
    if angle < 0:
        string = "m" + string
    return string

def find_matrix(neut, pos, directory):
    matrix_str = f"{ang_to_str(neut)}_to_{ang_to_str(pos)}.npy"
    dirs = []    
    for in_file in os.listdir(directory):
        if matrix_str == in_file:
            dirs.append(os.path.join(directory, in_file))
    if len(dirs)>1:
        raise ValueError(f"Too many rotation matrixes in directory with this name: {matrix_str}")
    elif len(dirs) == 0:
        raise ValueError(f"No files in directory with this name: {matrix_str}")
    else:
        return dirs[0]

def init_decoder(model_path):
    #crate the decoder model
    print('Load model ' + model_path)
    encoder_type = os.path.basename(model_path).split('-')[0]
    model = tf.keras.models.load_model(model_path, compile = False)

    #some models require different layer calling
    decoder_model = decoder_layer(model, model_path, encoder_type)
     

    return decoder_model, encoder_type
 

def init_file_matrix(model_path, out_dir, vector, neut, pos):
    matrix_dir = find_matrix(neut, pos, vector)
    vec_filename = os.path.basename(matrix_dir)

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    #dirname = f"{current_date}_{Path(vec_filename).stem}_centroid{os.path.basename(vector)}_{model_to_str(model_path)}"
    dirname = f"{Path(vec_filename).stem}"
    out_directory = os.path.join(out_dir, dirname)
    create_directory(out_directory)
    return matrix_dir, out_directory

def restore(df, decoder_model, encoder_type, model_path, out_directory, method):
    for index, item in df.iterrows():#tqdm(df.iterrows(), total=len(df)):
        embedding = item['embedding']
        img_dir = item['img_dir']
        
        if method == 'rotation':
            embedding = rotational_to_cartesian(embedding)
        embedding = define_tranform(embedding, encoder_type, model_path)
        out_arr = decoder_model.predict(embedding)

        # Save the result
        out_dir = os.path.join(out_directory, img_dir)
        out_img = np.squeeze(out_arr,axis=0)
        Path(os.path.split(out_dir)[0]).mkdir(parents=True, exist_ok=True)
        out_img = restore_original_image_from_array(out_img, encoder_type)
        save_img(out_dir, out_img)    
  

def tranform_and_restore(decoder_model, encoder_type, df, matrix_dir, model_path, out_directory, method, alpha, neut):
    df = df[df['angle'] == neut]
    vector = np.load(matrix_dir)

    for index, item in tqdm(df.iterrows(), total=len(df)):
        embedding = item['embedding']
        img_dir = item['img_dir']

        alphavector = alpha * vector
        trans_embedd = embedding + alphavector
        
        if method == 'rotation':
            trans_embedd = rotational_to_cartesian(trans_embedd)


        trans_embedd = define_tranform(trans_embedd, encoder_type, model_path)
     
        out_arr = decoder_model.predict(trans_embedd)


        # Save the result
        out_dir = os.path.join(out_directory, img_dir)
        out_img = np.squeeze(out_arr,axis=0)
        Path(os.path.split(out_dir)[0]).mkdir(parents=True, exist_ok=True)
        out_img = restore_original_image_from_array(out_img, encoder_type)
        save_img(out_dir, out_img)

    print(f"Files successfuly saved in {out_directory}")
    
def main(args):
    in_directory = args.inputs
    model_path = args.model
    restore_imgs = args.restore
    matrix_dir, out_directory = init_file_matrix(model_path, args.out_dir, args.vector, args.neut, args.pos)


    df = load_data_df(in_directory)
    people = read_txt(args.txt)

    df = df[df['person'].isin(people)]
    #load the vector and the data from file

    decoder_model, encoder_type = init_decoder(model_path)
    if restore_imgs:
        restore_imgs = restore(df, decoder_model, encoder_type, model_path, out_directory, args.neut, args.method)
    else:
        tranform_and_restore(decoder_model, encoder_type, df, matrix_dir, model_path, out_directory, args.method, args.alpha, args.neut)

if __name__ == '__main__':
  args = argparser()
  main(args)
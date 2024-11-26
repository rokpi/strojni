import os
from utils.utils_new import rotational_to_cartesian
from utils.utils_exceptions import define_tranform
from utils.utils import restore_original_image_from_array
from tensorflow.keras.preprocessing.image import save_img
from pathlib import Path
import numpy as np

def find_centroid(centroid_str, directory):
    dirs = []    
    for in_file in os.listdir(directory):
        if centroid_str == in_file:
            dirs.append(os.path.join(directory, in_file))
    if len(dirs)>1:
        raise ValueError(f"Too many rotation matrixes in directory with this name: {centroid_str}")
    elif len(dirs) == 0:
        raise ValueError(f"No files in directory with this name: {centroid_str}")
    else:
        return dirs[0]
    
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

def read_txt(txt_directory):
    with open(txt_directory) as file:
        lines = file.readlines()  
    people = []
    for line in lines:
        line = line.strip()
        if line:
            people.append(line)
    return people

def tranform_to_img(df, vector, alpha, method, encoder_type, decoder_model, model_path, out_directory, num, save = True):
    df_copy = df.copy(deep = True)

    for index, item in df_copy.iterrows():
        embedding = item['embedding']
        img_dir = item['img_dir']

        if img_dir[14:16] not in ['08']:
            continue
        alphavector = alpha * vector
        trans_embedd = embedding + alphavector

        if method == 'rotation':
            trans_embedd = rotational_to_cartesian(trans_embedd)

        df_copy.at[index,'embedding'] = trans_embedd
        trans_embedd = define_tranform(trans_embedd, encoder_type, model_path)
        
        out_arr = decoder_model.predict(trans_embedd)

        # Save the result
        if save:
            out_dir = os.path.join(out_directory, f"{num}_"+img_dir)
            out_img = np.squeeze(out_arr,axis=0)
            Path(os.path.split(out_dir)[0]).mkdir(parents=True, exist_ok=True)
            out_img = restore_original_image_from_array(out_img, encoder_type)
            save_img(out_dir, out_img)     

    return df_copy

import os
import argparse
from utils_new import model_to_str, getAngle, create_directory, get_all_filepaths
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
#from utils import preproc_img
#import tensorflow as tf
#from utils_exceptions import encoder_layer
from datetime import datetime
from models.AdaFace.inference import load_pretrained_model, img_to_embedding
from extract_embedds_ada import process_and_save_embeddings
import torch

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default=r'/home/rokp/test/test/dataset', type=str, help='Path to the hdf5 file')
    parser.add_argument('--inputs', default=r'/home/rokp/test/images_mtcnn_cplfw', type=str, help='A file containing the names of test images, an input image or a directory containing the input images.')
    args = parser.parse_args()
    return args

def check_img_requirements(img_name):
    if img_name[7:9] == '01':                       
        #if img_name[14:16] in ['08']: #, '07'
            face_viewpoint = img_name[10:13]
            if face_viewpoint in ['120', '090', '080','130', '140', '051', '050', '041', '190', '200', '010']:
                return True
            
    return False

def make_directories(input_dir, extra = None):
    filtered_img_list = []
    seen_persons = set()
    num_of_img = 0
    for img in input_dir:    
        filename = os.path.basename(img)
        #person_new = filename[0:3]  
        person_new = filename.split('_')[0]
        if person_new.isdigit():
            if check_img_requirements(filename):
                if extra:
                    filename = os.path.join(extra, filename)
                if person_new not in seen_persons:
                    seen_persons.add(person_new)
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img)})#})#
                    num_of_img = num_of_img + 1
                elif person_new in seen_persons:
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img)})#})#
                    num_of_img = num_of_img + 1
                        
        else:
            raise ValueError("Unexpected person classification (first three letters of basename should be an integer)")  
    print(f"Not as many as expected. Num of imgs: {num_of_img}.")
    print(f"Num of different people: {len(seen_persons)}")
    return filtered_img_list

def no_models():
    in_directory = '/home/rokp/test/images_mtcnn'
    out_dir = '/home/rokp/test/test/dataset/mtcnn/resnet-resnet'
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S") 
    filename = f"{current_date}"
    out_directory = create_directory(os.path.join(out_dir, filename))

    model_path = '/home/rokp/test/models/mtcnn/resnet-resnet/resnet-resnet.mtcnn.conv3_4_1x1_increase.20230127-104503.hdf5'
    dirname = f"{model_to_str(model_path)}_{Path(os.path.basename(in_directory)).stem}.npz"
    out_directory = os.path.join(out_directory, dirname)

    if not os.path.isfile(out_directory):
        print('Load model '+model_path)
        '''model = tf.keras.models.load_model(model_path, compile=False) 
        encoder_type = os.path.basename(model_path).split('-')[0]
        embedding_model = encoder_layer(model, encoder_type, model_path)'''
        #ADA
        model = load_pretrained_model('ir_101')
        feature, norm = model(torch.randn(2,3,112,112))

        #for i, layer in enumerate(model.layers):
        #    print(f"Layer {i}: {layer.name} - {layer.output_shape}")
        #makes the directory
        #current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        #shuffle the img list to get random images    
        #random.shuffle(img_list)  

        img_list = get_all_filepaths(in_directory,'.jpg')

        filtered_img_list = make_directories(img_list, extra = in_directory)
        df = pd.DataFrame(filtered_img_list)
        process_and_save_embeddings(df, model, out_directory)
        '''for item in tqdm(filtered_img_list, total = len(filtered_img_list)):
            img_dir = item['img_dir']
            file_path = os.path.join(in_directory, img_dir)
            if os.path.isfile(file_path):
                # Load and transform the image
                #in_arr = preproc_img(file_path, encoder_type, model.input_shape[1:3])
                
                # Prediction
                #out_arr = embedding_model.predict(in_arr)

                #ADA
                out_arr = img_to_embedding(file_path, model)
                #print(f"Out_arr shape: {out_arr.shape}")


                embedding = np.squeeze(out_arr, axis=0)                          
                item['embedding'] = embedding

            else:
                print("File %s doesn't exist." % file_path)

        df = pd.DataFrame(filtered_img_list)
        np.savez(out_directory, data=df.to_numpy(), columns=df.columns.values)'''
        print(f"File saved in {out_directory}")


def main(args):
    in_directory = args.inputs
    in_model_directory = args.models_dir
    #reads args.src, goes to directory and finds path/paths to file/files
    if os.path.isdir(in_model_directory):
        in_model_directory_list = os.listdir(in_model_directory)
        for preprocessing in in_model_directory_list:
            prep_dir = os.path.join(in_model_directory, preprocessing)
            if os.path.isdir(prep_dir):
                prep_dir_list = os.listdir(prep_dir)
                for model_subdir in prep_dir_list:
                    model_dir = os.path.join(prep_dir, model_subdir)
                    if os.path.isdir(model_dir):
                        print(f"Now processing in directory: {model_dir}")
                        model_files = [f for f in os.listdir(model_dir) if f.endswith('.hdf5')]
                        for model_file in model_files:
                            model_path = os.path.join(model_dir,model_file)

                            dirname = f"{model_to_str(model_path)}_{Path(os.path.basename(in_directory)).stem}.npz"
                            out_directory = os.path.join(model_dir, dirname)
                            if not os.path.isfile(out_directory):
                                print('Load model '+model_path)
                                model = tf.keras.models.load_model(model_path, compile=False) 
                                #for i, layer in enumerate(model.layers):
                                #    print(f"Layer {i}: {layer.name} - {layer.output_shape}")
                                encoder_type = os.path.basename(model_path).split('-')[0]

                                embedding_model = encoder_layer(model, encoder_type, model_path)

                                #makes the directory
                                #current_date = datetime.now().strftime("%Y%m%d_%H%M%S")


                                img_list = get_all_filepaths(in_directory,'.jpg')

                                #shuffle the img list to get random images    
                                #random.shuffle(img_list)             


                                filtered_img_list = make_directories(img_list)
                                for item in tqdm(filtered_img_list, total = len(filtered_img_list)):
                                    img_dir = item['img_dir']
                                    file_path = os.path.join(in_directory, img_dir)
                                    if os.path.isfile(file_path):
                                        # Load and transform the image
                                        in_arr = preproc_img(file_path, encoder_type, model.input_shape[1:3])
                                        
                                        # Prediction
                                        out_arr = embedding_model.predict(in_arr)
                                        #print(f"Out_arr shape: {out_arr.shape}")
                                        embedding = np.squeeze(out_arr, axis=0)                          
                                        item['embedding'] = embedding

                                    else:
                                        print("File %s doesn't exist." % file_path)

                                df = pd.DataFrame(filtered_img_list)
                                np.savez(out_directory, data=df.to_numpy(), columns=df.columns.values)
                                print(f"File saved in {out_directory}")
                            
    
if __name__ == '__main__':
  #args = argparser()
  #main(args)
  no_models()


    


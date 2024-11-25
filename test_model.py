import os
import argparse
from utils_new import model_to_str, getAngle
from utils import preproc_img, get_all_filepaths
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import save_img
from utils import restore_original_image_from_array
from datetime import datetime

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnet.conv4_3_3x3.20231029-120710.hdf5', type=str, help='Path to the hdf5 file')
    parser.add_argument('--inputs', default=r'/home/rokp/test/images_extract/txt_20240910_154216_imgs', type=str, help='A file containing the names of test images, an input image or a directory containing the input images.')
    parser.add_argument('--out_dir', default=r'/home/rokp/test/test/test', type=str, help='Out.')
    args = parser.parse_args()
    return args

def check_img_requirements(img_name):
    if img_name[7:9] == '01':                       
        if img_name[14:16] in ['08']: # '07',
            face_viewpoint = img_name[10:13]
            if face_viewpoint in ['130', '140', '051', '050', '041']:
                return True
            
    return False

def make_directories(input_dir):
    filtered_img_list = []
    seen_persons = set()
    num_of_img = 0
    for img in input_dir:    
        filename = os.path.basename(img)
        person_new = filename[0:3]  
        if person_new.isdigit():
            if check_img_requirements(filename):
                if person_new not in seen_persons:
                    seen_persons.add(person_new)
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img)})
                    num_of_img = num_of_img + 1
                elif person_new in seen_persons:
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img)}) 
                    num_of_img = num_of_img + 1
                        
        else:
            raise ValueError("Unexpected person classification (first three letters of basename should be an integer)")  
    print(f"Not as many as expected. Num of imgs: {num_of_img}.")
    print(f"Num of different people: {len(seen_persons)}")
    return filtered_img_list


def main(args):
  in_directory = args.inputs
  model_path = args.model
  out_dir = args.out_dir
  current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
  dirname = f"{model_to_str(model_path)}_ {current_date}_imgs_{Path(os.path.basename(in_directory)).stem}"
  out_directory = os.path.join(out_dir, dirname)
  model = tf.keras.models.load_model(model_path, compile=False) 

  encoder_type = os.path.basename(model_path).split('-')[0]

  img_list = get_all_filepaths(in_directory,'.jpg')
  #filtered_img_list = make_directories(img_list)
  filtered_img_list = img_list

  for item in tqdm(filtered_img_list, total = len(filtered_img_list)):
      img_dir = item#item['img_dir']
      file_path = os.path.join(in_directory, img_dir)
      if os.path.isfile(file_path):
          # Load and transform the image
          in_arr = preproc_img(file_path, encoder_type, model.input_shape[1:3])
          
          # Prediction
          out_arr = model.predict(in_arr)

          out_dir = os.path.join(out_directory, img_dir)
          out_img = np.squeeze(out_arr,axis=0)
          Path(os.path.split(out_dir)[0]).mkdir(parents=True, exist_ok=True)
          out_img = restore_original_image_from_array(out_img, encoder_type)
          save_img(out_dir, out_img) 
      else:
          print("File %s doesn't exist." % file_path)


if __name__ == '__main__':
  args = argparser()
  main(args)
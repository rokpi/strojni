import os
import argparse
from my_utils.utils_new import getAngle, create_directory, get_all_filepaths
import pandas as pd
from datetime import datetime
from save_swinface import process_and_save_embeddings

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
        light = filename[15:16]
        #person_new = filename[0:3]  
        person_new = filename.split('_')[0]
        if person_new.isdigit():
            if check_img_requirements(filename):
                if extra:
                    filename = os.path.join(extra, filename)
                if person_new not in seen_persons:
                    seen_persons.add(person_new)
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img), 'light': light})#
                    num_of_img = num_of_img + 1
                elif person_new in seen_persons:
                    filtered_img_list.append({'person': person_new, 'img_dir': filename, 'angle': getAngle(img), 'light': light})#
                    num_of_img = num_of_img + 1
                        
        else:
            raise ValueError("Unexpected person classification (first three letters of basename should be an integer)")  
    print(f"Not as many as expected. Num of imgs: {num_of_img}.")
    print(f"Num of different people: {len(seen_persons)}")
    return filtered_img_list

def no_models():
    in_directory = '/home/rokp/test/images/images_mtcnn'
    out_dir = '/home/rokp/test/dataset'
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S") 
    filename = f"{current_date}"
    out_directory = create_directory(os.path.join(out_dir, filename))
    out_directory = os.path.join(out_directory, 'swinface')
    
    if not os.path.isfile(out_directory):
        print('Load model '+'swinface')
        img_list = get_all_filepaths(in_directory,'.jpg')

        filtered_img_list = make_directories(img_list, extra = in_directory)
        df = pd.DataFrame(filtered_img_list)
        output_dir = os.path.join('/home/rokp/test/test', current_date)
        process_and_save_embeddings(df, out_directory)
        print(f"File saved in {out_directory}")                            
    
if __name__ == '__main__':
  no_models()


    


import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'strojni')))
from tqdm import tqdm
from my_utils.utils_new import create_directory, load_data_df, cosine_similarity,ang_to_str
from datetime import datetime
from base.apply_vector import init_decoder
from my_utils.utils_test import find_matrix, tranform_to_img, read_txt
from my_utils.utils_pca import get_pca_vectors, define_true_vector

def main():
    vec_directory = '/home/rokp/test/centroid/20240911_121715'
    out_dir = '/home/rokp/test/test/test'
    in_directory = '/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnetconv4_3_3x3_imgs_images_mtcnn_data.npz'
    txt_dir = '/home/rokp/test/launch_test_arcface.txt'
    txt_dir_train = '/home/rokp/test/launch_train_arcface.txt'
    model_path = '/home/rokp/test/models/mtcnn/arcface-resnet/arcface-resnet.conv4_3_3x3.20231029-120710.hdf5'
    start = -30
    end = 30
    alpha = 5
    divide = 40
    indexes = [0, 9, 19, 29, 39]
    save = True

    all_angles = [-30,-15,0,15,30]
    angles = [ang for ang in all_angles if ang >= start and ang <= end]
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")    
    dirname = f"{current_date}_{ang_to_str(start)}_to_{ang_to_str(end)}"
    out = os.path.join(out_dir, dirname)
    create_directory(out)

    people_train = read_txt(txt_dir_train)

    df = load_data_df(in_directory)
    decoder_model, encoder_type = init_decoder(model_path)

    eigenvectors, eigenvalues = get_pca_vectors(df, people_train)
    print(f"Eigenvalues: {eigenvalues}")

    people = ['133']#read_txt(txt_dir)
    cos_sims = []

    df= df[df['angle'] == start]

    for person in people:
        out_directory = create_directory(os.path.join(out, person))
        df_person = df[df['person'] == person]
        num = 0
        #for pca_vector_idx in tqdm(range(len(eigenvectors)), total = len(eigenvectors)):#eig_vectors:+
        #    df_person = df[df['person'] == person]
            
            #pca_out_directory = create_directory(os.path.join(out_directory, str(pca_vector_idx)))
        #    vector = eigenvectors[pca_vector_idx]
        '''for i in range(len(angles)-1):
            matrix_dir  = find_matrix(angles[i], angles[i+1], vec_directory)
            vector = np.load(matrix_dir)'''
        

        for vector_idx in indexes:

            #eig_vectors = [eigenvectors[i] for i in indexes]
            #vector = define_true_vector(vector, eigenvectors[pca_vector_idx])
            
            #save = False
            num = 0
            out_dir = create_directory(os.path.join(out_directory, f"idx_{vector_idx}"))
            df_person_new = df_person
            multiply = alpha*10**(-2)

            for j in range(1, divide+1):
                    #final_dir = create_directory(os.path.join(out_directory, f"{j}_{divide}_{ang_to_str(angles[0])}_to_{ang_to_str(angles[i+1])}"))
                num +=1
                '''if num % divide == 0:
                    save = True'''
                multiply *= 10**(2/divide)
                lenghth = np.linalg.norm(eigenvectors[vector_idx])
                df_person_new = tranform_to_img(df_person, eigenvectors[vector_idx], multiply, 'centroid', encoder_type, decoder_model, model_path, out_dir, num, save=save)
                #print(df_person_new['embedding'])
                #cos_sim = cosine_similarity(df_person['embedding'], df_person_new['embedding'])
                #df_person = df_person_new
                #cos_sims.append(cos_sim)

    #print(f"Files successfuly saved in {out}")
    '''print(f"Maximum value: {np.max(cos_sims)}")
    print(f"Minimum value: {np.min(cos_sims)}")
    print(f"Average value: {np.mean(cos_sims)}")'''


import os
import cv2
import glob

def convert_png_to_jpg(source_dir, target_dir):
    # Ustvari ciljno mapo, če ta še ne obstaja
    os.makedirs(target_dir, exist_ok=True)

    # Poišči vse .png datoteke v izvorni mapi
    png_files = glob.glob(os.path.join(source_dir, "*.png"))

    for png_file in tqdm(png_files, total = len(png_files)):
        # Naloži sliko
        img = cv2.imread(png_file)
        
        # Pridobi ime datoteke brez končnice
        filename = os.path.basename(png_file).replace(".png", ".jpg")

        # Določi pot za shranjevanje
        jpg_file = os.path.join(target_dir, filename)

        # Shrani sliko kot .jpg
        cv2.imwrite(jpg_file, img)

def delete_files_with_240(source_dir):
    # Poišči vse .jpg datoteke v izvorni mapi
    files = glob.glob(os.path.join(source_dir, "*.jpg"))

    # Preveri vsako datoteko, če ima '240' na enakem mestu
    for file_path in tqdm(files, total = len(files)):
        filename = os.path.basename(file_path)
        parts = filename.split('_')

        # Preveri, če je četrti element '240'
        if len(parts) > 3 and parts[3] == '041':
            os.remove(file_path)

def delete_files_in_txt(txt, source_dir):
    files_delete = read_txt(txt)
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    for file in tqdm(files, total = len(files)):
        if os.path.basename(file) in files_delete:
            os.remove(file)

import os
import shutil

def consolidate_files(main_directory):
    # Pridobivanje vseh map v glavni mapi
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(main_directory, subdirectory)
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]

        # Premik datotek v glavno mapo
        for file in files:
            old_file_path = os.path.join(subdirectory_path, file)
            new_file_path = os.path.join(main_directory, file)
            
            # Če obstaja datoteka z enakim imenom v glavni mapi, preimenuj datoteko
            if os.path.exists(new_file_path):
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(new_file_path):
                    new_file_path = os.path.join(main_directory, f"{base}_{counter}{extension}")
                    counter += 1

            shutil.move(old_file_path, new_file_path)

        # Izbriši prazno mapo
        shutil.rmtree(subdirectory_path)

def rename_files(source_dir):
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    files.sort()
    dictionary = {}
    iden = 1
    num = 1
    for file_path in tqdm(files, total = len(files)):
        basename = os.path.basename(file_path)
        split = basename.split('_')
        if len(split) > 2:
            new_filename = split[0] + split[1]
        elif len(split) == 2:
            new_filename = split[0]
        else: 
            new_filename = basename.split('.')[0]

        if new_filename in dictionary:
            person = dictionary[new_filename][0]
            num = dictionary[new_filename][1] + 1
            dictionary[new_filename] = (person, num, file_path)
        else:
            person = iden
            num = 1
            dictionary[new_filename] = (person, num, file_path)
            iden += 1
            

        name = f"{person}_{num}.jpg"
        directory = os.path.dirname(file_path)
        os.rename(file_path, os.path.join(directory, name))

    for key, value in dictionary.items() :
        if value[1] == 1:
            file_path = os.path.dirname(value[2])
            filename = f"{value[0]}_1.jpg"
            file = os.path.join(file_path, filename)
            os.remove(file)
        
def remove_files_one(source_dir):
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    files.sort()
    dictionary = {}
    iden = 1
    num = 1
    for file_path in tqdm(files, total = len(files)):
        basename = os.path.basename(file_path)
        split = basename.split('_')
        if len(split) > 2:
            new_filename = split[0] + split[1]
        elif len(split) == 2:
            new_filename = split[0]
        else: 
            new_filename = basename.split('.')[0]

        if new_filename in dictionary:
            person = dictionary[new_filename][0]
            num = dictionary[new_filename][1] + 1
            dictionary[new_filename] = (person, num, file_path)
        else:
            person = iden
            num = 1
            dictionary[new_filename] = (person, num, file_path)
            iden += 1

    for key, value in dictionary.items() :
        print(value[1])
        if value[1] == 1:
            #file_path = os.path.dirname(value[2])
            filename = f"{value[0]}_1.jpg"
            print(filename)
            #file = os.path.join(file_path, filename)
            #os.remove(file)
  
if __name__ == '__main__':

    # Uporaba funkcije
    txt = "/home/rokp/test/pairs_CPLFW.txt"
    source_dir = "/home/rokp/test/images/images_mtcnn_cplfw"
    main_directory = "/home/rokp/test/images_lfw"  # Zamenjaj s potjo do svoje mape
    #delete_files_in_txt(txt, source_dir)
    remove_files_one(source_dir)

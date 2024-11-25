import os
import numpy as np
from tqdm import tqdm
from utils_new import load_data_df, get_all_angles
from datetime import datetime
from utils_test import find_matrix, read_txt
def main():
    vec_directory = '/home/rokp/test/bulk/20241022_145238'
    in_directory = '/home/rokp/test/test/dataset/mtcnn/arcface-resnet/20241022_144810_08_svet_vsi_koti/arcface-resnetconv4_3_3x3_images_mtcnn.npz'
    start = -90
    end = 90
    txt_dir = '/home/rokp/test/launch_train_arcface.txt'
    people = read_txt(txt_dir)#['001'], '028', '042', '049', '086', '133']#

    main_angle = 0

    all_angles = get_all_angles()
    all_vectors = []
    for i in range(len(all_angles)-1):
        matrix_dir = find_matrix(all_angles[i], all_angles[i+1], vec_directory)
        vector = np.load(matrix_dir)
        length = np.linalg.norm(vector)
        all_vectors.append(vector)

    angles = [ang for ang in all_angles if ang >= start and ang <= end and ang != main_angle] 

    right = 0
    wrong = 0
    empty = 0
    napaka = 0
    df = load_data_df(in_directory)
    for person in tqdm(people, total = len(people)):
        df_person = df[df['person'] == person].copy(deep=True)
        main_embedding = np.array(df_person[df_person['angle'] == main_angle]['embedding'].tolist())
        for i in range(len(angles)):
            embedding = np.array(df_person[df_person['angle'] == angles[i]]['embedding'].tolist())
            if np.all(embedding == 0):
                napaka += 1
                continue
            result, bool = recognise_angle_vector(main_embedding, embedding, angles, all_vectors, main_angle, answer=i)
            if bool:
                empty +=1
            else:
                test  = [angles[result[1]]]
                #if result[1] < len(angles)-1:
                #    test = test + [angles[result[1]+1]]
                #if result[1] > 0:
                #    test = test + [angles[result[1]-1]]

                if angles[i] in test: #,angles[result[1]-1], , angles[result[1]+1]
                    right +=1
                else:
                    wrong += 1
                    print(f"Person: {person}, Angle: {angles[i]} Guess: {angles[result[1]]}")
                #print(f"Result is: {result}")
                #print(f"For angle {angles[i]} assumption is {angles[result[1]]}")
    print(f"Right: {right} \nWrong: {wrong} \nEmpty: {empty} \n Napaka: {napaka} \nProcent: {right/(right+wrong)}")

def tranform_main_embedding(main_embedding, main_angle, angle_idx, all_vectors, angles):
    all_angles = angles + [main_angle]
    all_angles.sort()
    main_idx = all_angles.index(main_angle)
    to_idx = all_angles.index(angles[angle_idx])
    difference = to_idx - main_idx
    if difference > 0:
        for i in range(difference-1):
            vec_idx = i + main_idx
            main_embedding += all_vectors[vec_idx]
        vector = all_vectors[main_idx+difference-1]
    elif difference < 0:
        for i in range(abs(difference)-1):
            vec_idx = main_idx-(i+1)
            main_embedding -= all_vectors[vec_idx]
        vector = -all_vectors[main_idx+difference]
    return main_embedding, vector

def recognise_angle_vector(main_embedding, embedding, angles, all_vectors, main_angle, answer = None):
    result = []
    empty = False
    final = (0, 0)
    for i in range(len(angles)):
        main_embedding_copy = np.copy(main_embedding)
        main_embedding_copy, vector = tranform_main_embedding(main_embedding_copy, main_angle, i, all_vectors, angles)
        point_vector =  embedding-main_embedding_copy

        trans_length = np.dot(point_vector, vector)/np.dot(vector, vector)
        #if it transforms on to the vector
        if np.abs(trans_length) <= 1.6:
            #distance
            if trans_length > 0:
                projection = trans_length * vector
                dist = np.linalg.norm(point_vector-projection)
            elif trans_length < 0:
                dist = np.linalg.norm(point_vector)      
            result.append((dist, i))
    if len(result) > 0:
        final = min(result, key=lambda x: x[0])
    else:
        empty = True
    if answer:
        if final[1] != answer:
            v = 1
    return final, empty        


if __name__ == '__main__':
  main()
import os
import numpy as np
from tqdm import tqdm
from utils_new import load_data_df, get_all_angles, cosine_similarity, ang_to_str
from datetime import datetime
from utils_test import read_txt, find_centroid

    
def recognise_angle_space(embedding, global_mean, all_centroids):
    embedding = np.subtract(embedding,global_mean)
    result = []
    empty = False
    final = (0, 0)
    embedding = embedding.reshape([1, 512])
    vectors = embedding - all_centroids
    dist = np.linalg.norm(vectors, axis = 1)
    if len(dist) != 0:
        minimum = np.min(dist)
        index = int(np.argmin(dist))
        final = (minimum, index)
    else:
        empty = True
    return final, empty  
    '''for i in range(len(all_centroids)):
        vector = embedding-all_centroids[i]
        dist = np.linalg.norm(vector)  #cosine_similarity(embedding, all_centroids[i]) #
        result.append((dist, i))
    if len(result) > 0:
        final = min(result, key=lambda x: x[0])
    else:
        empty = True'''
def main():
    vec_directory = '/home/rokp/test/bulk/20241029_133446_cent_vsi_kot_svetlobe'
    out_dir = '/home/rokp/test/test/test'
    in_directory = '/home/rokp/test/test/dataset/mtcnn/arcface-resnet/20241029_084028_vse_svetloba_vsi_koti/arcface-resnetconv4_3_3x3_images_mtcnn.npz'
    start = -90
    end = 90
    alpha = 1
    txt_dir = '/home/rokp/test/launch_train_arcface.txt'
    people = read_txt(txt_dir)#['001'], '028', '042', '049', '086', '133']#

    all_angles = get_all_angles()
    all_centroids = []
    angles = [ang for ang in all_angles if ang >= start and ang <= end]
    for i in range(len(angles)):
        centroid_str =  f"Centroid_{ang_to_str(angles[i])}.npy"
        centroid_dir = find_centroid(centroid_str, vec_directory)
        vector = np.load(centroid_dir)
        vector /=np.linalg.norm(vector)
        all_centroids.append(vector)

    global_mean = np.load(find_centroid("global.npy", vec_directory)) 

    right = 0
    wrong = 0
    empty = 0
    napaka = 0
    df = load_data_df(in_directory)
    for person in tqdm(people, total = len(people)):
        df_person = df[df['person'] == person].copy(deep=True)
        for i in range(len(angles)): 
            embedding = np.array(df_person[df_person['angle'] == angles[i]]['embedding'].tolist())
            if np.all(embedding == 0):
                napaka += 1
                continue
            result, bool = recognise_angle_space(embedding, global_mean, all_centroids)
            if bool:
                empty +=1
            else:
                test  = [angles[result[1]]]
                '''if result[1] < len(angles)-1:
                    test = test + [angles[result[1]+1]]
                if result[1] > 0:
                    test = test + [angles[result[1]-1]]'''


                if angles[i] in test: #,angles[result[1]+1], 
                    right +=1
                else:
                    wrong += 1
                #print(f"Result is: {result}")
                #print(f"For angle {angles[i]} assumption is {angles[result[1]]}")
    print(f"Right: {right} \nWrong: {wrong} \nEmpty: {empty} \nNapaka: {napaka}\nProcent: {right/(right+wrong)}")

      


if __name__ == '__main__':
  main()
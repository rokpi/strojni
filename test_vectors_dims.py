import os
import numpy as np
from tqdm import tqdm
from my_utils.utils_new import create_directory, load_data_df, get_all_angles,get_all_lights, ang_to_str,find_centroid, cosine_similarity
from datetime import datetime
from apply_vector import init_decoder, restore
from my_utils.utils_test import tranform_to_img, read_txt
import math

def get_and_test_vector(df_person, angles):
    df_vec_neut = df_person[df_person['angle']==angles[0]].copy(deep=True)
    df_vec_pos = df_person[df_person['angle']==angles[1]].copy(deep=True)

    embeddings_neut = df_vec_neut['embedding'].apply(lambda x: x.copy()).tolist()
    embeddings_pos = df_vec_pos['embedding'].apply(lambda x: x.copy()).tolist()
    '''floki = df_person_angle['embedding'].apply(lambda x: x.copy()).tolist()
    floki1 = floki[0]
    if not np.array_equal(floki1, neut_embedd):
        raise ValueError("Nista enaka")            
    print(f"Embeddings0: {embeddings_neut[0][:10]}\n")
    print(f"Embeddings1: {embeddings_pos[0][:10]}\n")
    print(f"Floki {floki[0][:10]}\n")'''
    pos_embedd = embeddings_pos[0]
    neut_embedd = embeddings_neut[0]            

    vector_new = np.subtract(pos_embedd ,neut_embedd)
    test = neut_embedd+vector_new

    if not np.allclose(test, pos_embedd, rtol = 1e-6):
        mask = ~np.isclose(test, pos_embedd, rtol = 1e-8)
        for index in np.where(mask)[0]:
            print(f"Razlika pri indeksu {index}: a = {test[index]}, b = {pos_embedd[index]}")
        raise ValueError("napaka")
    vector_new /= np.linalg.norm(vector_new)
    return vector_new

def get_P_np(eigenvectors, num_vectors):
    #get number of eigenvectors equal to num_vectors
    all_vectors = eigenvectors[:num_vectors].astype(np.float32)
    shape = all_vectors.shape[1]
    #test_vector =all_vectors[1].reshape(1, all_vectors[1].shape[0])
    #test = test_vector * test_vector.T

    P_mat_multiply = np.stack([
        #np.outer(all_vectors[i], all_vectors[i])
        all_vectors[i].reshape(1, shape)*all_vectors[i].reshape(1, shape).T
        for i in range(num_vectors)
    ], axis=0)

    # 3) Vsoto teh matrik seštej v eno samo matriko
    P_sum = P_mat_multiply.sum(axis=0)

    # 4) P = I - vsota rank-1 projekcij
    P = np.eye(shape, dtype=np.float32) - P_sum

    return P

def get_cos_sim(sim_array):
    similarities = []
    for i in range(len(sim_array)):
        for j in sim_array[i+1:]:
            similarities.append(cosine_similarity(sim_array[i], j))
    similarities = np.vstack(similarities)
    sim = np.mean(similarities)
    print(f"    Podobnost: {sim:.5f}")
    return sim

def main():
    centroid_directory = '/home/rokp/test/bulk/20250207_122859_light'
    out_dir = '/home/rokp/test/test'
    in_directory = '/home/rokp/test/dataset/mtcnn/vgg-vgg/multipie_vse/vgg-vggmtcnn_images_mtcnn.npz'
    txt_dir = '/home/rokp/test/launch_test_arcface.txt'
    txt_dir_train = '/home/rokp/test/strojni/launch/launch_train_arcface.txt'
    model_path = '/home/rokp/test/models/mtcnn/vgg-vgg/vgg-vgg.mtcnn.conv4_3.20230124-204737.hdf5'
    start_idx = 0#-90
    end_idx = 10#90
    alpha = 1
    lim__maxmul = 100
    divide = 1

    save = True
    limit_space = True
    segment = True
    people = ['001']#, '028', '042', '049', '086', '133']#read_txt(txt_dir)
    cent_basename = os.path.basename(centroid_directory)
    cent_type = cent_basename[16:21]
    if cent_type == 'angle':
        cent_all = get_all_angles()
        exclude = 'light'
        exclude_focus = 8
    elif cent_type == 'light':
        cent_all = get_all_lights()
        exclude = 'angle'
        exclude_focus = 0


    '''if end_idx + 1 > len(cent_all):
        raise ValueError(f"Too many. Demanded end idx: {end_idx}\n Length of cent_all: {len(cent_all)}")
    elif end_idx+1 == len(cent_all):
        print('Vse...')
    else:
        cent_all = cent_all[start_idx:end_idx+1]'''
    
    cent_all = [i for i in range(1, 14)]
    all_centroids = []
    for i in range(len(cent_all)):
        #if all_angles[i] in selection:
            centroid_str =  f"Centroid_{ang_to_str(cent_all[i])}.npy"
            centroid_dir = find_centroid(centroid_str, centroid_directory)
            vector = np.load(centroid_dir)
            #vector = apply_base(vector, base_vectors, VtV_inv)
            #vector /=np.linalg.norm(vector)
            all_centroids.append(vector)

    df = load_data_df(in_directory)
    unique_values = df['light'].unique()

    eigenvectors =np.load(find_centroid("eigenvectors.npy", centroid_directory))
    global_mean = np.mean(np.vstack(df['embedding']), axis=0).reshape(1, -1)
    df['embedding'] = df['embedding'].apply(lambda x: x - global_mean)

    #df = df[df[exclude] == exclude_focus]
    test_embeddings = []

    for j in range(len(cent_all)):
        tmp_embed = all_centroids[7]-all_centroids[j]
        test_embeddings.append(tmp_embed)

    test_embeddings = np.vstack(test_embeddings)
    norms = np.linalg.norm(test_embeddings, axis = 1)
    test_embeddings = test_embeddings / norms[:, np.newaxis]
    test_embedds_1 = test_embeddings[8:][::-1]#od 9 naprej
    test_embedds_2 = test_embeddings[:5]
    P21 = get_P_np(test_embedds_1, 5)
    P22 = get_P_np(test_embedds_1, 2)
    P23 = get_P_np(test_embedds_1, 1)

    P11 = get_P_np(test_embedds_2, 5)
    P12 = get_P_np(test_embedds_2, 2)
    P13 = get_P_np(test_embedds_2, 1)
    #P20 = get_P_np(test_embeddings[:6], 5)
    #P21 = get_P_np(test_embeddings[4:6], 1) 
    #P11 = get_P_np(test_embeddings[9:11], 1)
    #P10 = get_P_np(test_embeddings[9:], 5)



    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")    
    #dirname = f"{current_date}_{cent_type}_{ang_to_str(cent_all[start_idx])}_to_{ang_to_str(cent_all[end_idx])}"
    dirname = f"{current_date}_test"
    out = os.path.join(out_dir, dirname)
    create_directory(out)

    people_train = read_txt(txt_dir_train)

    decoder_model, encoder_type = init_decoder(model_path)

    if not segment: 
        df= df[df[cent_type] ==cent_all[start_idx]]
    
    #maxmul = eigenvalues*lim__maxmul#(eigenvalues/eigenvalues[0])*lim__maxmul
    multiply = alpha/divide
    #multiply = 10**(-2)
    df = df[df['light'] == 1]#.isin([i for i in range(1, 14)])]
    df = df[df['angle'] == 0]

    cosine_sim_1 = []
    cosine_sim_2 = []
    cosine_sim_3 = []

    for person in people:
        out_directory = create_directory(os.path.join(out, person))
        df_person = df[df['person'] == person]
        embeddings = np.vstack(df_person['embedding'])
        #vector = np.zeros_like(eigenvectors[0]) 
        for i in tqdm(range(1, len(cent_all)), total = len(cent_all) - 1):
            #if segment:
            #    df_person_focus = df_person[df_person[cent_type] == cent_all[i]].copy(deep = True)            

            #if len(df_person_focus) == 0:
            #    continue
            vector_new = all_centroids[i-1]-all_centroids[i]
            #vector_new=0
            embeddings += vector_new

            #vector_new = vector/np.linalg.norm(vector)

            '''coeficients = np.dot(eigenvectors.T, vector)
            
            #realizira gede na omejeno območje
            maska = np.abs(maxmul) < np.abs(coeficients)
            
            if limit_space:
                coeficients[maska] = np.sign(coeficients[maska]) * np.abs(maxmul[maska])
            
            vector_new = np.dot(eigenvectors, coeficients)'''

                #embeddings = np.vstack(df_person_focus['embedding'])
                #P_embeddings = np.dot(P, embeddings.T).T
            if i not in [5, 6, 7, 8, 9]:
                if i in [1, 2]:
                    P = P11
                elif i in [3]:
                    P = P12                    
                elif i in [4]:
                    P = P13
                elif i in [10]:
                    P = P23
                elif i in [11]:
                    P = P22
                elif i in [12, 13]:
                    P = P21
                P_embeddings = []
                for k in range(len(embeddings)):
                    P_embedding = np.dot(P, embeddings[k].reshape(embeddings.shape[1], 1)).T
                    P_embeddings.append(P_embedding)

                P_embeddings = np.vstack(P_embeddings)
                if i < 7:
                    cosine_sim_1.append(P_embeddings.copy())
                else: 
                    cosine_sim_3.append(P_embeddings.copy())
                P_embeddings = P_embeddings + global_mean                  
                #df_person['embedding'] = list(P_embeddings)
            else:
                cosine_sim_2.append(embeddings.copy())
                #df_person['embedding'] = list(embeddings+global_mean)
            #out_dir = create_directory(os.path.join(out_directory, str(cent_all[i])))
            #restore(df_person, decoder_model, encoder_type, model_path, out_directory,'lala')

            '''for j in range(1, divide+1):
                #multiply *= 10**(2/divide)
                num+=1
                if segment:
                    df_person_focus = tranform_to_img(df_person_focus, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)
                else:
                    df_person = tranform_to_img(df_person, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)'''
    
    print("podobnost...")
    all_sim = []
    all_sim.append(get_cos_sim(cosine_sim_1))
    all_sim.append(get_cos_sim(cosine_sim_2))
    all_sim.append(get_cos_sim(cosine_sim_3))
    all_sim = np.vstack(all_sim)
    print(f"Skupaj: {np.mean(all_sim):.5f}")

if __name__ == '__main__':
  main()


'''person1 = ['001']
person2 = ['002']
eigenvectors1, eigenvalues1, grouped1 = get_pca_vectors(df_old, person1)
eigenvectors2, eigenvalues2, grouped2 = get_pca_vectors(df_old, person2)

angles_w = [angles[i], angles[i+1]]
df_person1 = df_old[df_old['person'].isin(person1)].copy(deep = True)
df_person2 = df_old[df_old['person'].isin(person2)].copy(deep = True)
vector1 = get_and_test_vector(df_person=df_person1, df_person_angle=df_person_angle, angles=angles_w)
vector2 = get_and_test_vector(df_person=df_person2, df_person_angle=df_person_angle, angles=angles_w)


R = kabsch(eigenvectors2, eigenvectors1)'''
'''cos_sim = []
eigenvectors2_test = np.dot(eigenvectors2, R.T)
for j in range(len(eigenvectors1)):
    cos_sim.append(cosine_similarity(eigenvectors2_test[j], eigenvectors1[j]))
print(f"Cos sim avg: {np.mean(cos_sim)}")'''

'''vector1 = np.squeeze(vector1)
vector2 = np.squeeze(vector2)
vector_new = np.dot(R, vector2)
vector_new = np.squeeze(vector_new)

for i in range(5):
    print(f"Cos idx[{i}]: {cosine_similarity(eigenvectors1[i], vector1)}")
    print(f"Cos2 idx[{i}]: {cosine_similarity(eigenvectors2[i], vector2)}")'''         

#print(f"Cosine similarity between vector2/vectornew: {cosine_similarity(vector1, vector_new)}")

'''diff = []
big = 0
same = 0
for j in range(len(vector)):
    if round(vector[j], 4) != round(vector_new[j], 4):
        calc = np.abs(vector[j]-vector_new[j])
        if calc > 0.00001:
            big+=1
        diff.append(calc)
    else:
        same += 1
print(f"Število zamenjanih: {np.sum(maska)}")
print(f"Število enakih: {same} \nŠtevilo različnih: {len(diff)} \nMaksimalna razlika: {np.max(diff)} \nBig:{big}")
print(f"Max vector {np.max(vector)} \nMax new vectro: {np.max(vector_new)}")
print(f"Min vector {np.min(vector)} \nMin new vectro: {np.min(vector_new)}")
print(f"AbsMin vector {np.min(np.abs(vector))} \nAbsMin new vectro: {np.min(np.abs(vector_new))}")
sub = np.abs(coeficients)/np.abs(maxmul)
bignum = [i for i in sub if i > 1]
print(f"bignum: {len(bignum)}")'''
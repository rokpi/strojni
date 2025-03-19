import os
import numpy as np
from tqdm import tqdm
from my_utils.utils_new import create_directory, load_data_df, get_all_angles,get_all_lights, ang_to_str,find_centroid, cosine_similarity
from datetime import datetime
from apply_vector import init_decoder, restore
from my_utils.utils_test import tranform_to_img, read_txt
from sklearn.decomposition import PCA
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
    centroid_directory = '/home/rokp/test/dataset/mtcnn/resnet-vgg/20250503_multipie_vse/20250305_104313_light'
    in_directory = '/home/rokp/test/dataset/mtcnn/resnet-vgg/20250503_multipie_vse/resnet-vggmtcnn_images_mtcnn.npz'
    model_path = '/home/rokp/test/models/mtcnn/resnet-vgg/resnet-vgg.mtcnn.conv4_3.20230124-202043.hdf5'

    out_dir = '/home/rokp/test/test'
    txt_dir_train = '/home/rokp/test/strojni/launch/launch_train_arcface.txt'
    lim__maxmul = 100
    divide = 1

    original_idx = 7
    save = True
    limit_space = True
    segment = True
    people = ['001', '028', '042', '031', '133']#read_txt(txt_dir)
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
    
    #Scent_all = [i for i in range(1, 14)]
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
    #eigenvectors =np.load(find_centroid("eigenvectors.npy", centroid_directory))
    global_mean = np.mean(np.vstack(df['embedding']), axis=0).reshape(1, -1)
    #df['embedding'] = df['embedding'].apply(lambda x: x - global_mean)

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")    
    #dirname = f"{current_date}_{cent_type}_{ang_to_str(cent_all[start_idx])}_to_{ang_to_str(cent_all[end_idx])}"
    dirname = f"{current_date}_test"
    out = os.path.join(out_dir, dirname)
    create_directory(out)

    people_train = read_txt(txt_dir_train)

    decoder_model, encoder_type = init_decoder(model_path)

    if not segment: 
        df= df[df[cent_type] == cent_all[original_idx]]
        all_vectors = []
        for i in range(len(cent_all)):
            if i == original_idx:
                all_vectors.append(np.zeros(all_centroids[0].shape))
            else:
                all_vectors.append(all_centroids[i]-all_centroids[original_idx])

    
    #maxmul = eigenvalues*lim__maxmul#(eigenvalues/eigenvalues[0])*lim__maxmul
    #df = df[df['light'] == 7]
    df = df[df['angle'] == 0]

    for person in people:
        out_directory = create_directory(os.path.join(out, person))
        df_person = df[df['person'] == person]
        if not segment:
            embeddings = np.vstack(df_person['embedding']) - global_mean
            img_dirs = df_person['img_dir'].to_list()

        for i in tqdm(range(0, len(cent_all)), total = len(cent_all)):
            if segment:
                df_person_focus = df_person[df_person[cent_type] == cent_all[i]].copy(deep = True)  
                if len(df_person_focus) == 0:
                    continue
          
                embeddings = np.vstack(df_person_focus['embedding']) - global_mean
                img_dirs = df_person_focus['img_dir'].to_list()

                vector_new = all_centroids[i]-all_centroids[i-1]
                vector_new=0
            else:
                vector_new = all_vectors[i]

            vector_new /= divide
            
            
            #out_dir = create_directory(os.path.join(out_directory, str(cent_all[i])))
            for j in range(1, divide+1):
                embeddings_vec = embeddings + vector_new
                img_dirs_vec = [img_dir[:-25] + str(cent_all[i]) + ".jpg" for img_dir in img_dirs]

                #vector_new = vector/np.linalg.norm(vector)

                '''coeficients = np.dot(eigenvectors.T, vector)
                
                #realizira gede na omejeno območje
                maska = np.abs(maxmul) < np.abs(coeficients)
                
                if limit_space:
                    coeficients[maska] = np.sign(coeficients[maska]) * np.abs(maxmul[maska])
                
                vector_new = np.dot(eigenvectors, coeficients)'''
                if segment:
                    if save:
                        restore(embeddings_vec + global_mean, img_dirs_vec, decoder_model, encoder_type, model_path, out_directory, num = j)
                else:
                    if save:
                        restore(embeddings_vec + global_mean, img_dirs_vec, decoder_model, encoder_type, model_path, out_directory, num = j)

            '''for j in range(1, divide+1):
                #multiply *= 10**(2/divide)
                num+=1
                if segment:
                    df_person_focus = tranform_to_img(df_person_focus, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)
                else:
                    df_person = tranform_to_img(df_person, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)'''
    
        print(f"Files successfuly saved in {out_directory}")

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
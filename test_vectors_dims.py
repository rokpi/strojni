import os
import numpy as np
from tqdm import tqdm
from my_utils.utils_new import create_directory, load_data_df, cosine_similarity, get_all_angles,ang_to_str
from datetime import datetime
from base.apply_vector import init_decoder
from my_utils.utils_test import find_matrix, tranform_to_img, read_txt
from my_utils.utils_pca import get_pca_vectors, define_true_vector, kabsch

def get_and_test_vector(df_person, df_person_angle, angles):
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

def main():
    vec_directory = '/home/rokp/test/bulk/20241118_083956_vec_vgg_vse'
    out_dir = '/home/rokp/test/test'
    in_directory = '/home/rokp/test/dataset/mtcnn/vgg-vgg/20241111_124312_vgg/vgg-vggmtcnn_images_mtcnn.npz'
    txt_dir = '/home/rokp/test/launch_test_arcface.txt'
    txt_dir_train = '/home/rokp/test/strojni/launch/launch_train_arcface.txt'
    model_path = '/home/rokp/test/models/mtcnn/vgg-vgg/vgg-vgg.mtcnn.conv4_3.20230124-204737.hdf5'
    start = -90
    end = 90
    alpha = 1
    lim__maxmul = 100
    divide = 4

    save = True
    limit_space = True
    segment = False
    indexes = [i for i in range(512)]
    people = ['001']#, '028', '042', '049', '086', '133']#read_txt(txt_dir)

    all_angles = get_all_angles()
    angles = [ang for ang in all_angles if ang >= start and ang <= end]
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")    
    dirname = f"{current_date}_{ang_to_str(start)}_to_{ang_to_str(end)}"
    out = os.path.join(out_dir, dirname)
    create_directory(out)

    people_train = read_txt(txt_dir_train)

    df = load_data_df(in_directory)

    decoder_model, encoder_type = init_decoder(model_path)

    '''df = df[df['angle']==15]
    tranform_to_img(df, np.zeros([1,4096]),0, 'centroid', encoder_type, decoder_model, model_path, out, 0, save=save)'''

    #eigenvectors, eigenvalues, grouped = get_pca_vectors(df, people_train)
    #np.save(os.path.join(out_dir, 'eigenvectors.npy'), eigenvectors)
    #np.save(os.path.join(out_dir, 'eigenvalues.npy'), eigenvalues)

    eigenvectors = np.load('/home/rokp/test/test/values/eigenvectors.npy')
    eigenvalues = np.load('/home/rokp/test/test/values/eigenvalues.npy')
    num_vectors = 100
    base_vectors = eigenvectors[:,:num_vectors]

    if not segment: 
        df= df[df['angle'] == start]

    maxmul = eigenvalues*lim__maxmul#(eigenvalues/eigenvalues[0])*lim__maxmul
    multiply = alpha/divide
    #multiply = 10**(-2)

    for person in people:
        out_directory = create_directory(os.path.join(out, person))
        df_person = df[df['person'] == person]
        vector = np.zeros_like(eigenvectors[0]) 
        num = 0
        for i in tqdm(range(len(angles)-1), total = len(angles)-1):
            if segment:
                df_person_angle = df_person[df_person['angle'] == angles[i]].copy(deep = True)            

            matrix_dir  = find_matrix(angles[i], angles[i+1], vec_directory)
            vector = np.load(matrix_dir)

            VtV_inv = np.linalg.inv(base_vectors.T @ base_vectors)  # Inverz matrike
            p_proj = base_vectors @ (VtV_inv @ (base_vectors.T @ vector))
            vector_new = p_proj
            #base = eigenvectors[]

            #vector_new = vector/np.linalg.norm(vector)

            '''coeficients = np.dot(eigenvectors.T, vector)
            
            #realizira gede na omejeno območje
            maska = np.abs(maxmul) < np.abs(coeficients)
            
            if limit_space:
                coeficients[maska] = np.sign(coeficients[maska]) * np.abs(maxmul[maska])
            
            vector_new = np.dot(eigenvectors, coeficients)'''

            for j in range(1, divide+1):
                #multiply *= 10**(2/divide)
                num+=1
                if segment:
                    df_person_angle = tranform_to_img(df_person_angle, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)
                else:
                    df_person = tranform_to_img(df_person, vector_new, multiply, 'centroid', encoder_type, decoder_model, model_path, out_directory, num, save=save)
                    


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
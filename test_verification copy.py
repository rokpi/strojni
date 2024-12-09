import numpy as np
from my_utils.utils_new import find_centroid, load_data_df, cosine_similarity, create_directory, get_all_angles, ang_to_str
from tqdm import tqdm
import pandas as pd
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import os


def rotate_embedding_both(embedding,embedding2, target_idx, result1, result2, all_centroids, embedding_cent):
  emb_final = embedding.copy()
  if result1 != target_idx:
      vector = all_centroids[target_idx]-embedding_cent
      emb_final += vector
  if result2 != target_idx:
      embedding2_cent = all_centroids[result2]
      vector2 = all_centroids[target_idx]-embedding2_cent
      embedding2 += vector2
  return emb_final, embedding2

def heavy_embedding(embeddings, embedding, number):
    return embeddings + [embedding] * number

def calculate_average_weight(embedding,embedding2,result1, result2, type, weights, all_centroids):

  embeddings1 = []
  embeddings2 = []
  both = type == 'both'
  embedding = embedding.ravel()
  embedding2 = embedding2.ravel()

  for i in range(len(all_centroids)):
    if i != result1:
      vector1 = all_centroids[i]-all_centroids[result1]
      emb_final1 = embedding + vector1
      embeddings1 = heavy_embedding(embeddings1,emb_final1, weights[i])

    if both:
      if i != result2:
        vector2 = all_centroids[i]-all_centroids[result2]
        emb_final2 = embedding2 + vector2
        embeddings2 = heavy_embedding(embeddings2, emb_final2, weights[i])

  embeddings1 = heavy_embedding(embeddings1, embedding, weights[result1])
  embeddings2 = heavy_embedding(embeddings2, embedding2, weights[result2])

  embeddings1 = np.array(embeddings1)
  embeddings2 = np.array(embeddings2)

  embedding2 = np.mean(embeddings2, axis = 0)
  embedding = np.mean(embeddings1, axis = 0)
  return embedding, embedding2

def clean_df(df, array, angles = None):
  img_dirs = array[:, 1]
  if angles:
    df.columns = ['person', 'img_dir','angle', 'embedding'] 
  else:
    df.columns = ['person', 'img_dir', 'embedding']
  mask = ~df.set_index(['img_dir']).index.isin(
    img_dirs
  )
  cleaned_arr = np.array(df[mask].reset_index(drop=True).values)
  
  return cleaned_arr  

def calculate_needed(fpr, tpr):
  mask_eer = fpr + tpr -1
  eer_idx = int(np.argmin(np.abs(mask_eer)))

  fpr_001_mask = fpr-0.0001
  fpr_001_idx = int(np.argmin(np.abs(fpr_001_mask)))  
  fpr_01_mask = fpr-0.001
  fpr_01_idx = int(np.argmin(np.abs(fpr_01_mask)))
  fpr_1_mask = fpr-0.01
  fpr_1_idx = int(np.argmin(np.abs(fpr_1_mask)))

  '''print(f"eer:({fpr[eer_idx]:.5f},{tpr[eer_idx]:.5f})")
  print(f"fpr01:({fpr[fpr_01_idx]:.5f},{tpr[fpr_01_idx]:.5f})")
  print(f"fpr1:({fpr[fpr_1_idx]:.5f},{tpr[fpr_1_idx]:.5f})")'''
  return fpr[eer_idx], tpr[fpr_001_idx], tpr[fpr_01_idx], tpr[fpr_1_idx]

def save_graphs(similarity1, difference1, out_directory):
  similarity_scores = np.concatenate((similarity1,difference1))
  true_labels =np.concatenate((np.ones([1, len(similarity1)]), np.zeros([1, len(difference1)])), axis = 1)
  true_labels = true_labels.ravel()
  similarity_scores = similarity_scores.ravel()
  fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores) 
  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
  plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')

  # Nastavimo naslov, oznake osi in legendo
  plt.title("ROC Curve for Face Recognition Model")
  plt.xlabel("False Positive Rate (FPR)")
  plt.ylabel("True Positive Rate (TPR)")
  plt.xscale('log')
  plt.yscale('log')
  plt.xlim([1e-3, 1.2])
  plt.ylim([1e-1, 1.2])
  plt.legend(loc="lower right")

  # Shranimo graf kot .jpg datoteko
  plt.savefig(f"{out_directory}/ROC_skicit.jpg", format="jpg")

  bins = np.linspace(-0.3, 1, 10000)

  hist1, _ = np.histogram(similarity1, bins=bins, density=True)
  hist2, _ = np.histogram(difference1, bins=bins, density=True)
  overlap_area = np.sum(np.minimum(hist1, hist2) * np.diff(bins))

  plt.figure(figsize=(10, 6))
  plt.hist(similarity1, bins=300, color='blue', alpha = 0.5, label = "Similarity")
  plt.hist(difference1, bins=300, color='purple', alpha = 0.5, label = "Difference")  
  plt.title(f"Distribution of Similarity Scores \nOverlap Area: {overlap_area:.5f}")
  plt.xlabel("Similarity Score")
  plt.ylabel("Frequency")
  plt.xlim([-0.3, 1])
  #plt.ylim([0.0, 1000.0])
  plt.savefig(f"{out_directory}/dist_all.jpg", format="jpg")

  return overlap_area

def check(array_cleaned, array_test, what, all_centroids, angles = None, all = True):
    all_count = len(array_cleaned)
    #weights = [10,8,6,4,0,2,0,4,6,8,10]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    array_copy = array_cleaned.copy()
    difference = []
    wrong = 0
    right = 0
    num_random = math.ceil(len(array_copy) / len(array_test))

    for i in tqdm(range(len(array_test)), total=len(array_test)): #range(len(array_test)):#tqdm(range(len(array_test)), total=len(array_test)): #
        if angles:
          person_num = 0
          embedding_num = 3
          angle_num = 2
        else:
           person_num = 0
           embedding_num = 2
        person = array_test[i][person_num]
        embedding = array_test[i][embedding_num]

        if what == 'dif':
            # random_four
            if num_random > all_count:
               num_random = all_count
            test_group = array_copy[array_copy[:, person_num] != person]
            length = len(test_group)
            #indices = np.random.choice(array_copy.shape[0], num_random, replace=False)
            #test_group = array_copy[indices]
            #array_copy = np.delete(array_copy, indices, axis=0)
        elif what == 'sim':
            # same person
            test_group = array_copy[array_copy[:, person_num] == person]
            length = len(test_group)
            all_count -= length
        else:
            raise ValueError('Wrong input of what to check.')

        embeddings2 = np.vstack(test_group[:, embedding_num])
        
        if angles:
          angle = array_test[i][angle_num]
          angles2 = test_group[:, angle_num]


        embedding_test = embedding.reshape(1, -1)
        vectors1 = embedding_test - all_centroids
        result1 = int(np.argmin(np.linalg.norm(vectors1, axis=1)))

        vectors2 = embeddings2[:, np.newaxis, :] - all_centroids
        distances2 = np.linalg.norm(vectors2, axis = 2)
        results2 = np.argmin(distances2, axis = 1)
        embedding_cent = all_centroids[result1]
        results = result1 - results2

        print(results2.tolist())
        all_centroids = np.array(all_centroids)
        #rotate embedding one
        mask_no_rot = results != 0
        mask_no_rot = mask_no_rot.reshape(1, len(mask_no_rot))
        embedding2_cents = all_centroids[results2]
        vectors_rot_one = mask_no_rot*(embedding_cent-embedding2_cents)
        emb2_rot_one = embeddings2 + vectors_rot_one

        #rotate embedding both
        target_idx = 5
        mask_target_rot1 = result1 != target_idx
        mask_target_rot2 = results2 != target_idx
        #maska 1 ali 0
        emb1_rot_both = embedding+mask_target_rot1*all_centroids[target_idx]
        vectors_rot_both = mask_target_rot2*all_centroids[target_idx]
        emb2_rot_both = embeddings2 + vectors_rot_both

        #calculate average weight
        vectors1_avg = np.delete(all_centroids, result1)-all_centroids[result1]
        embs1_avg = np.concatenate((embedding, embedding + vectors1_avg), axis = 0)
        emb1_avg = (weights*embs1_avg)/np.sum(weights)

        vectors2_avg = np.delete(all_centroids, results2)-all_centroids[results2]
        embs2_avg = np.concatenate((embedding, embedding + vectors2_avg), axis = 0)
        emb2_avg = (weights*embs2_avg)/np.sum(weights)        





        for j in range(length):
            embedding2 = embeddings2[j]
            num = 0
            if angles:
              angle2 = angles2[j]
              if angles[result1] != angle:
                  num += 1
              if angles[results2[j]] != angle2:
                  num += 1

            #if it is not the same angle
            if all:
              dif_sim = check_all(embedding, embedding2, all_centroids, embedding_cent, result1, results2[j], results[j], weights)
            else:
              embedding_new = embedding
              '''if results[j] != 0:
                  embedding2_cent = all_centroids[results2[j]]
                  vector = embedding_cent-embedding2_cent
                  embedding2 += vector'''
              #embedding_new, embedding2 = rotate_embedding_both(embedding,embedding2, 5, result1, results2[j], all_centroids, embedding_cent)
              #embedding_new, embedding2 = calculate_average_weight(embedding,embedding2,result1, results2[j],'both', weights, all_centroids)
              dif_sim = cosine_similarity(embedding_new, embedding2)
            if num > 2:
                raise ValueError("Preveč napak")

            right += 2 - num
            wrong += num

            difference.append(dif_sim)

        if all_count <= 1:
            break

    return difference, wrong, right

def compute_cosine_similarities(embedding_pairs, epsilon=1e-8):
    # Dobimo embedding1 in embedding2 za vse pare
    embedding1 = embedding_pairs[:, 0, :]
    embedding2 = embedding_pairs[:, 1, :]
    
    # Norme embeddingov
    norm1 = np.linalg.norm(embedding1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embedding2, axis=1, keepdims=True)
    
    # Kosinusna podobnost
    dot_products = np.sum(embedding1 * embedding2, axis=1, keepdims=True)
    cosine_similarities = dot_products / (np.maximum(norm1 * norm2, epsilon))
    
    return cosine_similarities.squeeze()

def apply_base(vector, base_vectors, V_inv):
  return base_vectors @ (V_inv @ (base_vectors.T @ vector))

def check_all(embedding, embedding2, all_centroids, embedding_cent, result1, result2, difference, weights):
  final_embeddings = [embedding for _ in range(10)]
  final_embeddings = np.array(final_embeddings).reshape(5,2,-1)
  final_embeddings[0][1] = embedding2
  if difference != 0:
      embedding2_cent = all_centroids[result2]
      vector = embedding_cent-embedding2_cent
      embedding2 += vector
  final_embeddings[1][1] = embedding2    
  embedding_new, embedding2 = rotate_embedding_both(embedding,embedding2, 5, result1, result2, all_centroids, embedding_cent)
  final_embeddings[2][0] = embedding_new
  final_embeddings[2][1] = embedding2
  embedding_new, embedding2 = calculate_average_weight(embedding,embedding2,result1, result2,'one', weights, all_centroids)
  final_embeddings[3][1] = embedding2
  embedding_new, embedding2 = calculate_average_weight(embedding,embedding2,result1, result2,'both', weights, all_centroids)
  final_embeddings[4][0] = embedding
  final_embeddings[4][1] = embedding2
  dif_sim = compute_cosine_similarities(final_embeddings)
  return dif_sim

def compare_embeddings(df, df1):
    if len(df) != len(df1):
        return False  # Različna dolžina, torej niso enaki

    for i in range(len(df)):
        emb1 = df['embedding'].iloc[i]
        emb2 = df1['embedding'].iloc[i]

        if not np.allclose(emb1, emb2, atol=1e-5):
            return False

    return True

def main():
  df = load_data_df('/home/rokp/test/dataset/mtcnn/vgg-vgg/20241111_125030_vgg_cplfw/vgg-vggmtcnn_images_mtcnn_cplfw.npz')
  centroid_directory = '/home/rokp/test/bulk/20241118_080229_cent_vgg_vsi_kot_svetlobe'
  out_dir = '/home/rokp/test/ROC'
  total = 1
  loop = False
  columns = df.shape[1]
  if columns == 4:
    print("Angles are")
    angles_are = True
  elif columns == 3:
    print("No angles")
    angles_are = False
  else:
    raise ValueError("Length of df is not typical")
  
  b_check_all = True
  descriptions = ["Normal", "Rotate One", "Rotate Both", "Average one", "Average both"]

  if angles_are:
    cent_angles = df['angle'].unique().tolist()
    cent_angles.sort()
    angles = cent_angles
  else:
    cent_angles = get_all_angles()#
    angles = None

  '''eigenvectors = np.load('/home/rokp/test/test/values/eigenvectors.npy')
  eigenvalues = np.load('/home/rokp/test/test/values/eigenvalues.npy')
  base_vectors = eigenvectors[:,:2]  
  VtV_inv = np.linalg.inv(base_vectors.T @ base_vectors)  # Inverz matrike'''

  all_centroids = []
  #selection = [-90,-85,-80,-30,0,30,80,85,90]
  for i in range(len(cent_angles)):
      #if all_angles[i] in selection:
        centroid_str =  f"Centroid_{ang_to_str(cent_angles[i])}.npy"
        centroid_dir = find_centroid(centroid_str, centroid_directory)
        vector = np.load(centroid_dir)
        #vector = apply_base(vector, base_vectors, VtV_inv)
        #vector /=np.linalg.norm(vector)
        all_centroids.append(vector)
  
  global_mean = np.load(find_centroid("global.npy", centroid_directory)) 
  #global_mean = np.mean(np.vstack(df['embedding']), axis=0).reshape(1, -1)
  df['embedding'] = df['embedding'].apply(lambda x: x - global_mean)

  current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
  filepath = create_directory(os.path.join(out_dir, current_date))

  all_arrea = []
  eer_all = []
  tpr_01_all = []
  tpr_1_all = []

  if not loop:
    total = 1

  for i in range(total):
    print(f"{i}/{total}")
    #random picture of person, comparison with pictures of that person
    sim_one_arr = np.array(df.groupby('person').apply(lambda x: x.sample(1)).reset_index(drop=True).copy(deep=True).values)
    sim_cleaned = clean_df(df, sim_one_arr, angles = angles)

    df_cleaned = pd.DataFrame(sim_cleaned)

    #one random picture taken df_cleaned for every person and compared between different identities
    dif_one_arr = np.array(df_cleaned.groupby(0).apply(lambda x: x.sample(1)).reset_index(drop=True).values)
    dif_cleaned = clean_df(df, dif_one_arr, angles= angles)

    similarity1, wrong1, right1 = check(sim_cleaned, sim_one_arr, 'sim', all_centroids, angles= angles, all = b_check_all)#
    difference1, wrong2, right2 = check(dif_cleaned, dif_one_arr,'dif', all_centroids, angles= angles, all=b_check_all)#
  
    similarity_scores = np.array(similarity1 + difference1)
    true_labels =np.concatenate((np.ones([1, len(similarity1)]), np.zeros([1, len(difference1)])), axis = 1)
    true_labels = true_labels.ravel()
    #similarity_scores = similarity_scores.ravel()
    #similarity_scores = np.abs(similarity_scores)

    eer_tmp = []
    tpr01_tmp = []
    tpr1_tmp = []
    for i in range(similarity_scores.shape[1]):
      tmp_scores = similarity_scores[:, i].ravel()
      fpr, tpr, thresholds = roc_curve(true_labels, tmp_scores) 
      eer, tpr01, tpr1 = calculate_needed(fpr, tpr)
      eer_tmp.append(eer)
      tpr01_tmp.append(tpr01)
      tpr1_tmp.append(tpr1)

    eer_all.append(eer_tmp)
    tpr_01_all.append(tpr01_tmp)
    tpr_1_all.append(tpr1_tmp)


  eer_all_array = np.array(eer_all)
  tpr_01_all_array = np.array(tpr_01_all)
  tpr_1_all_array = np.array(tpr_1_all)

  eer_df = process_table(eer_all_array, "EER")
  tpr_01_df = process_table(tpr_01_all_array, "TPR_01")
  tpr_1_df = process_table(tpr_1_all_array, "TPR_1")

  final_df = pd.concat([eer_df, tpr_01_df, tpr_1_df], axis=1) 

  final_df['Description'] = descriptions[:similarity_scores.shape[1]]
  final_df.to_excel(os.path.join(filepath,"data.xlsx"), index=False)
  similarity1 = np.array(similarity1)
  difference1 = np.array(difference1)
  for i in range(similarity_scores.shape[1]):  
    directory = os.path.join(filepath, descriptions[i])
    os.makedirs(directory, exist_ok=True)
    overlap_area = save_graphs(similarity1[:,i], difference1[:,i], directory)
  print(f'Saved all in {filepath}')

def process_table(data, table_name):
    # Povprečje po vrsticah
    mean_values = np.mean(data, axis=0)
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)
    
    # Združimo vse v eno DataFrame
    result_df = pd.DataFrame({
        f"{table_name}_mean": mean_values,
        f"{table_name}_max": max_values,
        f"{table_name}_min": min_values
    })
    return result_df
   

  
if __name__ == '__main__':
  main()


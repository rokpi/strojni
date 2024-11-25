import numpy as np
from utils_new import load_data_df, cosine_similarity, create_directory, get_all_angles, ang_to_str
from utils_test import find_matrix
from test_rotation_space import find_centroid
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

def clean_df(df, array):
  img_dirs = array[:, 1]
  df.columns = ['person', 'img_dir','angle', 'embedding'] #
  mask = ~df.set_index(['img_dir']).index.isin(
    img_dirs
  )
  cleaned_arr = np.array(df[mask].reset_index(drop=True).values)
  
  return cleaned_arr  

def calculate_needed(fpr, tpr):
  mask_eer = fpr + tpr -1
  eer_idx = int(np.argmin(np.abs(mask_eer)))
  fpr_01_mask = fpr-0.001
  fpr_01_idx = int(np.argmin(np.abs(fpr_01_mask)))
  fpr_1_mask = fpr-0.01
  fpr_1_idx = int(np.argmin(np.abs(fpr_1_mask)))

  print(f"eer:({fpr[eer_idx]:.5f},{tpr[eer_idx]:.5f})")
  print(f"fpr01:({fpr[fpr_01_idx]:.5f},{tpr[fpr_01_idx]:.5f})")
  print(f"fpr1:({fpr[fpr_1_idx]:.5f},{tpr[fpr_1_idx]:.5f})")
  return tpr[eer_idx], tpr[fpr_01_idx], tpr[fpr_1_idx]

def save_graphs(similarity1, difference1, out_directory):
  similarity_scores = np.array(similarity1 + difference1)
  true_labels =np.concatenate((np.ones([1, len(similarity1)]), np.zeros([1, len(difference1)])), axis = 1)
  true_labels = true_labels.ravel()
  similarity_scores = similarity_scores.ravel()
  similarity_scores = np.abs(similarity_scores)

  fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)  
  calculate_needed(fpr, tpr)
  # Izračunamo AUC (Area Under the Curve)
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

def check(array_cleaned, array_test, what, all_centroids, angles = None):
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
            indices = np.random.choice(array_copy.shape[0], num_random, replace=False)
            test_group = array_copy[indices]
            array_copy = np.delete(array_copy, indices, axis=0)
        elif what == 'sim':
            # same person
            test_group = array_copy[array_copy[:, person_num] == person]
        else:
            raise ValueError('Wrong input of what to check.')

        length = len(test_group)
        all_count -= length

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

        for j in range(length):
            embedding2 = embeddings2[j]
            if angles:
              angle2 = angles2[j]

              num = 0
              if angles[result1] != angle:
                  num += 1
              if angles[results2[j]] != angle2:
                  num += 1

            #if it is not the same angle
            #embedding_new = embedding
            '''if results[j] != 0:
                embedding2_cent = all_centroids[results2[j]]
                vector = embedding_cent-embedding2_cent
                embedding2 += vector'''
            #embedding_new, embedding2 = rotate_embedding_both(embedding,embedding2, 5, result1, results2[j], all_centroids, embedding_cent)
            embedding_new, embedding2 = calculate_average_weight(embedding,embedding2,result1, results2[j],'both', weights, all_centroids)

            if num > 2:
                raise ValueError("Preveč napak")

            right += 2 - num
            wrong += num
            dif_sim = cosine_similarity(embedding_new, embedding2)
            difference.append(dif_sim)

        if all_count <= 1:
            break

    return difference, wrong, right


def main():
  df = load_data_df('/home/rokp/test/test/dataset/mtcnn/adaface/20241125_125326_ada_vse/adaface_images_mtcnn.npz')
  centroid_directory = '/home/rokp/test/bulk/20241125_125803_cent_ada_vse'
  out_dir = '/home/rokp/test/test/ROC'
  total = 50
  loop = False
  all_angles = get_all_angles()#df['angle'].unique().tolist()
  all_angles.sort()
  all_centroids = []
  #selection = [-90,-85,-80,-30,0,30,80,85,90]
  for i in range(len(all_angles)):
      #if all_angles[i] in selection:
        centroid_str =  f"Centroid_{ang_to_str(all_angles[i])}.npy"
        centroid_dir = find_centroid(centroid_str, centroid_directory)
        vector = np.load(centroid_dir)
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
    sim_cleaned = clean_df(df, sim_one_arr)

    df_cleaned = pd.DataFrame(sim_cleaned)

    #one random picture taken df_cleaned for every person and compared between different identities
    dif_one_arr = np.array(df_cleaned.groupby(0).apply(lambda x: x.sample(1)).reset_index(drop=True).values)
    dif_cleaned = clean_df(df, dif_one_arr)

    similarity1, wrong1, right1 = check(sim_cleaned, sim_one_arr, 'sim', all_centroids, angles=all_angles)#
    difference1, wrong2, right2 = check(dif_cleaned, dif_one_arr,'dif', all_centroids, angles=all_angles)#

    similarity_scores = np.array(similarity1 + difference1)
    true_labels =np.concatenate((np.ones([1, len(similarity1)]), np.zeros([1, len(difference1)])), axis = 1)
    true_labels = true_labels.ravel()
    similarity_scores = similarity_scores.ravel()
    similarity_scores = np.abs(similarity_scores)
    fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores) 
    eer, tpr01, tpr1 = calculate_needed(fpr, tpr)
    eer_all.append(eer)
    tpr_01_all.append(tpr01)
    tpr_1_all.append(tpr1)

  with open(os.path.join(filepath, "roc_data.txt"), "w") as file:
    file.write("eer\n")
    file.write(f"{np.mean(eer_all):.5f}\n")
    file.write(f"{np.min(eer_all):.5f}\n")
    file.write(f"{np.max(eer_all):.5f}\n")
    file.write("tpr_01\n")    
    file.write(f"{np.mean(tpr_01_all):.5f}\n")
    file.write(f"{np.min(tpr_01_all):.5f}\n")
    file.write(f"{np.max(tpr_01_all):.5f}\n")
    file.write("tpr_1\n")    
    file.write(f"{np.mean(tpr_1_all):.5f}\n")
    file.write(f"{np.min(tpr_1_all):.5f}\n")
    file.write(f"{np.max(tpr_1_all):.5f}\n")
  
  '''print(f"eer\n  Avg: {np.mean(eer_all):.5f}\n  Min: {np.min(eer_all):.5f}\n  Max: {np.max(eer_all):.5f}")
  print(f"tpr_01\n  Avg: {np.mean(tpr_01_all):.5f}\n  Min: {np.min(tpr_01_all):.5f}\n  Max: {np.max(tpr_01_all):.5f}")
  print(f"tpr_1\n  Avg: {np.mean(tpr_1_all):.5f}\n  Min: {np.min(tpr_1_all):.5f}\n  Max: {np.max(tpr_1_all):.5f}")'''
  #out_directory = create_directory(os.path.join(filepath, f"{i}_from_{total}"))
  overlap_area = save_graphs(similarity1, difference1, filepath)
  #my_ROC(similarity1, difference1, filepath)
  '''bins = np.linspace(-0.3, 1, 10000)

  hist1, _ = np.histogram(similarity1, bins=bins, density=True)
  hist2, _ = np.histogram(difference1, bins=bins, density=True)
  overlap_area = np.sum(np.minimum(hist1, hist2) * np.diff(bins))
  all_arrea.append(overlap_area)

  print(all_arrea)
  print(f"Avg: {np.mean(all_arrea):.5f}\nMin: {np.min(all_arrea):.5f}\nMax: {np.max(all_arrea):.5f}")
  print("Similarity")
  print(f"Right: {right1}\nWrong: {wrong1}\nPercent: {right1/(right1 + wrong1)}")

  print("\nDifference")
  print(f"Right: {right2}\nWrong: {wrong2}\nPercent: {right2/(right2 + wrong2)}")'''






       

  
if __name__ == '__main__':
  main()


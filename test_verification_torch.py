import numpy as np
from my_utils.utils_new import find_centroid, load_data_df, create_directory, get_all_angles, ang_to_str
from my_utils.utils_pca import get_pca_vectors
from tqdm import tqdm
import pandas as pd
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader

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

  '''bins = np.linspace(-0.3, 1, 10000)

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
  plt.savefig(f"{out_directory}/dist_all.jpg", format="jpg")'''

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {
            'person': row[0],
            'img_dir': row[1],
        }
        if 'angle' in row:
          data['angle'] = torch.tensor(row['angle'], dtype=torch.float32)
          data['embedding'] = torch.tensor(row[3], dtype=torch.float32)
        else:
          data['embedding'] = torch.tensor(row[2], dtype=torch.float32)
        return data

def check_torch(array_cleaned, array_test, what, all_centroids, angles = None, all = True):
  all_count = len(array_cleaned)
  #weights = [10,8,6,4,0,2,0,4,6,8,10]
  weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

  if angles:
    person_num = 0
    embedding_num = 3
    angle_num = 2
  else:
      person_num = 0
      embedding_num = 2

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  array_copy = array_cleaned.copy()
  wrong = 0
  right = 0
  list_normal = []
  list_rot_one = []
  list_rot_both = []
  list_avg_one = []
  list_avg_both = []
  difference = []
  num_random = math.ceil(len(array_copy) / len(array_test))
  df_test = pd.DataFrame(array_test)
  dataset = EmbeddingDataset(df_test)
  dataloader = DataLoader(dataset, batch_size=256, num_workers=6, shuffle=False)
  all_centroids_torch = torch.tensor(all_centroids, device = device, dtype=torch.float32)
  all_centroids_cpu = torch.tensor(all_centroids, dtype=torch.float32)
  weights = torch.tensor(np.array(weights), device = device, dtype=torch.float32)

  with torch.no_grad():
      for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        person = batch['person']
        all_embedding = batch['embedding'].to(device)

        for i in range(len(person)):
          if what == 'dif':
            test_group = (array_copy[array_copy[:, person_num] != person[i]])#[:100]
          else: 
            test_group = (array_copy[array_copy[:, person_num] == person[i]])
          test_group_embeddings = [torch.tensor(embedding) for embedding in test_group[:, embedding_num]]
          embeddings2 = torch.vstack(test_group_embeddings).to(device)

          #if angles:
            #angle = batch['angle']
            #angles2 = test_group[:, angle_num]

          # Compute distances for the test embedding
          vectors1 = all_embedding[i] - all_centroids_torch
          distances1 = torch.norm(vectors1, dim=1)
          result1 = int(torch.argmin(distances1).item())

          # Compute distances for all test group embeddings
          vectors2 = embeddings2[:, None, :] - all_centroids_torch
          distances2 = torch.norm(vectors2, dim=2)
          results2 = torch.argmin(distances2, dim=1)

          embedding_cent = all_centroids_torch[result1][None,:]#.clone().detach().to(device).to(torch.float32)
          results = result1 - results2

          #embeddings2 = embeddings2.clone().detach().to(device).to(torch.float32)
          embedding = all_embedding[i]
          #results2 = results2.clone().detach().to(device).to(torch.long)
          
          # Rotate embedding one
          base_no_rot = (results != 0).view(-1, 1)
          mask_no_rot = base_no_rot.repeat(1, embedding.shape[1])#.to(device)  # (length, 512)
          embedding2_cents = all_centroids_torch[results2]
          vectors_rot_one = mask_no_rot * (embedding_cent - embedding2_cents)
          emb2_rot_one = embeddings2 + vectors_rot_one

          # Rotate embedding both
          target_idx = 5
          target_idx = torch.tensor(target_idx, device=device)
          mask_target_rot1 = (result1 != target_idx).repeat(1, embedding.shape[1])
          mask_target_rot2 = (results2 != target_idx).view(-1, 1).repeat(1, embedding.shape[1])
          emb1_rot_both = embedding + mask_target_rot1 * all_centroids_torch[target_idx]
          emb2_rot_both = embeddings2 + mask_target_rot2 * all_centroids_torch[target_idx]

          # Calculate average weight
          mask_avg = torch.arange(all_centroids_torch.size(0), device=all_centroids_torch.device) != result1
          filtered_centroids = all_centroids_torch[mask_avg]
          vectors1_avg = filtered_centroids - all_centroids_torch[result1].unsqueeze(0)
          vectors1_avg = torch.cat([all_centroids_torch[:result1], all_centroids_torch[result1 + 1:]])
          embs1_avg = torch.cat((embedding, embedding + vectors1_avg), dim=0)
          weights_array = weights.view(-1, 1)
          weights_vec = weights_array.repeat(1, embedding.shape[1])
          sum_avg = weights.sum()
          embs1_avg = torch.sum((weights_vec * embs1_avg) / sum_avg, dim=0).view(1, -1)

          # Create matrix for each example
          mask = torch.ones((len(results2), all_centroids_torch.shape[0]), dtype=torch.bool, device=device)
          mask[torch.arange(len(results2)), results2] = False  # Exclude centroid at the current angle
          resulting_matrices = []
          resulting_matrices = torch.stack(
                        [all_centroids_torch[mask[i]].view(-1, all_centroids_torch.shape[1]) for i in range(len(results2))]
                    )
          # Process embeddings
          embeddings2_exp = embeddings2.unsqueeze(1)
          vectors2_avg = embeddings2_exp + resulting_matrices
          embs2_avg = torch.cat((embeddings2_exp, vectors2_avg), dim=1)
          weights_vec_exp = weights_vec.unsqueeze(0)
          embs2_avg = torch.sum((weights_vec_exp * embs2_avg) / sum_avg, dim=1)

          #normal
          sim_normal = compute_cosine_similarities_torch(embedding, embeddings2).reshape(len(results2), 1)
          #rotate one
          sim_rot_one = compute_cosine_similarities_torch(embedding, emb2_rot_one).reshape(len(results2), 1)    
          #rotate both
          sim_rot_both = compute_cosine_similarities_torch(emb1_rot_both, emb2_rot_both).reshape(len(results2), 1)
          #average one
          sim_avg_one = compute_cosine_similarities_torch(embs1_avg, embeddings2).reshape(len(results2), 1)
          #average both
          sim_avg_both = compute_cosine_similarities_torch(embs1_avg, embs2_avg).reshape(len(results2), 1)

          ##TORCH HSTACK
          stack = torch.hstack([sim_normal, sim_rot_one, sim_rot_both, sim_avg_one, sim_avg_both])
          difference.append(stack.cpu().numpy())
          '''list_normal.append(sim_normal.cpu().numpy())
          list_rot_one.append(sim_rot_one.cpu().numpy())
          list_rot_both.append(sim_rot_both.cpu().numpy())
          list_avg_one.append(sim_avg_one.cpu().numpy())
          list_avg_both.append(sim_avg_both.cpu().numpy())'''
          torch.cuda.empty_cache()
          del resulting_matrices
          del embedding
          del embeddings2
          del embedding_cent
          del embedding2_cents
          del mask_avg
          del mask
          del mask_no_rot
          del results2
  return difference#list_normal, list_rot_one, list_rot_both, list_avg_one, list_avg_both, wrong, right

class EmbeddingDatasetAll(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {
            'person': row[0],
            'img_dir': row[1],
            'results1': row['results1'],
            'results': row['results'],
            'centroid': row['centroid'],
        }
        if 'angle' in row:
          data['angle'] = torch.tensor(row['angle'], dtype=torch.float32)
          data['embedding'] = torch.tensor(row[3], dtype=torch.float32)
        else:
          data['embedding'] = torch.tensor(row[2], dtype=torch.float32)
        return data

def check_torch_all(array_cleaned, array_test, all_centroids, angles = None):
  all_count = len(array_cleaned)
  #weights = [10,8,6,4,0,2,0,4,6,8,10]
  weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

  if angles:
    person_num = 0
    embedding_num = 3
    angle_num = 2
  else:
      person_num = 0
      embedding_num = 2

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  array_copy = array_cleaned.copy()
  wrong = 0
  right = 0
  difference = []
  similarity =[]

  #average
  #result
  all_centroids_torch = torch.tensor(all_centroids, device = device, dtype=torch.float32)

  weights = torch.tensor(np.array(weights), device = device, dtype=torch.float32)
  embeddings2 = torch.tensor(np.vstack(array_copy[:, embedding_num]), device = device, dtype=torch.float32)
  embeddings1 = torch.tensor(np.vstack(array_test[:, embedding_num]), device = device, dtype=torch.float32)
  weights_array = weights.view(-1, 1)
  weights_vec = weights_array.repeat(1, embeddings1.shape[1])

  target_idx = 5
  target_idx = torch.tensor(target_idx, device=device)

  # Compute distances for the test embedding
  vectors1 = embeddings1[:, None, :] - all_centroids_torch
  distances1 = torch.norm(vectors1, dim=2)
  all_results1 = torch.argmin(distances1, dim=1)
  del vectors1
  del distances1

  # Compute distances for all test group embeddings
  vectors2 = embeddings2[:, None, :] - all_centroids_torch
  distances2 = torch.norm(vectors2, dim=2)
  results2 = torch.argmin(distances2, dim=1)
  del vectors2
  del distances2

  all_embeddings1_cent = all_centroids_torch[all_results1]#[None,:]#.clone().detach().to(device).to(torch.float32)
  all_results = (all_results1[:, None] - results2[None, :]).cpu().numpy()
  all_results1 = all_results1.cpu().numpy()

  df_test = pd.DataFrame(array_test)
  df_test['results1'] = all_results1
  df_test['results'] = list(all_results)
  df_test['centroid'] = list(all_embeddings1_cent.cpu().numpy())
  del all_embeddings1_cent

  eigenvectors = np.load('/home/rokp/test/bulk/20241217_133231/eigenvectors.npy')
  #eigenvalues = np.load('/home/rokp/test/bulk/20241217_133231/eigenvalues.npy')

  '''all_vectors = []
  for i in range(len(all_centroids_torch)-1):
    for j in range(len(all_centroids_torch[i+1:])):
      all_vectors.append(all_centroids_torch[i].unsqueeze(0)-all_centroids_torch[j].unsqueeze(0))
  all_vectors = torch.vstack(all_vectors)'''
  all_vectors = torch.tensor(eigenvectors[:2], device = device, dtype=torch.float32)
  P_mat_multiply =torch.vstack([torch.mm(all_vectors[i].unsqueeze(1), all_vectors[i].unsqueeze(1).T).unsqueeze(0) for i in range(len(all_vectors))])
  P_sum = torch.sum(P_mat_multiply, dim = 0)
  P = torch.eye(all_vectors.shape[1], dtype=torch.float32).to(device).to(device)-P_sum
  P_embeddings2 = torch.mm(P, embeddings2.T).T
  del P_mat_multiply
  del P_sum

  dataset = EmbeddingDatasetAll(df_test)
  dataloader = DataLoader(dataset, batch_size=150, num_workers=6, shuffle=False)
  with torch.no_grad():
      for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        person = batch['person']
        results = batch['results'].to(device)
        results1 = batch['results1'].to(device)
        embeddings1 = batch['embedding'].squeeze(1).to(device)
        embeddings1_cent = batch['centroid'].to(device)

        # Rotate embedding one
        mask_no_rot = results.unsqueeze(2) != 0
        mask_no_rot = mask_no_rot.repeat(1,1,embeddings1.shape[1])#.to(device)  # (length, 512)
        cent_emb2_rot = all_centroids_torch[results2]#torch.vstack([all_centroids_torch[result2].unsqueeze(0) for result2 in results2])
        vectors_rot_one = mask_no_rot * (embeddings1_cent.unsqueeze(1)-cent_emb2_rot.unsqueeze(0))
        emb2_rot_one = embeddings2.unsqueeze(0) + vectors_rot_one
        del mask_no_rot
        del vectors_rot_one

        # Rotate embedding both
        mask_target_rot1 = (results1 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
        mask_target_rot2 = (results2 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
        emb1_rot_both = embeddings1 + mask_target_rot1 * all_centroids_torch[target_idx]
        emb2_rot_both = embeddings2 + mask_target_rot2 * all_centroids_torch[target_idx]
        del mask_target_rot1
        del mask_target_rot2

        # Calculate average weight
        #mask_avg = torch.arange(all_centroids_torch.size(0), device=device).unsqueeze(0) != results1.unsqueeze(1)
        #filtered_centroids = torch.vstack([all_centroids_torch[mask].unsqueeze(0) for mask in mask_avg])
        #vectors1_avg = filtered_centroids - all_centroids_torch[results1].unsqueeze(1)
        vectors1_avg = torch.vstack([torch.cat([all_centroids_torch[:results1[i]], all_centroids_torch[results1[i] + 1:]]).unsqueeze(0) for i in range(len(results1))])
        embs1_avg = torch.cat((embeddings1.unsqueeze(1), embeddings1.unsqueeze(1) + vectors1_avg), dim=1)
        sum_avg = weights.sum()
        embs1_avg = torch.sum((weights_vec.unsqueeze(0) * embs1_avg), dim=1)/ sum_avg
        del vectors1_avg

        vectors2_avg = torch.vstack([torch.cat([all_centroids_torch[:results2[i]], all_centroids_torch[results2[i] + 1:]]).unsqueeze(0) for i in range(len(results2))])
        embs2_avg = torch.cat((embeddings2.unsqueeze(1), embeddings2.unsqueeze(1) + vectors2_avg), dim=1)
        embs2_avg = torch.sum((weights_vec.unsqueeze(0) * embs2_avg), dim=1)/ sum_avg
        del vectors2_avg

        shape = [len(embeddings1),len(results2), 1]
        #normal
        normal = compute_cosine_similarities_torch(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0)).reshape(shape)
        #rotate one
        rot_one = compute_cosine_similarities_torch(embeddings1.unsqueeze(1), emb2_rot_one).reshape(shape)    
        #rotate both
        rot_both = compute_cosine_similarities_torch(emb1_rot_both.unsqueeze(1), emb2_rot_both.unsqueeze(0)).reshape(shape)
        #average one
        avg_one = compute_cosine_similarities_torch(embs1_avg.unsqueeze(1), embeddings2.unsqueeze(0)).reshape(shape)
        #average both
        avg_both = compute_cosine_similarities_torch(embs1_avg.unsqueeze(1), embs2_avg.unsqueeze(0)).reshape(shape)


        del emb2_rot_one
        del emb1_rot_both
        del emb2_rot_both
        del embs1_avg
        del embs2_avg

        P_embeddings1 = torch.mm(P, embeddings1.T).T
        P_normal = compute_cosine_similarities_torch(P_embeddings1.unsqueeze(1), P_embeddings2.unsqueeze(0)).reshape(shape)
        del embeddings1

        stack_dif = []
        stack_sim = []
        for i in range(len(person)):
          mask_diff = torch.tensor(array_copy[:, person_num] != person[i])
          mask_sim = ~mask_diff
          stack_dif = torch.hstack([normal[i][mask_diff],rot_one[i][mask_diff], rot_both[i][mask_diff], avg_one[i][mask_diff], avg_both[i][mask_diff], P_normal[i][mask_diff]])
          stack_sim = torch.hstack([normal[i][mask_sim],rot_one[i][mask_sim], rot_both[i][mask_sim], avg_one[i][mask_sim], avg_both[i][mask_sim], P_normal[i][mask_sim]])
          difference.append(stack_dif.cpu().numpy())
          similarity.append(stack_sim.cpu().numpy())     
        
        torch.cuda.empty_cache()
        del person
        del results
        del results1
        del embeddings1_cent
  return difference, similarity

def check(array_cleaned, array_test, what, all_centroids, angles = None, all = True):
    all_count = len(array_cleaned)
    #weights = [10,8,6,4,0,2,0,4,6,8,10]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    array_copy = array_cleaned.copy()
    wrong = 0
    right = 0
    difference = []
    num_random = math.ceil(len(array_copy) / len(array_test))

    for i in tqdm(range(len(array_test)), total=len(array_test)): 
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

        all_centroids = np.array(all_centroids)
        embedding_shape = (1, embedding.shape[1])

        #rotate embedding one
        base_no_rot = (results != 0).reshape(len(results), 1)
        mask_no_rot = np.tile(base_no_rot, embedding_shape) #(length, 512)
        embedding2_cents = all_centroids[results2]
        vectors_rot_one = mask_no_rot*(embedding_cent-embedding2_cents)
        emb2_rot_one = embeddings2 + vectors_rot_one

        #rotate embedding both
        target_idx = 5
        mask_target_rot1 = result1 != target_idx
        mask_target_rot1 = np.tile(mask_target_rot1, embedding_shape)
        mask_target_rot2 = (results2 != target_idx).reshape(len(results2), 1)
        mask_target_rot2 = np.tile(mask_target_rot2, embedding_shape) #maska 1 ali 0
        emb1_rot_both = embedding + mask_target_rot1*all_centroids[target_idx]
        emb2_rot_both = embeddings2 + mask_target_rot2*all_centroids[target_idx]

        #calculate average weight
        vectors1_avg = np.delete(all_centroids, result1, axis = 0)- all_centroids[result1]
        embs1_avg = np.concatenate((embedding, embedding + vectors1_avg), axis = 0)
        weights_array = np.array(weights).reshape(len(weights), 1)
        weights_vec = np.tile(weights_array, embedding_shape)
        sum_avg = np.sum(weights)
        embs1_avg = np.sum((weights_vec*embs1_avg)/sum_avg, axis = 0).reshape(embedding_shape)

        #Naredi matriko kateri bodo uporabljeni za vsak primer
        mask = np.ones((len(results2), all_centroids.shape[0]), dtype=bool)
        mask[np.arange(len(results2)), results2] = False #ne smemo uporabiti centroide kota na katerem smo
        resulting_matrices = [all_centroids[mask[i]].reshape(all_centroids.shape[0] - 1, all_centroids.shape[1])
                              for i in range(len(results2))]
        resulting_matrices = np.array(resulting_matrices)
        #resulting_matrices = all_centroids[mask].reshape(len(results2), all_centroids.shape[0] - 1, all_centroids.shape[1])# Uporabi maske za selekcijo

        embeddings2_exp = embeddings2[:, np.newaxis, :]
        vectors2_avg = embeddings2_exp + resulting_matrices
        embs2_avg = np.concatenate((embeddings2_exp, vectors2_avg), axis = 1)
        weights_vec_exp = weights_vec[np.newaxis, :]
        embs2_avg = np.sum((weights_vec_exp*embs2_avg)/sum_avg, axis = 1)

        #normal
        sim_normal = compute_cosine_similarities(embedding, embeddings2).reshape(len(results2))
        #rotate one
        sim_rot_one = compute_cosine_similarities(embedding, emb2_rot_one).reshape(len(results2))    
        #rotate both
        sim_rot_both = compute_cosine_similarities(emb1_rot_both, emb2_rot_both).reshape(len(results2))
        #average one
        sim_avg_one = compute_cosine_similarities(embs1_avg, embeddings2).reshape(len(results2))
        #average both
        sim_avg_both = compute_cosine_similarities(embs1_avg, embs2_avg).reshape(len(results2))

        difference.append(np.concatenate((sim_normal, sim_rot_one, sim_rot_both, sim_avg_one, sim_avg_both)))

    return difference, wrong, right

def compute_cosine_similarities_torch(embeddings1, embeddings2, epsilon=1e-8):
    # Norme embeddingov
    norm1 = torch.norm(embeddings1, dim=2, keepdim=True)
    norm2 = torch.norm(embeddings2, dim=2, keepdim=True)

    # Kosinusna podobnost
    dot_products = torch.sum(embeddings1 * embeddings2, dim=2, keepdim=True)
    cosine_similarities = dot_products / (torch.clamp(norm1 * norm2, min=epsilon))
    
    return cosine_similarities.squeeze()

def compute_cosine_similarities(embeddings1, embeddings2, epsilon=1e-8):
    
    # Norme embeddingov
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Kosinusna podobnost
    dot_products = np.sum(embeddings1 * embeddings2, axis=1, keepdims=True)
    cosine_similarities = dot_products / (np.maximum(norm1 * norm2, epsilon))
    
    return cosine_similarities.squeeze()

def main():
  df = load_data_df('/home/rokp/test/dataset/arcface/cplfw/arcface.npz')

  centroid_directory = '/home/rokp/test/bulk/20241203_161446_cent_arcface_vse'
  out_dir = '/home/rokp/test/ROC'
  total = 5
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
  descriptions = ["Normal", "Rotate One", "Rotate Both", "Average one", "Average both", "P_normal"]

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
  tpr_001_all = []
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
    #list_normal1, list_rot_one1, list_rot_both1, list_avg_one1, list_avg_both1, wrong1, right1 = check_torch(sim_cleaned, sim_one_arr, 'sim', all_centroids, angles= angles, all = b_check_all)#
    #list_normal, list_rot_one, list_rot_both, list_avg_one, list_avg_both, wrong2, right2 = check_torch(dif_cleaned, dif_one_arr,'dif', all_centroids, angles= angles, all=b_check_all)#

    #similarity = check_torch_all(sim_cleaned, sim_one_arr, 'sim', all_centroids, angles= angles, all = b_check_all)
    #difference = check_torch_all(dif_cleaned, dif_one_arr,'dif', all_centroids, angles= angles, all=b_check_all)

    difference, similarity = check_torch_all(sim_cleaned, sim_one_arr, all_centroids, angles= angles)
    array_sim = np.concatenate(similarity)
    array_diff = np.concatenate(difference)
    similarity_scores = np.concatenate((array_sim, array_diff), axis = 0)
    true_labels =np.concatenate((np.ones([1, len(array_sim)]), np.zeros([1, len(array_diff)])), axis = 1)
    true_labels = true_labels.ravel()

    eer_tmp = []
    tpr001_tmp = []
    tpr01_tmp = []
    tpr1_tmp = []
    for i in range(similarity_scores.shape[1]):
      tmp_scores = similarity_scores[:, i].ravel()
      fpr, tpr, thresholds = roc_curve(true_labels, tmp_scores) 
      eer, tpr001, tpr01, tpr1 = calculate_needed(fpr, tpr)
      eer_tmp.append(eer)
      tpr001_tmp.append(tpr001)
      tpr01_tmp.append(tpr01)
      tpr1_tmp.append(tpr1)

    eer_all.append(eer_tmp)
    tpr_001_all.append(tpr001_tmp)
    tpr_01_all.append(tpr01_tmp)
    tpr_1_all.append(tpr1_tmp)


  eer_all_array = np.array(eer_all)
  tpr_001_all_array = np.array(tpr_001_all)
  tpr_01_all_array = np.array(tpr_01_all)
  tpr_1_all_array = np.array(tpr_1_all)

  eer_df = process_table(eer_all_array, "EER")
  tpr_001_df = process_table(tpr_001_all_array, "TPR_001")
  tpr_01_df = process_table(tpr_01_all_array, "TPR_01")
  tpr_1_df = process_table(tpr_1_all_array, "TPR_1")

  final_df = pd.concat([eer_df, tpr_001_df, tpr_01_df, tpr_1_df], axis=1) 

  final_df['Description'] = descriptions[:similarity_scores.shape[1]]
  final_df.to_excel(os.path.join(filepath,"data.xlsx"), index=False)
  for i in range(similarity_scores.shape[1]):  
    directory = os.path.join(filepath, descriptions[i])
    os.makedirs(directory, exist_ok=True)
    save_graphs(array_sim[:,i], array_diff[:,i], directory)
  print(f'Saved all in {filepath}')

def process_table(data, table_name):
    # Povprečje po vrsticah
    mean_values = np.mean(data, axis=0)
    #max_values = np.max(data, axis=0)
    #min_values = np.min(data, axis=0)
    
    # Združimo vse v eno DataFrame
    result_df = pd.DataFrame({
        f"{table_name}_mean": mean_values,
        #f"{table_name}_max": max_values,
        #f"{table_name}_min": min_values
    })
    return result_df
   

  
if __name__ == '__main__':
  main()


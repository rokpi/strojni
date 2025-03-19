import numpy as np
from my_utils.utils_new import find_centroid, load_data_df, create_directory, get_all_angles,get_all_lights, ang_to_str
from tqdm import tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader
from openpyxl import Workbook
from sklearn.decomposition import PCA

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

def save_graphs(fpr, tpr, out_directory):
  '''similarity_scores = np.concatenate((similarity1,difference1))
  true_labels =np.concatenate((np.ones([1, len(similarity1)]), np.zeros([1, len(difference1)])), axis = 1)
  true_labels = true_labels.ravel()
  similarity_scores = similarity_scores.ravel()
  fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)''' 
  #roc_auc = auc(fpr, tpr)

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
            #'results': row['results'],
            'centroid': row['centroid'],
            'original_index': idx
        }
        if 'angle' in row:
          data['angle'] = torch.tensor(row['angle'], dtype=torch.float32)
          data['embedding'] = torch.tensor(row['embedding'], dtype=torch.float32)
        else:
          data['embedding'] = torch.tensor(row[2], dtype=torch.float32)
        return data

def get_P(eigenvectors, num_vectors, device):
  #get number of eigenvectors equal to num_vectors
  all_vectors = torch.tensor(eigenvectors[:num_vectors], device = device, dtype=torch.float32)
  #multiply vector * vector.T
  P_mat_multiply =torch.vstack([torch.mm(all_vectors[i].unsqueeze(1), all_vectors[i].unsqueeze(1).T).unsqueeze(0) for i in range(len(all_vectors))])
  #sum of vectors, I-sum
  P_sum = torch.sum(P_mat_multiply, dim = 0)
  P = torch.eye(all_vectors.shape[1], dtype=torch.float32).to(device)-P_sum
  del P_mat_multiply
  del P_sum
  return P

class Embeddings2Dataset(Dataset):
    def __init__(self, embeddings2):
        """
        Parametri:
          embeddings2: torch.Tensor, oblika (1, 7876, 3, 512)
        """
        # Odstranimo prvo dimenzijo, da dobimo tensor oblike (7876, 3, 512)
        self.embeddings = embeddings2.squeeze(0)
        
    def __len__(self):
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        # Vrne embedding oblike (1, 3, 512) – tako bo v DataLoaderju prvi dimenzija batcha
        sample = self.embeddings[idx]  # oblika: (3, 512)
        return sample.unsqueeze(0)     # oblika: (1, 3, 512)


def check_torch_all(df, weights, all_centroids, eigenvectors, angles_are, cent_type, go_PCA = True):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  wrong = 0
  right = 0
  difference = []
  similarity =[]
  #df = df.iloc[:100]
  array_all = df['embedding']

  df_copy = df.copy()
  #average
  #result
  all_centroids_torch = torch.tensor(all_centroids, device = device, dtype=torch.float32).squeeze()

  weights = torch.tensor(np.array(weights), dtype=torch.float32)

  np_array_all = np.vstack(array_all)

  if go_PCA:
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(np_array_all)

    '''pca = PCA(n_components=np_array_all.shape[1])
    X_pca = pca.fit_transform(np_array_all)

    # Nastavi prve tri PCA komponente na nič
    X_pca[:, :1] = 0'''

    # Rekonstruiraj podatke z uporabo obrnjenega PCA (dimenzionalnost ostane enaka)
    np_array_all = pca.inverse_transform(X_pca)
    df['embedding'] = np_array_all.tolist()

  embeddings2 = torch.tensor(np_array_all, device = device, dtype=torch.float32).squeeze()

  weights_array = weights.view(-1, 1)
  weights_vec = weights_array.repeat(1, embeddings2.shape[1]).to(device)
  target_idx = 5
  target_idx = torch.tensor(target_idx, device=device)

  # Compute distances for all test group embeddings
  vectors2 = embeddings2[:, None, :] - all_centroids_torch
  distances2 = torch.norm(vectors2, dim=2)
  results2 = torch.argmin(distances2, dim=1)
  del vectors2
  del distances2

  if angles_are:
    unique_values = df[cent_type].unique()
    unique_values.sort()
    unique_values = torch.tensor(np.vstack(unique_values), device = device, dtype=torch.float32)
    tensor_list = np.array(df[cent_type], dtype=int)
    test = unique_values[results2].squeeze(1)
    array_right = torch.tensor(tensor_list, device=device)
    mask_right = torch.zeros_like(array_right, dtype=torch.bool)
    mask_right = (array_right == test)
    right = torch.sum(mask_right).cpu().item()
    wrong = len(results2)-right
    print(f"Pravilno določeni: {right}")
    print(f"Napačno določeni: {wrong}")
    print(f"Procent pravilno: {(right/(right+wrong)*100)}%")
    del array_right
    del mask_right
    del unique_values


  all_embeddings1_cent = all_centroids_torch[results2].cpu().numpy()#[None,:]#.clone().detach().to(device).to(torch.float32)

  df_copy['results1'] = results2.cpu().numpy()
  df_copy['centroid'] = list(all_embeddings1_cent)
  del all_embeddings1_cent

  '''
  test_vectors = []
  
  if cent_type == 'angle':
    selection = [i for i in range(3,8) if i != 5]
    for j in selection:
      tmp_embed = all_centroids[5]-all_centroids[j]
      test_vectors.append(tmp_embed)

  elif cent_type == 'light':
    selection = [i for i in range(1,15) if i != 7] 
    for j in selection:
      tmp_embed = all_centroids[7]-all_centroids[j]
      test_vectors.append(tmp_embed)

  test_vectors = np.vstack(test_vectors)
  norms = np.linalg.norm(test_vectors, axis = 1)
  test_vectors = test_vectors / norms[:, np.newaxis]
  test_embedds_1 = np.vstack(test_vectors[7:][::-1])#od 9 naprej
  test_embedds_2 = np.vstack(test_vectors[:5])'''

  '''P21 = get_P(test_embedds_1, 5, device)
  P22 = get_P(test_embedds_1, 2, device)
  P23 = get_P(test_embedds_1, 1, device)

  P11 = get_P(test_embedds_2, 5, device)
  P12 = get_P(test_embedds_2, 2, device)
  P13 = get_P(test_embedds_2, 1, device)'''

  #P2, P_embeddings2_2 = get_P(eigenvectors, 2, embeddings2, device)
  #P3 = get_P(eigenvectors, 3, device)
  #P_embeddings2_3 = torch.mm(P3, embeddings2.T).T

  '''P4, P_embeddings2_4 = get_P(eigenvectors, 4, embeddings2, device)
  P5, P_embeddings2_5 = get_P(eigenvectors, 5, embeddings2, device)
  P10, P_embeddings2_10 = get_P(eigenvectors, 10, embeddings2, device)
  P25, P_embeddings2_25 = get_P(eigenvectors, 25, embeddings2, device)'''

  '''all_vectors = []
  for i in range(len(all_centroids_torch)-1):
    for j in range(len(all_centroids_torch[i+1:])):
      all_vectors.append(all_centroids_torch[i].unsqueeze(0)-all_centroids_torch[j].unsqueeze(0))
  all_vectors = torch.vstack(all_vectors)'''


  dataset = EmbeddingDatasetAll(df_copy)
  dataloader = DataLoader(dataset, batch_size = 100, num_workers=6, shuffle=False)
  with torch.no_grad():
      for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        person = batch['person']
        #results = batch['results'].to(device)
        original_indices = batch['original_index']
        results1 = batch['results1'].to(device)
        embeddings1 = batch['embedding'].squeeze(1).to(device)

        embeddings1_cent = batch['centroid'].to(device).squeeze()
        shape = [len(embeddings1),len(results2), 1]

        results = results1.unsqueeze(1)-results2.unsqueeze(0)
        # Rotate embedding one
        mask_no_rot = results.unsqueeze(2) != 0
        mask_no_rot = mask_no_rot.repeat(1,1,embeddings1.shape[1])#.to(device)  # (length, 512)
        cent_emb2_rot = all_centroids_torch[results2]#torch.vstack([all_centroids_torch[result2].unsqueeze(0) for result2 in results2])
        vectors_rot_one = mask_no_rot * (embeddings1_cent.unsqueeze(1)-cent_emb2_rot.unsqueeze(0))
        emb2_rot_one = embeddings2.unsqueeze(0) + vectors_rot_one
        del mask_no_rot
        del vectors_rot_one

        #rotate one
        rot_one = compute_cosine_similarities_torch(embeddings1.unsqueeze(1), emb2_rot_one).reshape(shape)    
        del emb2_rot_one

        # Rotate embedding both
        mask_target_rot1 = (results1 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
        mask_target_rot2 = (results2 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
        emb1_rot_both = embeddings1 + mask_target_rot1 * all_centroids_torch[target_idx]
        emb2_rot_both = embeddings2 + mask_target_rot2 * all_centroids_torch[target_idx]
        del mask_target_rot1
        del mask_target_rot2

        #rotate both
        rot_both = compute_cosine_similarities_torch(emb1_rot_both.unsqueeze(1), emb2_rot_both.unsqueeze(0)).reshape(shape)
        del emb1_rot_both
        del emb2_rot_both

        # Calculate average weight
        '''vectors1_avg = torch.vstack([torch.cat([all_centroids_torch[:results1[i]], all_centroids_torch[results1[i] + 1:]]).unsqueeze(0) for i in range(len(results1))])
        embs1_avg = torch.cat((embeddings1.unsqueeze(1), embeddings1.unsqueeze(1) + vectors1_avg), dim=1)
        sum_avg = weights.sum()
        embs1_avg = torch.sum((weights_vec.unsqueeze(0) * embs1_avg), dim=1)/ sum_avg
        del vectors1_avg

        vectors2_avg = torch.vstack([torch.cat([all_centroids_torch[:results2[i]], all_centroids_torch[results2[i] + 1:]]).unsqueeze(0) for i in range(len(results2))])
        embs2_avg = torch.cat((embeddings2.unsqueeze(1), embeddings2.unsqueeze(1) + vectors2_avg), dim=1)
        embs2_avg = torch.sum((weights_vec.unsqueeze(0) * embs2_avg), dim=1)/ sum_avg
        del vectors2_avg'''

        vectors1_avg = torch.vstack([torch.cat([all_centroids_torch[:results1[i]], all_centroids_torch[results1[i] + 1:]]).unsqueeze(0) for i in range(len(results1))])
        embs1_avg = torch.cat((embeddings1.unsqueeze(1), embeddings1.unsqueeze(1) + vectors1_avg), dim=1)
        del vectors1_avg
        
        vectors2_avg = torch.vstack([torch.cat([all_centroids_torch[:results2[i]], all_centroids_torch[results2[i] + 1:]]).unsqueeze(0) for i in range(len(results2))])
        embs2_avg = torch.cat((embeddings2.unsqueeze(1), embeddings2.unsqueeze(1) + vectors2_avg), dim=1)
        del vectors2_avg

        '''embs1_P = []
        for i in range(1,14):
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
            P_embeddings = torch.mm(P, embs1_avg[:, i, :].T).T  
            embs1_P.append(P_embeddings.unsqueeze(1))
          else:
            embs1_P.append(embs1_avg[:, i, :].unsqueeze(1)) 
        
        embs1_P = torch.cat(embs1_P, dim = 1)'''

        if cent_type == 'angle':
          start_idx = 3
          end_idx = 8
        if cent_type == 'light':
           start_idx = 5
           end_idx = 10

        sims_1 =[]
        sims_2 =[]
        sims_3 =[]
        dataset2 = Embeddings2Dataset(embs2_avg)
        batch_size = 256  # Nastavi batch size glede na razpoložljivo pomnilniško zmogljivost
        dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

        # Iteriramo čez batch-e embeddings2
        for batch2 in dataloader2:
          # DataLoader vrne batch oblike (batch_size, 1, 3, 512)
          # Za naš izračun potrebujemo embeddings2 oblike (1, batch_size, 3, 512)
          batch2 = batch2.transpose(0, 1)  # Zamenjamo prvo in drugo dimenzijo
          # Izračunamo kosinusne podobnosti za ta batch:
          #sims_1.append(compute_cosine_similarities_pairs_3(embs1_avg[:,1:5,:].unsqueeze(1), batch[:,:,1:5,:])) 
          sims_2.append(compute_cosine_similarities_pairs_3(embs1_avg[:,start_idx:end_idx,:].unsqueeze(1), batch2[:,:,5:10,:])) 
          #sims_3.append(compute_cosine_similarities_pairs_3(embs1_avg[:,10:15,:].unsqueeze(1), batch[:,:,10:15,:]))


        sum_avg = weights.sum()
        embs1_avg = torch.sum((weights_vec.unsqueeze(0) * embs1_avg), dim=1)/ sum_avg

        #average one
        avg_one = compute_cosine_similarities_torch(embs1_avg.unsqueeze(1), embeddings2.unsqueeze(0)).reshape(shape)


        embs2_avg = torch.sum((weights_vec.unsqueeze(0) * embs2_avg), dim=1)/ sum_avg

        #average both
        avg_both = compute_cosine_similarities_torch(embs1_avg.unsqueeze(1), embs2_avg.unsqueeze(0)).reshape(shape)
        del embs1_avg
        del embs2_avg

        #test_tensor_1 = torch.cat(sims_1, dim = 1)
        test_tensor_2 = torch.cat(sims_2, dim = 1)
        #test_tensor_3 = torch.cat(sims_3, dim = 1)

        sim_3 = (torch.mean(test_tensor_2, dim = 2)).reshape(shape)# + torch.mean(test_tensor_2, dim = 2)+ torch.mean(test_tensor_3, dim = 2)).reshape(shape)

        #normal
        normal = compute_cosine_similarities_torch(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0)).reshape(shape)

        #emb1_P3 = torch.mm(P3, embeddings1.T).T

        #sim_3 = compute_cosine_similarities_torch(emb1_P3.unsqueeze(1), P_embeddings2_3.unsqueeze(0)).reshape(shape)

        '''emb1_P2 = torch.mm(P2, embeddings1.T).T
        sim_2 = compute_cosine_similarities_torch(emb1_P2.unsqueeze(1), P_embeddings2_2.unsqueeze(0)).reshape(shape)

        emb1_P4 = torch.mm(P4, embeddings1.T).T
        sim_4 = compute_cosine_similarities_torch(emb1_P4.unsqueeze(1), P_embeddings2_4.unsqueeze(0)).reshape(shape)        

        emb1_P5 = torch.mm(P5, embeddings1.T).T
        sim_5 = compute_cosine_similarities_torch(emb1_P5.unsqueeze(1), P_embeddings2_5.unsqueeze(0)).reshape(shape)

        emb1_P10 = torch.mm(P10, embeddings1.T).T
        sim_10 = compute_cosine_similarities_torch(emb1_P10.unsqueeze(1), P_embeddings2_10.unsqueeze(0)).reshape(shape)

        emb1_P25 = torch.mm(P25, embeddings1.T).T
        sim_25 = compute_cosine_similarities_torch(emb1_P25.unsqueeze(1), P_embeddings2_25.unsqueeze(0)).reshape(shape)
        del emb1_P2
        del emb1_P4
        del emb1_P5
        del emb1_P10
        del emb1_P25'''

        #del emb1_P3
        del embeddings1

        for i in range(len(person)):
          
          mask_diff = torch.tensor((df['person'] != person[i]).to_numpy())
          mask_sim = ~mask_diff
          mask_sim[original_indices[i]] = False
          #stack_dif = torch.hstack([sim_2[i][mask_diff],sim_3[i][mask_diff], sim_4[i][mask_diff], sim_5[i][mask_diff], sim_10[i][mask_diff], sim_25[i][mask_diff]])
          #stack_sim = torch.hstack([sim_2[i][mask_sim],sim_3[i][mask_sim], sim_4[i][mask_sim], sim_5[i][mask_sim], sim_10[i][mask_sim], sim_25[i][mask_sim]])
          stack_dif = torch.hstack([normal[i][mask_diff],rot_one[i][mask_diff], rot_both[i][mask_diff], avg_one[i][mask_diff], avg_both[i][mask_diff], sim_3[i][mask_diff]])
          stack_sim = torch.hstack([normal[i][mask_sim],rot_one[i][mask_sim], rot_both[i][mask_sim], avg_one[i][mask_sim], avg_both[i][mask_sim], sim_3[i][mask_sim]])
          difference.append(stack_dif.cpu().numpy())
          similarity.append(stack_sim.cpu().numpy())     
        
        torch.cuda.empty_cache()
        del person
        del results
        del results1
        del embeddings1_cent
        del mask_diff
        del mask_sim
        del stack_dif
        del stack_sim
  del embeddings2
  del all_centroids_torch
  del weights
  del weights_vec
  return difference, similarity

def torch_get_results(all_centroids_ang, embeddings2):
  vectors2_ang = embeddings2[:, None, :] - all_centroids_ang
  distances2_ang = torch.norm(vectors2_ang, dim=2)
  results2_ang = torch.argmin(distances2_ang, dim=1)
  del vectors2_ang
  del distances2_ang
  return results2_ang

def torch_get_right_wrong(results2_ang, tensor_list_ang, device):
  array_right = torch.tensor(tensor_list_ang, device=device).squeeze(1)
  #array_right = all_centroids_torch[array_right]
  mask_right = (array_right == results2_ang)
  right = torch.sum(mask_right).cpu().item()
  wrong = len(results2_ang)-right
  print(f"Pravilno določeni: {right}")
  print(f"Napačno določeni: {wrong}")
  print(f"Procent pravilno: {(right/(right+wrong)*100)}%")
  del array_right
  del mask_right

def torch_rot_one(embeddings1, results, results2, all_centroids_torch, embeddings1_cent):
  mask_no_rot = results.unsqueeze(2) != 0
  mask_no_rot = mask_no_rot.repeat(1,1,embeddings1.shape[1])#.to(device)  # (length, 512)
  cent_emb2_rot = all_centroids_torch[results2]#torch.vstack([all_centroids_torch[result2].unsqueeze(0) for result2 in results2])
  vectors_rot_one = mask_no_rot * (embeddings1_cent.unsqueeze(1)-cent_emb2_rot.unsqueeze(0))
  return vectors_rot_one

def torch_rot_both(results1, results2, target_idx, embeddings1):
  mask_target_rot1 = (results1 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
  mask_target_rot2 = (results2 != target_idx).view(-1, 1).repeat(1, embeddings1.shape[1])
  return mask_target_rot1, mask_target_rot2

def torch_average(embeddings1,embeddings2, results1, results2, weights_vec, weights, all_centroids_torch):
  vectors1_avg = torch.vstack([torch.cat([all_centroids_torch[:results1[i]], all_centroids_torch[results1[i] + 1:]]).unsqueeze(0) for i in range(len(results1))])
  embs1_avg = torch.cat((embeddings1.unsqueeze(1), embeddings1.unsqueeze(1) + vectors1_avg), dim=1)
  sum_avg = weights.sum()
  embs1_avg = torch.sum((weights_vec.unsqueeze(0) * embs1_avg), dim=1)/ sum_avg
  del vectors1_avg

  vectors2_avg = torch.vstack([torch.cat([all_centroids_torch[:results2[i]], all_centroids_torch[results2[i] + 1:]]).unsqueeze(0) for i in range(len(results2))])
  embs2_avg = torch.cat((embeddings2.unsqueeze(1), embeddings2.unsqueeze(1) + vectors2_avg), dim=1)
  embs2_avg = torch.sum((weights_vec.unsqueeze(0) * embs2_avg), dim=1)/ sum_avg
  del vectors2_avg
  return embs1_avg, embs2_avg

class EmbeddingDatasetBoth(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {
            'person': row[0],
            'img_dir': row[1],
            'results1_ang': row['results1_ang'],
            'centroid_ang': row['centroid_ang'],
            'results1_lig': row['results1_lig'],
            'centroid_lig': row['centroid_lig'],
            'original_index': idx
        }
        if 'angle' in row:
          data['angle'] = torch.tensor(row['angle'], dtype=torch.float32)
          data['embedding'] = torch.tensor(row['embedding'], dtype=torch.float32)
        else:
          data['embedding'] = torch.tensor(row[2], dtype=torch.float32)
        return data

def check_torch_both(df, weights_ang, weights_lig, centroids_angles, centroids_lights, eigenvectors, angles_are):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  wrong = 0
  right = 0
  difference = []
  similarity =[]
  #df = df.iloc[:100]
  array_all = df['embedding']

  df_copy = df.copy()
  #average
  #result
  all_centroids_ang = torch.tensor(centroids_angles, device = device, dtype=torch.float32)
  all_centroids_lig = torch.tensor(centroids_lights, device = device, dtype=torch.float32)

  weights_ang = torch.tensor(np.array(weights_ang), dtype=torch.float32)
  weights_lig = torch.tensor(np.array(weights_lig), dtype=torch.float32)
  embeddings2 = torch.tensor(np.vstack(array_all), device = device, dtype=torch.float32)
  #embeddings1 = torch.tensor(np.vstack(array_test[:, embedding_num]), device = device, dtype=torch.float32)
  weights_array_ang = weights_ang.view(-1, 1)
  weights_vec_ang = weights_array_ang.repeat(1, embeddings2.shape[1]).to(device)
  weights_array_lig = weights_lig.view(-1, 1)
  weights_vec_lig = weights_array_lig.repeat(1, embeddings2.shape[1]).to(device)

  target_idx_ang = torch.tensor(5, device=device)
  target_idx_lig = torch.tensor(8, device=device)

  # Compute distances for all test group embeddings
  results2_ang = torch_get_results(all_centroids_ang, embeddings2)
  results2_lig = torch_get_results(all_centroids_lig, embeddings2)

  if angles_are:
    tensor_list_ang = np.vstack(df['angle'])
    tensor_list_lig = np.vstack(df['light'])
    torch_get_right_wrong(results2_ang, tensor_list_ang, device)
    torch_get_right_wrong(results2_lig, tensor_list_lig, device)

  all_embeddings1_cent_ang = all_centroids_ang[results2_ang].cpu().numpy()
  all_embeddings1_cent_lig = all_centroids_lig[results2_lig].cpu().numpy()

  #df_test = pd.DataFrame(array_test)
  df_copy['results1_ang'] = results2_ang.cpu().numpy()
  df_copy['results1_lig'] = results2_lig.cpu().numpy()
  #df_test['results'] = list(all_results)
  df_copy['centroid_ang'] = list(all_embeddings1_cent_ang)
  df_copy['centroid_lig'] = list(all_embeddings1_cent_lig)
  del all_embeddings1_cent_ang
  del all_embeddings1_cent_lig

  P3, P_embeddings2_3 = get_P(eigenvectors, 3, embeddings2, device)

  dataset = EmbeddingDatasetBoth(df_copy)
  dataloader = DataLoader(dataset, batch_size=75, num_workers=6, shuffle=False)
  with torch.no_grad():
      for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        person = batch['person']
        #results = batch['results'].to(device)
        original_indices = batch['original_index']
        results1_ang = batch['results1_ang'].to(device)
        embeddings1_cent_ang = batch['centroid_ang'].to(device)
        results1_lig = batch['results1_lig'].to(device)
        embeddings1_cent_lig = batch['centroid_lig'].to(device)
        embeddings1 = batch['embedding'].squeeze(1).to(device)

        results_ang = results1_ang.unsqueeze(1)-results2_ang.unsqueeze(0)
        results_lig = results1_lig.unsqueeze(1)-results2_lig.unsqueeze(0)

        # Rotate embedding one
        #damo vmes med eno želeno centroido in drugo
        vectors_rot_one_ang = torch_rot_one(embeddings1, results_ang, results2_ang, all_centroids_ang, embeddings1_cent_ang)
        vectors_rot_one_lig = torch_rot_one(embeddings1, results_lig, results2_lig, all_centroids_lig, embeddings1_cent_lig)
        emb2_rot_one = embeddings2.unsqueeze(0) + vectors_rot_one_lig/2 + vectors_rot_one_ang/2

        # Rotate embedding both
        mask_target_rot1_ang, mask_target_rot2_ang = torch_rot_both(results1_ang, results2_ang, target_idx_ang, embeddings1)
        mask_target_rot1_lig, mask_target_rot2_lig = torch_rot_both(results1_lig, results2_lig, target_idx_lig, embeddings1)
        emb1_rot_both = embeddings1 + mask_target_rot1_ang * all_centroids_ang[target_idx_ang]/2+ mask_target_rot1_lig * all_centroids_lig[target_idx_lig]/2
        emb2_rot_both = embeddings2 + mask_target_rot2_ang * all_centroids_ang[target_idx_ang]/2+ mask_target_rot2_lig * all_centroids_lig[target_idx_lig]/2
        del mask_target_rot1_ang
        del mask_target_rot2_ang
        del mask_target_rot1_lig
        del mask_target_rot2_lig
        
        # Calculate average weight
        embs1_avg_ang, embs2_avg_ang = torch_average(embeddings1,embeddings2, results1_ang, results2_ang, weights_vec_ang, weights_ang, all_centroids_ang)
        embs1_avg_lig, embs2_avg_lig = torch_average(embeddings1,embeddings2, results1_lig, results2_lig, weights_vec_lig, weights_lig, all_centroids_lig)

        embs1_avg = (embs1_avg_ang+embs1_avg_lig)/2
        embs2_avg = (embs2_avg_ang+embs2_avg_lig)/2

        shape = [len(embeddings1),len(results2_ang), 1]
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

        emb1_P3 = torch.mm(P3, embeddings1.T).T
        sim_3 = compute_cosine_similarities_torch(emb1_P3.unsqueeze(1), P_embeddings2_3.unsqueeze(0)).reshape(shape)

        del emb1_P3
        del embeddings1

        for i in range(len(person)):
          
          mask_diff = torch.tensor((df['person'] != person[i]).to_numpy())
          mask_sim = ~mask_diff
          mask_sim[original_indices[i]] = False
          stack_dif = torch.hstack([normal[i][mask_diff],rot_one[i][mask_diff], rot_both[i][mask_diff], avg_one[i][mask_diff], avg_both[i][mask_diff], sim_3[i][mask_diff]])
          stack_sim = torch.hstack([normal[i][mask_sim],rot_one[i][mask_sim], rot_both[i][mask_sim], avg_one[i][mask_sim], avg_both[i][mask_sim], sim_3[i][mask_sim]])
          difference.append(stack_dif.cpu().numpy())
          similarity.append(stack_sim.cpu().numpy())     
        
        torch.cuda.empty_cache()
        del person
        del results_ang
        del results_lig
        del results1_ang
        del results1_lig
        del embeddings1_cent_ang
        del embeddings1_cent_lig
        del mask_diff
        del mask_sim
        del stack_dif
        del stack_sim
  del embeddings2
  del all_centroids_ang
  del all_centroids_lig
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

def compute_cosine_similarities_torch_all(embeddings1, embeddings2, epsilon=1e-8):
    # Norme embeddingov
    norm1 = torch.norm(embeddings1, dim=2, keepdim=True)
    norm2 = torch.norm(embeddings2, dim=2, keepdim=True)

    # Kosinusna podobnost
    dot_products = torch.sum(embeddings1 * embeddings2, dim=2, keepdim=True)
    cosine_similarities = dot_products / (torch.clamp(norm1 * norm2, min=epsilon))
    
    return cosine_similarities.squeeze()

def compute_cosine_similarities_pairs_3(embeddings1, embeddings2, epsilon=1e-8):
    """
    Izračuna kosinusne podobnosti med vsemi pari embeddingov, kjer ima vsaka instanca 3 embeddinge.

    Parametri:
      embeddings1: torch.Tensor, oblika (100, 1, 3, 512)
      embeddings2: torch.Tensor, oblika (1, batch_size, 3, 512)
      epsilon: majhna vrednost za preprečitev deljenja z 0

    Vrne:
      cosine_sim: torch.Tensor, oblika (100, batch_size, 9)
                  Za vsako kombinacijo med 100 in batch_size primeri dobimo 9 kosinusnih podobnosti,
                  ki predstavljajo vse pare med 3 embeddingi prvega in 3 embeddingi drugega tenzorja.
    """
    # Razširimo dimenzije, da bomo lahko ustvarili vse kombinacije med embeddingi:
    # embeddings1: (100, 1, 3, 512) -> (100, 1, 3, 1, 512)
    A = embeddings1.unsqueeze(3)
    # embeddings2: (1, batch_size, 3, 512) -> (1, batch_size, 1, 3, 512)
    B = embeddings2.unsqueeze(2)
    
    # Z broadcastingom dobimo tenzor oblike (100, batch_size, 3, 3, 512)
    # Izračunamo skalarni produkt vzdolž embedding dimenzije (-1)
    dot_products = torch.sum(A * B, dim=-1)  # Oblika: (100, batch_size, 3, 3)
    
    # Izračunamo norme za embeddinge
    norm_A = torch.norm(embeddings1, dim=-1).unsqueeze(3)  # (100, 1, 3, 1)
    norm_B = torch.norm(embeddings2, dim=-1).unsqueeze(2)  # (1, batch_size, 1, 3)
    
    # Produkt norm, oblika: (100, batch_size, 3, 3)
    norm_mult = norm_A * norm_B
    
    # Izračunamo kosinusne podobnosti
    cosine_sim = dot_products / torch.clamp(norm_mult, min=epsilon)  # Oblika: (100, batch_size, 3, 3)
    del dot_products
    # Preoblikujemo zadnji dve dimenziji (3, 3) v eno (9)
    cosine_sim = cosine_sim.view(cosine_sim.size(0), cosine_sim.size(1), -1)  # Oblika: (100, batch_size, 9)
    
    return cosine_sim

def compute_cosine_similarities(embeddings1, embeddings2, epsilon=1e-8):
    
    # Norme embeddingov
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Kosinusna podobnost
    dot_products = np.sum(embeddings1 * embeddings2, axis=1, keepdims=True)
    cosine_similarities = dot_products / (np.maximum(norm1 * norm2, epsilon))
    
    return cosine_similarities.squeeze()

def save(tpr, fpr, descriptions, out_directory):
    fpr = fpr.cpu().numpy()
    tpr = tpr.cpu().numpy()

    wb = Workbook()
    ws = wb.active
    ws.title = "ROC Data"

    # Dodaj naslove stolpcev
    headers = []
    for i in range(fpr.shape[1]):
        headers.append(f"FPR {descriptions[i]}")
        headers.append(f"TPR {descriptions[i]}")
    ws.append(headers)

    # Zapiši podatke vrstico po vrstico
    for row in zip(*[val for pair in zip(fpr.T, tpr.T) for val in pair]):
        ws.append(row)

    # Shrani datoteko
    wb.save(out_directory)

def process_table(data, table_name):
    # Povprečje po vrsticah
    #mean_values = np.mean(data, axis=0)
    #max_values = np.max(data, axis=0)
    #min_values = np.min(data, axis=0)
    
    # Združimo vse v eno DataFrame
    result_df = pd.DataFrame({
        f"{table_name}": data.ravel(),
        #f"{table_name}_mean": data,
        #f"{table_name}_max": max_values,
        #f"{table_name}_min": min_values
    })
    return result_df
    
def calculate_roc(similarity, difference, out_dir, descriptions, current_date):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  array_sim = np.concatenate(similarity)
  array_diff = np.concatenate(difference)
  print(f"Similarity tested: {array_sim.shape[0]}")
  print(f"Difference tested: {array_diff.shape[0]}")
  print(f"Together: {array_diff.shape[0]+array_sim.shape[0]}")

  tpr1 = []
  tpr01 = []
  tpr001 = []
  tpr0001 = []
  eer = []
  for i in range(array_sim.shape[1]):
    similarity_scores = torch.tensor(np.concatenate((array_sim[:, i], array_diff[:, i]), axis = 0), device = device, dtype=torch.float32)
    #print(similarity_scores.shape)
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True, dim=0)
    del sorted_scores

    true_labels = torch.tensor(np.concatenate((np.ones([1, len(array_sim)]), np.zeros([1, len(array_diff)])), axis = 1), device = device)
    true_labels = true_labels.T
    sorted_labels = true_labels[sorted_indices, 0]
    del sorted_indices
    del true_labels

    # True positives in false positives
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1 - sorted_labels, dim=0)

    # Skupno število pozitivnih in negativnih primerov
    total_positives = tps[-1]
    total_negatives = fps[-1]

    # TPR in FPR
    tpr = tps / total_positives
    fpr = fps / total_negatives

    mask_eer = fpr + tpr -1
    eer_idx = torch.argmin(torch.abs(mask_eer), dim = 0)
    del mask_eer

    fpr_0001_mask = fpr-0.00001
    fpr_0001_idx = torch.argmin(torch.abs(fpr_0001_mask), dim = 0)
    del fpr_0001_mask

    fpr_001_mask = fpr-0.0001
    fpr_001_idx = torch.argmin(torch.abs(fpr_001_mask), dim = 0)
    del fpr_001_mask

    fpr_01_mask = fpr-0.001
    fpr_01_idx = torch.argmin(torch.abs(fpr_01_mask), dim = 0)
    del fpr_01_mask

    fpr_1_mask = fpr-0.01
    fpr_1_idx = torch.argmin(torch.abs(fpr_1_mask), dim = 0)
    del fpr_1_mask

    tpr0001.append(tpr[fpr_0001_idx].cpu().numpy())
    tpr001.append(tpr[fpr_001_idx].cpu().numpy())
    tpr01.append(tpr[fpr_01_idx].cpu().numpy())
    tpr1.append(tpr[fpr_1_idx].cpu().numpy())
    eer.append(fpr[eer_idx].cpu().numpy())

    #filepath = create_directory(os.path.join(out_dir, descriptions[i]))
    #save_graphs(fpr.cpu().numpy(), tpr.cpu().numpy(), out_directory=filepath)

  eer_all_array = np.vstack(eer)
  tpr_0001_all_array = np.vstack(tpr0001)
  tpr_001_all_array = np.vstack(tpr001)
  tpr_01_all_array = np.vstack(tpr01)
  tpr_1_all_array = np.vstack(tpr1)

  head_df = pd.DataFrame({"Description": descriptions})
  eer_df = process_table(eer_all_array, "EER")
  tpr_0001_df = process_table(tpr_0001_all_array, "TPR_0001")
  tpr_001_df = process_table(tpr_001_all_array, "TPR_001")
  tpr_01_df = process_table(tpr_01_all_array, "TPR_01")
  tpr_1_df = process_table(tpr_1_all_array, "TPR_1")

  final_df = pd.concat([head_df, eer_df, tpr_0001_df, tpr_001_df, tpr_01_df, tpr_1_df], axis=1) 
  final_df.to_excel(os.path.join(out_dir,f"{current_date}_data.xlsx"), index=False)
  print(f'Saved all in {out_dir}')

def get_centroids(df, centroid_directory, selection):
  columns = df.shape[1]
  if columns == 5:
    print("Angles are")
    angles_are = True
  elif columns == 3:
    print("No angles")
    angles_are = False
  else:
    raise ValueError("Length of df is not typical")

  cent_basename = os.path.basename(centroid_directory)
  cent_type = cent_basename[16:21]
  if cent_type == 'angle':
    if angles_are:
      cent_angles = df['angle'].unique().tolist()
      cent_angles.sort()
    else:
      cent_angles = get_all_angles()#
  elif cent_type == 'light':
    cent_angles = get_all_lights()#

  all_centroids = []
  for i in range(len(cent_angles)):
      if cent_angles[i] in selection:
        centroid_str =  f"Centroid_{ang_to_str(cent_angles[i])}.npy"
        centroid_dir = find_centroid(centroid_str, centroid_directory)
        vector = np.load(centroid_dir)
        #vector = apply_base(vector, base_vectors, VtV_inv)
        #vector /=np.linalg.norm(vector)
        all_centroids.append(vector)
  return all_centroids, angles_are

def main():
  df = load_data_df('/home/rokp/test/dataset/swinface/cplfw/swinface.npz')
  #df = df.iloc[43000:]
  #prvih35 = [f"{i:03d}" for i in range(1, 36)]
  #df = df[df['person'].isin(prvih35)]
  #df = df.drop(columns=['light', 'angle'])
  go_PCA = False
  examine = 'angle'
  centroid_dir_ang = '/home/rokp/test/dataset/swinface/svetloba/20250312_135249_angle_36'
  centroid_dir_lig = '/home/rokp/test/dataset/swinface/svetloba/20250312_135256_light_36'
  out_dir = '/home/rokp/test/ROC'
  total = 5
  loop = False

  descriptions = ["Normal", "Rot One", "Rot Both", "Avg One", "Avg Both", "Avg Cos"]#['2', '3', '4', '5', '10', '25'] #
  #selection = [light for light in range(0,14)]#do 13
  selection_ang = get_all_angles()
  selection_lig = get_all_lights()
  centroids_angles, angles_are_ang = get_centroids(df, centroid_dir_ang, selection_ang)
  centroids_lights, angles_are_lig = get_centroids(df, centroid_dir_lig, selection_lig)
  angles_are = angles_are_lig and angles_are_ang
  #global_mean = np.load(find_centroid("global.npy", centroid_directory)) 

  #df = df[df['light'].isin(selection)]
  #eigenvectors =np.load(find_centroid("eigenvectors.npy", centroid_dir_ang))
  eigenvectors = 0
  global_mean = np.mean(np.vstack(df['embedding']), axis=0).reshape(1, -1)
  df['embedding'] = df['embedding'].apply(lambda x: x - global_mean)

  current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
  filepath = create_directory(os.path.join(out_dir, current_date))

  if not loop:
    total = 1
  #weights = [10,8,6,4,0,2,0,4,6,8,10]
  weights_ang = np.ones(len(centroids_angles))#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  weights_lig = np.ones(len(centroids_lights))#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  
  if examine == 'angle':
    difference, similarity = check_torch_all(df, weights_ang, centroids_angles, eigenvectors, angles_are, 'angle', go_PCA=go_PCA)
  elif examine == 'light':
    difference, similarity = check_torch_all(df, weights_lig, centroids_lights, eigenvectors, angles_are, 'light', go_PCA=go_PCA)
  else:
    raise ValueError("nepoznan examine")
  #difference, similarity = check_torch_both(df, weights_ang, weights_lig, centroids_angles, centroids_lights, eigenvectors, angles_are)
  calculate_roc(similarity, difference, filepath, descriptions, current_date)

if __name__ == '__main__':
  main()
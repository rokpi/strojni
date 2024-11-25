import numpy as np
from utils_new import load_data_df, divide_dataframe_one, cosine_similarity
import pandas as pd
from datetime import datetime
import os

def optimal_rotation_matrix(p1, p2):
    """
    Izračunaj optimalno rotacijsko matriko, ki preslika p1 v p2 v n-dimenzionalnem prostoru.
    Metoda uporablja SVD, da zagotovi optimalno rotacijsko transformacijo brez strižnih ali razteznih operacij.
    """
    # Najprej normaliziraj oba vektorja
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)
    
    p1_norm = np.squeeze(p1_norm).flatten()
    p2_norm = np.squeeze(p2_norm).flatten()
    # Izračunaj matriko korelacije
    H = np.dot(p1_norm[:, np.newaxis], p2_norm[np.newaxis, :])
    
    # Singularni vrednostni razcep (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Izračun optimalne rotacijske matrike (brez strižnih transformacij)
    R = np.dot(Vt.T, U.T)
    
    # Preveri, ali je rotacijska matrika ustrezna (determinanta mora biti 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R



def generate_random_matrix(dimension):
    matrix = np.random.randn(1, dimension)
    return matrix

def check_rotation_matrix(R):
    should_be_identity = np.dot(R.T, R)
    identity = np.eye(R.shape[0])
    orthogonality_check = np.allclose(should_be_identity, identity)

    determinant = np.linalg.det(R)
    det_check = np.isclose(determinant, 1.0)

    return orthogonality_check, det_check

def main():
  '''# Sintetični podatki
  A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

  # Rotiraj A za znan kot (npr. 45 stopinj okoli osi z)
  theta = np.radians(45)
  R_known = np.array([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta),  np.cos(theta), 0],
      [0, 0, 1]
  ])
  B = np.dot(A, R_known.T)'''

  num_embeddings = 200
  angles = [-30, -15, 0, 15, 30]
  results =[]

  df = load_data_df('/home/rokp/test/models/mtcnn/vgg-vgg/vgg-vggmtcnn_imgs_images_mtcnn_data.npz')
  df = divide_dataframe_one(df, num_embeddings)

  for i in range(len(angles)):
    angles_j = angles[i+1:]
    for j in angles_j:    
      df_take = df[df['angle'].isin([angles[i], j])]

      grouped = df_take.pivot(index = 'person', columns = 'angle', values = 'embedding')

      grouped['angle_tuple'] = list(zip(grouped[angles[i]], grouped[j]))

      angle_tuples = grouped['angle_tuple'].to_list()

      cos_sims =  []
      for ang in angle_tuples:
        cos_sim = cosine_similarity(ang[0], ang[1])
        cos_sims.append(cos_sim)
      
      max_sim = min(cos_sims)
      min_sim = max(cos_sims)
      avg_sim = np.mean(cos_sims)

      results.append({
          'Angles': f"{angles[i]}, {j}",
          'Max_Sim': max_sim,
          'Min_Sim': min_sim,
          'Avg_Sim': avg_sim
      })
  
  df_results = pd.DataFrame(results)
  current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_file = os.path.join('/home/rokp/test/excel', f'{current_date}_cos_sim.xlsx')
  df_results.to_excel(output_file, index=False)  
  print(f'File successfully saved in: {output_file}')





if __name__ == '__main__':

  #main()
  #R_calculated = kabsch_algorithm(A, B)
  #cos_sim = cosine_similarity(B, np.dot(A, R_calculated.T))
  #print(f"Cosine similarity: {cos_sim}")

  #orthogonal, determinant_correct = check_rotation_matrix(R)
  #print(f"Is orthogonal: {orthogonal}, Determinant = 1: {determinant_correct}")
    

  A = generate_random_matrix(2048)
  B = generate_random_matrix(2048)

  print("starting calculation")
  R = optimal_rotation_matrix(A, B)
  print("ended")
  cos_sim = cosine_similarity(B, np.dot(A, R.T))
  print(f"cosine similariy is: {cos_sim}")



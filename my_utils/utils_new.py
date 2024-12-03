import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def find_centroid(centroid_str, directory):
    dirs = []    
    for in_file in os.listdir(directory):
        if centroid_str == in_file:
            dirs.append(os.path.join(directory, in_file))
    if len(dirs)>1:
        raise ValueError(f"Too many rotation matrixes in directory with this name: {centroid_str}")
    elif len(dirs) == 0:
        raise ValueError(f"No files in directory with this name: {centroid_str}")
    else:
        return dirs[0]
    
def ang_to_str(angle):
    string = str(abs(angle)).zfill(2)    
    if angle < 0:
        string = "m" + string
    return string

def get_all_filepaths(root_dir, ext):
  filepaths = []
  for root, _, files in os.walk(root_dir):
    for file in files:
      if file.endswith(ext):
        filepath = os.path.join(root, file)
        rel_filepath = os.path.relpath(filepath, root_dir)
        filepaths.append(rel_filepath)
  filepaths = sorted(filepaths)
  return filepaths

def get_all_angles():
    return [-90 , -85, -80, -30, -15, 0, 15, 30, 80, 85, 90]
def save_rot_npz(df, out_dir, in_directory):
    filename = Path(os.path.basename(in_directory))
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname  = f"rot_{current_date}"
    out_directory = create_directory(os.path.join(out_dir, dirname))
    fileDirectory = os.path.join(out_directory, f"rot_{filename}")

    df['embedding'] = df['embedding'].apply(cartesian_to_rotational)
    
    np.savez(fileDirectory, data=df.to_numpy(), columns=df.columns.values)
    print(f"File saved in {fileDirectory}")
    return fileDirectory

'''
def cartesian_to_rotational(point):
    r = np.linalg.norm(point)
    x = point[:point.shape[0]-1]
    y = point[1:]
    thetas = np.arctan2(x, y)
    result = np.append(r, thetas)
    return result

def rotational_to_cartesian(point):
    r = point[0]
    thetas = point[1:]

    x = r * np.sin(thetas)
    y = r * np.cos(thetas[point.shape[0]-2])
    result = np.append(x, y)

    return result
'''
def cartesian_to_rotational(point):
    rot_dims = []
    r = np.linalg.norm(point)

    for x in point.flatten()[1:][::-1]:
        sin = 1
        theta = np.arccos(x/(r * sin))
        rot_dims.append(theta)
        sin_theta = np.sin(theta)
        sin *= sin_theta

    rot_dims.append(r)

    coordinates = np.array(rot_dims)
    coordinates = coordinates[::-1]
    return coordinates

def rotational_to_cartesian(point):
    car_dims = []
    r = point[0]
    mul = r
    for rot in point[2:][::-1]:
        x = mul * np.cos(rot)
        mul *= np.sin(rot)
        car_dims.append(x)
    
    x = mul * np.cos(rot)
    car_dims.append(x)
    x = mul * np.sin(rot)
    car_dims.append(x)
    coordinates = np.array(car_dims)
    coordinates = coordinates[::-1]
    return coordinates



def cosine_similarity(embedding1, embedding2, epsilon=1e-8):
    embedding1 = np.squeeze(embedding1)
    embedding2 = np.squeeze(embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    cos_sim = np.dot(embedding1, embedding2) / (max(norm1 * norm2, epsilon))
    
    return cos_sim

def divide_dataframe_one(df, test_num):
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    grouped = shuffled_df.groupby('person')
    unique_persons = list(grouped.groups.keys())
    if test_num <= len(unique_persons):
        test_persons  = unique_persons[:test_num]

        df_test = df[df['person'].isin(test_persons)]
    else:
        raise ValueError(f"Test and train people together should be less then {len(unique_persons)}!")

    return df_test

def load_data_df(in_directory):
    #load dataframe df from .npz file
    loaded = np.load(in_directory, allow_pickle= True)
    df = pd.DataFrame(loaded['data'], columns=loaded['columns'])
    #convert 'embedding' back to numpy array
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
    return df   

def getAngle(filename):
    basename = os.path.basename(filename)
    viewpoint = int(basename[10:13])
    if viewpoint == 120:
        angle = -90
    elif viewpoint == 90:
        angle = -85
    elif viewpoint == 80:
        angle = -80
    elif viewpoint == 130:
        angle = -30                        
    elif viewpoint == 140:
        angle = -15
    elif viewpoint == 51:
        angle = 0
    elif viewpoint == 50:
        angle = 15
    elif viewpoint == 41:
        angle = 30
    elif viewpoint == 190:
        angle = 80
    elif viewpoint == 200:
        angle = 85
    elif viewpoint == 10:
        angle = 90
    else:
        raise ValueError("Invalid input of angle!")
    return angle

def model_to_str(model_directory):
    filename_split = os.path.basename(model_directory).split('.')
    model_name = filename_split[0]
    if filename_split[1] != 'hdf5':
      model_name = model_name + filename_split[1]
    else:
      model_name = model_name + 'raw'
       
    return model_name

def create_directory(out_directory):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

        print(f"Directory created: {out_directory}")
    else:
        print(f"Directory already exists: {out_directory}")
    return out_directory

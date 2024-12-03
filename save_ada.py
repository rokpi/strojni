import torch
import os
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import glob
from datetime import datetime
import sys
sys.path.append('/home/rokp/test/models/AdaFace')
from face_alignment import align
import net
adaface_models = {
    'ir_101':"/home/rokp/test/models/AdaFace/pretrained/adaface_ir101_webface12m.ckpt",
}

    
def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

# Razred za dataset
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img_paths = row['img_dir']
            imgs = Image.open(img_paths).convert('RGB')  # Odpri sliko in pretvori v RGB

        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_paths}: {e}")
        return {
            'person': row['person'],   # Podatek o osebi
            'img_dir': row['img_dir'], # Pot do slike
            'angle': row['angle'],      # Kot obraza
            'tensor': row['tensor']

        }


# Beleženje manjkajočih obrazov
def log_missing_faces(filepath, datetime):
    desktop_path = '/home/rokp/test/strojni/missing'
    os.makedirs(desktop_path, exist_ok=True)
    log_file_path = os.path.join(desktop_path, f"{datetime}_adaface_missing.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{filepath}\n")

def preprocess_img(df, output_dir, datetime):
    dataset = ImageDataset(df)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=6, shuffle=False)

    if torch.cuda.is_available():
        device = 'cuda'
        print("Procesiranje bo potekalo na GPU.")
    else:
        device = 'cpu'
        print("Procesirannje bo poteklo na CPU.")
    
    device = 'cpu'

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="image"):
            img_dir = batch['img_dir'][0]
            aligned_rgb_img = align.get_aligned_face(img_dir)
            if aligned_rgb_img:
                bgr_tensor_input = to_input(aligned_rgb_img).to(device)
                results.append({
                    'person': batch['person'][0],   # Ostane string
                    'img_dir': img_dir,            # Ostane string
                    'angle': batch['angle'][0],    # Pretvori tensor v int
                    'tensor': bgr_tensor_input.cpu().numpy()
                })
            else:
                log_missing_faces(img_dir, datetime)
                continue


    results_df = pd.DataFrame(results)

    # Shrani v .npz format
    file_path = output_dir
    results_df.to_pickle(file_path)
    '''np.savez(
        file_path,
        data=results_df,        # Shranite podatke kot NumPy array
        columns=results_df.columns.to_list()  # Shranite imena stolpcev
    )'''
    print(f"Embeddingi in podatki shranjeni v {file_path}")
def divide_pickle():
    df = pd.read_pickle('/home/rokp/test/test/20241202_095535')
    chunk_size = 10000
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        chunk.to_pickle(f'/home/rokp/test/chunk_2/chunk_{start // chunk_size}.pkl')
    print("Datoteka je razdeljena na manjše dele.")


def process_and_save_embeddings(model, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    results = []
    chunk_files = sorted(glob.glob('/home/rokp/test/chunk/ada/chunk_*.pkl'))

    for chunk_file in chunk_files:
        chunk_df = pd.read_pickle(chunk_file)  # Naloži en kos
        dataset = ImageDataset(chunk_df)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
                images = batch['tensor'].squeeze(1)
                images = images.to(device)
                features, _ = model(images)  # Izračunaj embeddinge
                embeddings = features.cpu().numpy()  # Pretvori v NumPy

                # Združi podatke
                for i in range(len(embeddings)):
                    results.append({
                        'person': batch['person'][i],               # Ostane string
                        'img_dir': batch['img_dir'][i],              # Ostane string
                        'angle': batch['angle'][i].item(),           # Pretvori tensor v int
                        'embedding': embeddings[i].tolist()         # Pretvori v seznam
                    })

    # Pretvori rezultate v DataFrame
    results_df = pd.DataFrame(results)

    # Shrani v .npz format
    file_path = output_dir
    np.savez(
        file_path,
        data=results_df.to_numpy(),  # Shranite podatke kot NumPy array
        columns=results_df.columns.to_list()  # Shranite imena stolpcev
    )

    print(f"Embeddingi in podatki shranjeni v {file_path}")

if __name__ == '__main__':
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S") 
    out_dir = os.path.join('/home/rokp/test/test', 'test_ada.npz')
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    process_and_save_embeddings(model, out_dir)
    '''model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_image_path = '/home/rokp/test/images_raw_jpg'
    features = []
    for fname in tqdm(sorted(os.listdir(test_image_path)), total = len(os.listdir(test_image_path))):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)

        if aligned_rgb_img:
            bgr_tensor_input = to_input(aligned_rgb_img)
            feature, _ = model(bgr_tensor_input)
            features.append(feature)
        else:
            log_missing_faces(path, current_date)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)'''
    


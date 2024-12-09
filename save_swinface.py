import cv2
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

# Razred za dataset
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'person': row['person'],   # Podatek o osebi
            'img_dir': row['img_dir'], # Pot do slike
            #'angle': row['angle'],      # Kot obraza
            'img': self.transform(row['img_dir'])
        }
    
    #@torch.no_grad()
    def transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img
    
#@torch.no_grad()
def process_and_save_embeddings(df, output_dir):
    sys.path.append('/home/rokp/test/models/SwinFace/swinface_project')
    from model import build_model
    cfg = SwinFaceCfg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg)
    dict_checkpoint = torch.load('/home/rokp/test/models/SwinFace/checkpoints/checkpoint_step_79999_gpu_0.pt')
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])
    model = model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    results = []
    dataset = ImageDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=6, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            images = batch['img'].to(device)
            # Združi podatke
            for i in range(len(images)):
                results.append({
                    'person': batch['person'][i],               # Ostane string
                    'img_dir': batch['img_dir'][i],              # Ostane string
                    #'angle': batch['angle'][i].item(),           # Pretvori tensor v int
                    'embedding': model(images[i])['Recognition'][0].cpu().numpy().tolist()         # Pretvori v seznam
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

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512



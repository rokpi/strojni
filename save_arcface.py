import argparse

import cv2
import numpy as np
import torch

import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append('/home/rokp/test/models/arcface_torch')
from backbones import get_model

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
    
    def transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img
    

def process_and_save_embeddings(df, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = get_model('r100', fp16=False)
    net.load_state_dict(torch.load('/home/rokp/test/glint360k_cosface_r100_fp16_0.1/backbone.pth'))
    net = net.to(device)
    net.eval()
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
                    'embedding': net(images[i]).cpu().numpy().tolist()         # Pretvori v seznam
                })

    # Pretvori rezultate v DataFrame
    results_df = pd.DataFrame(results)

    # Shrani v .npz format
    file_path = os.path.join(output_dir, 'arcface.npz')
    np.savez(
        file_path,
        data=results_df.to_numpy(),  # Shranite podatke kot NumPy array
        columns=results_df.columns.to_list()  # Shranite imena stolpcev
    )

    print(f"Embeddingi in podatki shranjeni v {file_path}")
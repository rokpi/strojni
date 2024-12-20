from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

# Definicija Dataset razreda
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizacija
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'person': row['person'], # Dodatni podatek
            'tensor': row['tensor'],  # Tensor slike
            'angle': row['angle'],   # Dodatni podatek
            'img_dir': row['img_dir'] # Pot do slike
        }

# Inicializacija modela in podatkov
def process_and_save_embeddings(model, output_dir):
    df = pd.read_pickle('/home/rokp/test/test/20241202_084132')
    print(df)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dataset = ImageDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            images = torch.stack([torch.tensor(img) for img in batch['img_tensor']]).to(device)
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



import os
import sys
import cv2
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

import torch
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

config_path = 'src/config.json'
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config(config_path)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class AffecNet_Dataset(Dataset):
    def __init__(self, image_paths, origin_au_vecs, final_au_vecs, trans=None):
        self.image_paths = image_paths
        self.origin_au_vecs = origin_au_vecs
        self.final_au_vecs = final_au_vecs
        self.trans = trans
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        origin_au_vec = self.origin_au_vecs[idx]
        final_au_vec = self.final_au_vecs[idx]
        
        if self.trans:
            image = trans(image)
        
        origin_au_vec = torch.round(torch.tensor(origin_au_vec), decimals=2)
        final_au_vec = torch.round(torch.tensor(final_au_vec), decimals=2)
        
        return image, origin_au_vec, final_au_vec
    
def get_dataset(labels_path):
    with open(labels_path, 'r') as file:
        data = file.readlines()
        
    image_paths = []
    origin_au_vecs = []
    final_au_vecs = []
    for dt in data:
        image_infor = dt.split('\t')
        image_path = image_infor[0]
        
        origin_au_vec = image_infor[1][1:-1].split(', ')
        origin_au_vec_num = np.array([float(num) if num[0] != '-' else -float(num[1:]) for num in origin_au_vec])
        
        final_au_vec = image_infor[2][1:-2].split(', ')
        final_au_vec_num = np.array([float(num) if num[0] != '-' else -float(num[1:]) for num in final_au_vec])
        
        image_paths.append(image_path)
        origin_au_vecs.append(origin_au_vec_num)
        final_au_vecs.append(final_au_vec_num)
    return image_paths, origin_au_vecs, final_au_vecs

def get_dataloader(labels_path, mode='train'):
    image_paths, origin_au_vecs, final_au_vecs = get_dataset(labels_path)
    
    train_image_paths, val_image_paths, train_origin_au_vecs, val_origin_au_vecs, \
        train_final_au_vecs, val_final_au_vecs = train_test_split(image_paths, origin_au_vecs, final_au_vecs, test_size=0.2, shuffle=True)
    
    if mode == 'train':
        train_dataset = AffecNet_Dataset(train_image_paths, train_origin_au_vecs, train_final_au_vecs, trans=trans)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
        return train_loader
    
    elif mode == 'val':
        val_dataset = AffecNet_Dataset(val_image_paths, val_origin_au_vecs, val_final_au_vecs, trans=trans)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=True)
        return val_loader
    
    else:
        print("Invalid mode, mode must be in ['train', 'val']")
        return None

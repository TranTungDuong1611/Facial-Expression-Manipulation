import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm
from src.losses import *
from src.model.generator import *
from src.model.discriminator import *
from src.model.gan_imation import *
from src.data.dataloader import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config

config_path = 'src/config.json'
config = load_config(config_path)

def training_loop(D, G, model, train_loader, epochs, lr, beta1, beta2, step_size_D, step_size_G, device=device):
    model.to(device)
    
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size_D, gamma=0.1)
    
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size_G, gamma=0.1)
    
    loss_D_lst = []
    loss_G_lst = []
    
    for epoch in range(epochs):
        
        for real_sample, origin_au, final_au in tqdm(train_loader):
            real_sample = real_sample.to(device)
            origin_au = origin_au.to(device)
            final_au = final_au.to(device)
            
            optimizer_D.zero_grad()
            loss_D = model._forward_D(real_sample, origin_au)
            loss_D.backward()
            optimizer_D.step()
            loss_D_lst.append(loss_D)
            
            if (epoch + 1) % 5 == 0:
                optimizer_G.zero_grad()
                loss_G = model._forward_G(real_sample, origin_au, final_au)
                loss_G.backward()
                optimizer_G.step()
                loss_G_lst.append(loss_G)
        
        scheduler_D.step()
        scheduler_G.step()
            
        gen_image_origin_final, gen_image_final_origin, _, _ = model(real_sample, origin_au, final_au)
        _, axes = plt.subplots(1, 3, figsize=(10, 10))
        axes[0].imshow(real_sample)
        axes[0].axis('off')
        axes[0].set_title('Real Image')
        
        axes[1].imshow(gen_image_origin_final)
        axes[1].axis('off')
        axes[1].set_title('Real to Fake Image')
        
        axes[2].imshow(gen_image_final_origin)
        axes[2].axis('off')
        axes[2].set_title('Fake to Real Image')
            
        print(f"Epoch: {epoch}/{epochs}\t loss_D: {loss_D:.4f}\tloss_G: {loss_G:.4f}")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="Number of training epochs")
    parser.add_argument('--lr', type=np.float32, default=config['train']['lr'], help="Learning rate")
    parser.add_argument('--beta1', type=np.float32, default=config['train']['beta1'])
    parser.add_argument('--beta2', type=np.float32, default=config['train']['beta2'])
    parser.add_argument('--step_size_D', type=int, default=config['train']['step_size_D'])
    parser.add_argument('--step_size_G', type=int, default=config['train']['step_size_G'])
    
    args = parser.parse_args()
    
    model = GANImation()
    
    train_loader = get_dataloader(labels_path='dataset\AffectNet\train\labels.txt', mode='train')
    
    print(f"Training....")
    training_loop(
        D=model.discriminator,
        G=model.generator,
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        step_size_D=args.step_size_D,
        step_size_G=args.step_size_G
    )

if __name__ == '__main__':
    main()
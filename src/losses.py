import os
import sys
import json
sys.path.append(os.getcwd())

import torch
import torchvision

from src.model.generator import *
from src.model.discriminator import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = 'src/config.json'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config

config = load_config(config_path)

def gradient_penalty(D, real_sample, fake_sample, device=device, penalty_coef=config['network']['lambda_gp']):
    # initialize random interpolation distribution
    alpha = torch.rand(real_sample.size(0), 1, 1, 1).expand_as(real_sample)
    interpolates = (alpha * real_sample + (1 - alpha) * fake_sample).detach().requires_grad_(True)
    
    # calculate DI(I~)
    D_I, _ = D(interpolates)
    
    # calculate gradient DI(I~)
    D_I_grad = torch.autograd.grad(
        outputs=D_I,
        inputs=interpolates,
        grad_outputs=torch.ones_like(D_I),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    
    D_I_grad = D_I_grad.view(D_I_grad.size(0), -1)
    norm_D_I_grad = torch.norm(D_I_grad, p=2, dim=1)
    grad_penalty_loss = torch.mean((norm_D_I_grad - 1) ** 2)
    return grad_penalty_loss * penalty_coef

def image_adversarial_loss(D, real_sample, fake_sample):
    loss_fake, _ = D(fake_sample)
    loss_fake = torch.mean(loss_fake, dim=(1, 2))
    loss_real, _ = D(real_sample)
    loss_real = torch.mean(loss_real, dim=(1, 2))
    
    final_loss = loss_fake - loss_real + gradient_penalty(D, real_sample, fake_sample)
    return final_loss

def attention_loss(fake_sample, penalty_coef=config['network']['lambda_TV']):
    variation_regularization_i = torch.sum((fake_sample[:, :, :-1, :] - fake_sample[:, :, 1:, :]) ** 2)
    variation_regularization_j = torch.sum((fake_sample[:, :, :, 1:] - fake_sample[:, :, :, :-1]) ** 2)
    
    total_variation_regularization = variation_regularization_i + variation_regularization_j
    total_variation_regularization = torch.mean(total_variation_regularization)
    
    fake_sample = fake_sample.view(fake_sample.size(0), -1)
    norm_attention_mask = torch.norm(fake_sample, p=2, dim=1) / (fake_sample.size(1))
    
    total_attetion_loss = penalty_coef * total_variation_regularization + norm_attention_mask
    return total_attetion_loss


def condition_expression_loss(D, real_sample, fake_sample, final_au, origin_au, mode='G'):
    _, pred_au_fake = D(fake_sample)
    aux_fake_final_loss = torch.norm((pred_au_fake - final_au), p=2, dim=1)
    print(pred_au_fake.size())
    pred_au_fake = pred_au_fake.view(pred_au_fake.size(0), -1)
    aux_fake_final_loss = torch.square(aux_fake_final_loss) / pred_au_fake.size(1)
    print(aux_fake_final_loss.size())
    
    _, pred_au_origin = D(real_sample)
    aux_origin_final_loss = torch.norm((pred_au_origin - origin_au), p=2, dim=1)
    pred_au_origin = pred_au_origin.view(pred_au_origin.size(0), -1)
    aux_origin_final_loss = torch.square(aux_origin_final_loss) / pred_au_origin.size(1)
    
    if mode == 'G':
        return aux_fake_final_loss
    elif mode == 'D':
        return aux_origin_final_loss
    
    
def identity_loss(G, real_sample, fake_sample, origin_au):
    identity_sample, _ = G(fake_sample, origin_au)
    identity_sample = identity_sample.view(identity_sample.size(0), -1)
    real_sample = real_sample.view(real_sample.size(0), -1)
    
    identity_loss = torch.abs((identity_sample - real_sample))
    identity_loss = torch.mean(identity_loss, dim=1)
    
    return identity_loss


def full_loss_G(D, G, real_sample, fake_sample, origin_au, final_au, lambda_A=config['network']['lambda_A'], lambda_y=config['network']['lambda_y'], lambda_idt=config['network']['lambda_idt']):
    final_loss = image_adversarial_loss(D, real_sample, fake_sample) + \
                lambda_y * condition_expression_loss(D, real_sample, fake_sample, final_au, origin_au) + \
                lambda_A * (attention_loss(fake_sample) + attention_loss(real_sample)) + \
                lambda_idt * identity_loss(G, real_sample, fake_sample, origin_au)
    final_loss = final_loss.detach().requires_grad_(True)
    
    # calculates total loss in a batch
    final_loss = torch.sum(final_loss)
    return final_loss

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from src.model.generator import *
from src.model.discriminator import *
from src.losses import *

class GANImation(nn.Module):
    def __init__(self, au_dims=5):
        super(GANImation, self).__init__()
        
        self.generator = Generator(au_dims=au_dims)
        self.discriminator = Discriminator(au_dims=au_dims)
        
    def forward(self, image, origin_au_vec, final_au_vec, mode="gan"):
        if mode == "gan":
            # generated image of generator
            image_reg_origin_final, attention_reg_origin_final = self.generator(image, final_au_vec)
            gen_image_origin_final = image_reg_origin_final * attention_reg_origin_final + (1 - attention_reg_origin_final) * image
            # the input pass through the discriminator
            out_real, out_aux = self.discriminator(gen_image_origin_final)
            # the generated image is concatinated with the final au vector and pass through the second
            # generator for get the original image
            image_reg_final_origin, attention_reg_final_origin = self.generator(gen_image_origin_final, origin_au_vec)
            gen_image_final_origin = image_reg_final_origin * attention_reg_final_origin + (1 - attention_reg_final_origin) * gen_image_origin_final
            
            return gen_image_origin_final, gen_image_final_origin, out_real, out_aux
        elif mode == "generator_real_fake":
            fake_sample, attention_map = self.generator(image, final_au_vec)
            return fake_sample, attention_map
        elif mode == "generator_fake_real":
            fake_sample, attention_map = self.generator(image, origin_au_vec)
            return fake_sample, attention_map
        elif mode == "discriminator":
            out_real, out_aux = self.discriminator(image)
            
    def _forward_G(self, real_sample, origin_au_vec, final_au_vec):
        fake_sample, attention_map_fake  = self.generator(real_sample, final_au_vec)
        gen_image = fake_sample * attention_map_fake + (1 - attention_map_fake) * real_sample
        
        self.loss_G = full_loss_G(self.discriminator, self.generator, real_sample, gen_image, origin_au_vec, final_au_vec)
        return self.loss_G
    
    def _forward_D(self, real_sample, origin_au, final_au):
        fake_sample, attention_map_fake = self.generator(real_sample, final_au)
        gen_image = fake_sample * attention_map_fake + (1 - attention_map_fake) * real_sample
        
        out_real, out_au = self.discriminator(real_sample)
        
        # D(real_I)
        self.loss_D_real = torch.mean(out_real) * config['network']['lambda_y']
        self.loss_D_cond = condition_expression_loss(self.discriminator, real_sample, fake_sample, final_au, origin_au, mode='D') * config['network']['lambda_y']
        
        # D(fake_I)
        out_fake, _ = self.discriminator(gen_image)
        self.loss_D_fake = torch.mean(out_fake) * config['network']['lambda_y']
        
        # gradient penalty D
        self.gradient_penalty = gradient_penalty(self.discriminator, real_sample, gen_image)
        
        return self.loss_D_fake - self.loss_D_real + self.loss_D_cond + self.gradient_penalty
        
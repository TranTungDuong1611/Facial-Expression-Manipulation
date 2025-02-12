import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from src.model.generator import *
from src.model.discriminator import *

class GANImation(nn.Module):
    def __init__(self, in_channels, au_dims=5):
        super(GANImation, self).__init__()
        
        self.gen_origin_final = Generator(au_dims=au_dims)
        self.gen_final_origin = Generator(au_dims=au_dims)
        self.discriminator = Discriminator(au_dims=au_dims)
        
    def forward(self, image, origin_au_vec, final_au_vec):
        # generated image of generator
        image_reg_origin_final, atttention_reg_origin_final = self.gen_origin_final(image, origin_au_vec)
        gen_image_origin_final = image_reg_origin_final * atttention_reg_origin_final + (1 - image_reg_origin_final) * image
        # the input pass through the discriminator
        out_real, out_aux = self.discriminator(gen_image_origin_final)
        # the generated image is concatinated with the final au vector and pass through the second
        # generator for get the original image
        image_reg_final_origin, attention_reg_final_origin = self.gen_final_origin(gen_image_origin_final, final_au_vec)
        gen_image_final_origin = image_reg_final_origin * attention_reg_final_origin + (1 - image_reg_final_origin) * gen_image_origin_final
        
        return gen_image_origin_final, gen_image_final_origin, out_real, out_aux
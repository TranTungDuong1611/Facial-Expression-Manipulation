import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __int__(self, au_dims, img_size=128, conv_dims=64, num_repeat=6):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(
            in_channels=3,
            out_channels=conv_dims,
            kernel_size=4,
            stride=2,
            padding=1
        ))
        layers.append(nn.LeakyReLU(0.01))
        
        cur_dims = conv_dims
        for _ in range(num_repeat):
            layers.append(nn.Conv2d(
                in_channels=cur_dims,
                out_channels=cur_dims*2,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            layers.append(nn.LeakyReLU(0.01))
            cur_dims = cur_dims * 2
            
        k_size = img_size // (np.power(2, num_repeat))

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            in_channels=cur_dims,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=cur_dims,
            out_channels=au_dims,
            kernel_size=k_size,
            stride=k_size,
            bias=False
        )
        
    def forward(self, x):
        x = self.main(x)
        out_real = self.conv1(x)
        out_aux = self.conv2(x)
        return out_real.squeeze(), out_aux.squeeze()
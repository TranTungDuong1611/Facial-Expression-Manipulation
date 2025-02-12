import torch
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    def __init__(self, au_dims, conv_dim=64, bottle_neck_len=6):
        super(Generator, self).__init__()
        self.au_dims = au_dims
        self.conv_dim = conv_dim
        self.bottle_neck_len = bottle_neck_len
        
        # convolution down
        self.conv1_down = nn.Sequential(
            nn.Conv2d(
                in_channels=(au_dims+3),
                out_channels=self.conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim, affine=True),
            nn.ReLU()
        )
        
        self.conv2_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim*2, affine=True),
            nn.ReLU()
        )
        
        self.conv3_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conv_dim*2,
                out_channels=self.conv_dim*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim*4, affine=True),
            nn.ReLU()
        )
        
        self.conv4_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conv_dim*4,
                out_channels=self.conv_dim*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim*8, affine=True),
            nn.ReLU()
        )
        
        
        # bottle neck
        current_dim = self.conv_dim*8
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=current_dim,
                out_channels=current_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(current_dim, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=current_dim,
                out_channels=current_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(current_dim, affine=True)
        )
        
        # convolution up
        self.conv4_up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.conv_dim*8,
                out_channels=self.conv_dim*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim*4, affine=True),
            nn.ReLU()
        ),
        
        self.conv3_up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.conv_dim*4,
                out_channels=self.conv_dim*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim*2, affine=True),
            nn.ReLU()
        ),
        
        self.conv2_up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.conv_dim*2,
                out_channels=self.conv_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(self.conv_dim, affine=True),
            nn.ReLU()
        ),
        
        self.conv1_up_image_reg = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.conv_dim,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.Tanh()
        ),
        
        self.conv1_up_attention_reg = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.conv_dim,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.Sigmoid()
        )
        
    def forward(self, x, c):
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.conv1_down(x),
        x = self.conv2_down(x),
        x = self.conv3_down(x),
        x = self.conv4_down(x),
        
        for _ in range(self.bottle_neck_len):
            x = self.bottle_neck(x)
            
        x = self.conv4_up(x),
        x = self.conv3_up(x),
        x = self.conv2_up(x),
        
        image_reg = self.conv1_up_image_reg(x)
        attention_reg = self.conv1_up_attention_reg(x)
        
        return image_reg, attention_reg
    
    
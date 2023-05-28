import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class input_embedding_conv(nn.Module):
    def __init__(self,image_size_y,image_size_x,patch_size,dim,channel):
        super().__init__()
        self.image_size_y=image_size_y
        self.image_size_x=image_size_x
        self.patch_size=patch_size
        self.dim=dim
        self.channel=channel

        self.proj=nn.Conv2d(channel,dim,kernel_size=patch_size,stride=patch_size)

        num_patches= (image_size_y//patch_size)*(image_size_x//patch_size)

        self.class_token=nn.Parameter(torch.zeros(1,4,dim))
        self.pos_emb=nn.Parameter(torch.rand(1,num_patches+4,dim))

    def forward(self,image):
        batch_size,channels,width,height= image.size()

        x = self.proj(image)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches_x*n_pathces_y)
        out=x.transpose(-1,-2) #(n_samples,n_pathces,emb_dim
        class_token=self.class_token.expand(batch_size,-1,-1)
        out=torch.cat([class_token,out],dim=1)

        batch_size,n,h=out.shape
        out=out+self.pos_emb[:,:(n+4)]  #n_samples,n_pathces+4,emb_dim
        return out


class input_embedding_linear(nn.Module):
    def __init__(self,img_size_y,img_size_x,patch_size,dim,channel):
        super().__init__()
        self.img_size_y= img_size_y
        self.img_size_x = img_size_x
        self.dim=dim
        self.patch_size=patch_size
        self.channel=channel

        self.linear_emb=nn.Linear(channel*(patch_size**2),dim)
        num_patches=(img_size_x//patch_size)*(img_size_y//patch_size)

        self.class_token=nn.Parameter(torch.zeros(1,4,dim))
        self.pos_emb=nn.Parameter(torch.rand(1,num_patches+4,dim))
    def forward(self,image):
        batch_size,channel,width,height=image.shape
        arrange=Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size,p2=self.patch_size)
        out=arrange(image)
        out=self.linear_emb(out)
        class_token = self.class_token.expand(batch_size, -1, -1)
        out = torch.cat([class_token, out], dim=1)
        batch_size, n, h = out.shape
        out = out + self.pos_emb[:, :(n + 4)]
        return out


from Full_Quantum_Multihead_Attentions import *
from Hybrid_Quantum_Multihead_Attentions import *
from input_embedding import *
from FeedForwardNetworks import *


class one_block(nn.Module):
    def __init__(self,dim,mlp_dim,num_head,n_layer,n_qubits,n_seq_1):
        super().__init__()
        self.dim=dim
        self.mlp_dim=mlp_dim
        self.num_head=num_head

        self.normalization1=nn.LayerNorm(dim)
        self.multiheadattention=Quantum_Multihead_Attention1(dim,num_head,n_layer,n_qubits,n_seq_1)
        self.mlp=feedforward(dim,mlp_dim,dim)
        self.normalization2 = nn.LayerNorm(dim)

    def forward(self,input):
        out=input+self.multiheadattention(self.normalization1(input))
        out1=out+self.mlp(self.normalization2(out))
        return out1

class transformer_block(nn.Module):
    def __init__(self,dim,mlp_dim,out_dim,num_heads,depth,n_layer,n_qubits,n_seq_1):
        super().__init__()
        self.dim=dim
        self.mlp_dim=mlp_dim
        self.out_dim=out_dim
        self.num_heads=num_heads
        self.depth=depth
        self.blocks=nn.ModuleList([one_block(dim,mlp_dim,num_heads,n_layer,n_qubits,n_seq_1) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)

    def forward(self,image):
        out=image
        for block in self.blocks:
            out=block(out)
        out=self.norm(out)
        return out


class vit(nn.Module):
    def __init__(self,image_size_y,image_size_x,patch_size,dim,channel,mlp_dim,num_heads,out_dim,depth,n_layer,n_qubit,n_seq_1):
        super().__init__()
        self.dim=dim
        self.mlp_dim=mlp_dim
        self.out_dim=out_dim
        self.num_heads=num_heads
        self.depth=depth
        self.image_size_x=image_size_x
        self.image_size_y=image_size_y
        self.patch_size=patch_size
        self.dim=dim
        self.channel=channel

        self.input=input_embedding_conv(image_size_y,image_size_x,patch_size,dim,channel)
        self.transformer=transformer_block(dim,mlp_dim,out_dim,num_heads,depth,n_layer,n_qubit,n_seq_1)
        self.last_mlp=nn.Linear(8*dim,out_dim)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,images):
        input_emb=self.input(images)
        out_transformer=self.transformer(input_emb)
        out_transformer=out_transformer[:,:8]
        out_transformer=out_transformer.reshape(-1,8*self.dim)
        out=self.last_mlp(out_transformer)
        return out

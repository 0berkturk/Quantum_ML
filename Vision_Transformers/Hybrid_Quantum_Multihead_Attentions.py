import torch.nn as nn
from einops import rearrange
import pennylane as qml
import torch

class Classic_multihead_attention(nn.Module):
    def __init__(self,dim,num_heads,n_qubit):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scaling=head_dim**-0.5
        self.qkv=nn.Linear(dim,3*dim)
        self.last_linear=nn.Linear(dim,dim)

    def forward(self,input):
        batch_size,n_patch_1,embed_dim=input.size()
        qkv1=self.qkv(input)
        qkv1=qkv1.chunk(3,dim=-1)
        q,k,v=map(lambda t: rearrange(t,'b n (h d) -> b h n d', h=self.num_heads), qkv1)
        k=k.transpose(-2,-1)
        qk=(q@k)*self.scaling
        qk=qk.softmax(dim=-1)
        qkv=qk@v
        qkv = qkv.transpose(1, 2).reshape(batch_size, n_patch_1, embed_dim)
        out = self.last_linear(qkv)
        return out

class Hybrid_Quantum_Multihead_Attention1(nn.Module):
    def __init__(self,embed_dim,num_heads,n_layer,n_qubits,n_seq_1):
        # note n_qubit is embed_dim
        super().__init__()
        self.dim=embed_dim
        self.num_heads=num_heads
        head_dim=embed_dim//num_heads
        self.scaling=head_dim**-0.5
        self.last_linear=nn.Linear(embed_dim,embed_dim)

        def Q_circuit(inputs, weights): ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits+2),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits+2))
            return qml.probs(wires=range(n_qubits+2))

        self.weight_shapes = {"weights": (n_layer, n_qubits+2,3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits+2))
        self.Q_layer = qml.QNode(Q_circuit,self.dev,interface='torch')  #diff_method="parameter-shift"

        self.QKVW_m = qml.qnn.TorchLayer(self.Q_layer, self.weight_shapes)

    def forward(self,x):
        b,n_patch_1,embed_dim = x.shape
        x=x.reshape(b,-1)
        x4=torch.cat((x,x,x,x),1)

        QKVW = self.QKVW_m(x4).reshape(b,4,n_patch_1,embed_dim )
        Q=QKVW[:,0]
        K=QKVW[:,1]
        V = QKVW[:, 2]
        #W = QKVW[:, 3] ## this is optinal to use

        Q = rearrange(Q,'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        K = K.transpose(-2, -1)
        qk = (Q @ K) * self.scaling
        qk = qk.softmax(dim=-1)
        qkv = qk @ V
        qkv = qkv.transpose(1, 2).reshape(b, n_patch_1, embed_dim)
        out = self.last_linear(qkv)
        return out


class Hybrid_Quantum_Multihead_Attention2(nn.Module):  ## QKVW tek bir circuit ile hesaplandı. Bölünmedi
    def __init__(self, embed_dim, num_heads, n_layer, n_qubits, n_seq_1):
        # note n_qubit is embed_dim
        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5
        self.last_linear = nn.Linear(4*embed_dim, embed_dim)

        # n_qubits= embed_dim ### if angle embed,math.log2(embed_dim)

        def Q_circuit(inputs, weights):  ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits + 2), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits + 2))
            return qml.probs(wires=range(n_qubits + 2))

        self.weight_shapes = {"weights": (n_layer, n_qubits + 2, 3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits + 2))
        self.Q_circuit = qml.QNode(Q_circuit, self.dev, interface='torch')  # diff_method="parameter-shift"

        self.QKVW_m = qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes)

    def forward(self, x):
        b, n_patch_1, embed_dim = x.shape
        x = x.reshape(b, -1)
        x4 = torch.cat((x, x, x, x), 1)

        QKVW = self.QKVW_m(x4).reshape(b, n_patch_1, 4*embed_dim)
        out = self.last_linear(QKVW)
        return out

class Hybrid_Quantum_Multihead_Attention3(nn.Module):
    def __init__(self, embed_dim, num_heads, n_layer, n_qubits, n_patch):
        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5
        self.last_linear = nn.Linear(4*n_patch, n_patch)

        def Q_circuit(inputs, weights):  ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits + 2), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits + 2))
            return qml.probs(wires=range(n_qubits + 2))

        self.weight_shapes = {"weights": (n_layer, n_qubits + 2, 3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits + 2))

        self.Q_circuit = qml.QNode(Q_circuit, self.dev, interface='torch')  # diff_method="parameter-shift"
        self.QKVW_m = qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes)

    def forward(self, x):
        b, n_patch_1, embed_dim = x.shape
        x = x.reshape(b, -1)
        x4 = torch.cat((x, x, x, x), 1)

        QKVW = self.QKVW_m(x4).reshape(b,embed_dim,4*n_patch_1)
        out = self.last_linear(QKVW).reshape(b,n_patch_1,embed_dim)
        return out


class Hybrid_Quantum_Multihead_Attention4(nn.Module):
    def __init__(self, embed_dim, num_heads, n_layer, n_qubits, n_seq_1):
        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5
        self.last_linear = nn.Linear(embed_dim, embed_dim)

        def QK_circuit(inputs, weights):  ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits + 1), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits + 1))
            return qml.probs(wires=range(n_qubits + 1))

        def V_circuit(inputs, weights):  ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self.weight_shapes = {"weights": (n_layer, n_qubits + 1, 3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits + 1))
        self.QK_node = qml.QNode(QK_circuit, self.dev, interface='torch')  # diff_method="parameter-shift"

        self.weight_shapes2 = {"weights": (n_layer, n_qubits, 3)}
        self.dev2 = qml.device("default.qubit", wires=int(n_qubits))
        self.V_node = qml.QNode(V_circuit, self.dev2, interface='torch')

        self.QK = qml.qnn.TorchLayer(self.QK_node, self.weight_shapes)
        self.V = qml.qnn.TorchLayer(self.V_node, self.weight_shapes2)

    def forward(self, x):
        b, n_patch_1, embed_dim = x.shape
        x = x.reshape(b, -1)
        xx = torch.cat((x, x), 1)

        QK = self.QK(xx).reshape(b, 2, n_patch_1, embed_dim)
        Q, K = QK[:, 0], QK[:, 1]

        V = self.V(x).reshape(b, n_patch_1, -1)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        K = K.transpose(-2, -1)
        qk = (Q @ K) * self.scaling
        qk = qk.softmax(dim=-1)
        qkv = qk @ V
        qkv = qkv.transpose(1, 2).reshape(b, n_patch_1, embed_dim)
        out = self.last_linear(qkv)
        return out

class Hybrid_Quantum_Multihead_Attention5(nn.Module):  ## QK tek bir circuit ile hesaplandı. Bölünmedi
    def __init__(self,embed_dim,num_heads,n_layer,n_qubits,n_seq_1):
        super().__init__()
        self.dim=embed_dim
        self.num_heads=num_heads
        head_dim=embed_dim//num_heads
        self.scaling=head_dim**-0.5
        self.last_linear=nn.Linear(embed_dim,embed_dim)

        def QK_circuit(inputs, weights): ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits+1),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits+1))
            return qml.probs(wires=range(n_qubits+1))

        def V_circuit(inputs, weights): ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self.weight_shapes = {"weights": (n_layer, n_qubits+1,3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits+1))
        self.QK_Node = qml.QNode(QK_circuit,self.dev,interface='torch')  #diff_method="parameter-shift"

        self.weight_shapes2 = {"weights": (n_layer, n_qubits, 3)}
        self.dev2 = qml.device("default.qubit", wires=int(n_qubits))
        self.V_Node = qml.QNode(V_circuit, self.dev2, interface='torch')

        self.QK = qml.qnn.TorchLayer(self.QK_Node, self.weight_shapes)
        self.V = qml.qnn.TorchLayer(self.V_Node, self.weight_shapes2)

    def forward(self,x):
        b,n_patch_1,embed_dim = x.shape
        x=x.reshape(b,-1)
        xx=torch.cat((x,x),1)

        qk = self.QK(xx).reshape(b,-1,1)
        qk =nn.AvgPool2d((2,1))(qk)
        qk = qk.reshape(b,n_patch_1,embed_dim)

        V = self.V(x).reshape(b, n_patch_1, -1)

        qk = rearrange(qk, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        qk = qk * self.scaling
        qk = qk.softmax(dim=-1)
        qkv = qk @ V
        qkv = qkv.transpose(1, 2).reshape(b, n_patch_1, embed_dim)
        out = self.last_linear(qkv)
        return out


class Hybrid_Quantum_Multihead_Attention6(nn.Module):
    def __init__(self, embed_dim, num_heads, n_layer, n_qubits, n_seq_1):

        super().__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5
        self.last_linear = nn.Linear(embed_dim, embed_dim)

        def Q_circuit(inputs, weights):  ### ANOTHER WAY, combine inputs.
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))


        self.weight_shapes = {"weights": (n_layer, n_qubits, 3)}
        self.dev = qml.device("default.qubit", wires=int(n_qubits))
        self.Q_node = qml.QNode(Q_circuit, self.dev, interface='torch')


        self.Q = qml.qnn.TorchLayer(self.Q_node, self.weight_shapes)
        self.V = qml.qnn.TorchLayer(self.Q_node, self.weight_shapes)
        self.K = qml.qnn.TorchLayer(self.Q_node, self.weight_shapes)

    def forward(self, x):
        b, n_patch_1, embed_dim = x.shape
        x = x.reshape(b, -1)
        Q = self.Q(x).reshape(b, n_patch_1, -1)
        K = self.Q(x).reshape(b, n_patch_1, -1)
        V = self.V(x).reshape(b, n_patch_1, -1)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        K = K.transpose(-2, -1)
        qk = (Q @ K) * self.scaling
        qk = qk.softmax(dim=-1)
        qkv = qk @ V
        qkv = qkv.transpose(1, 2).reshape(b, n_patch_1, embed_dim)
        out = self.last_linear(qkv)
        return out
import pennylane as qml
import torch
import torchvision
import torch.nn as nn
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import qiskit
import time
from data_load import *
"""""Input(tensor) + Classical NN + Quantum Circuit + Classical NN + Softmax + Labels """


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
print(device)


class Quanvolutional_NN(nn.Module):
    def __init__(self,n_layer):
        super().__init__()
        self.weight_shapes = {"weights": (n_layer, 4,3)}
        self.dev = qml.device("default.qubit", wires=4)

        def circuit(inputs,weights):
            qml.AngleEmbedding(inputs,wires=range(4))
            qml.StronglyEntanglingLayers(weights,wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)] #qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


        self.qlayer = qml.QNode(circuit, self.dev,diff_method="parameter-shift" ,interface="torch")
        self.qlayer1=qml.qnn.TorchLayer(self.qlayer,self.weight_shapes)
        self.clayer1 = nn.Linear(4*14*14,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,image):
        b,a,q,_ = image.shape
        out = torch.zeros((b,4, 14, 14)).to('cuda')

        for j in range(0,28,2):
            for k in range(0,28,2):
                x=torch.cat((image[:,:,j, k],
                        image[:,:,j, k + 1],
                        image[:,:,j + 1, k],
                        image[:,:,j + 1, k + 1]),1)

                q_results = self.qlayer1(x)
                # Assign expectation values to different channels of the output pixel (j/2, k/2)

                for c in range(4):
                    out[:,c,j // 2, k // 2] = q_results[:,c]

        out=out.reshape(b,-1)
        out = self.clayer1(out)
        return out



class Classical_NN(nn.Module):
    def __init__(self,n_qubits,n_layer):
        super().__init__()

        self.clayer1=nn.Linear(28*28,n_qubits)

        self.clayer2=nn.Linear(n_qubits,10)

        self.clayer3 = nn.Linear(n_qubits,2)
        self.softmax=nn.Softmax(dim=1)
        self.gelu=nn.GELU()

    def forward(self,x):
        x=x.reshape(-1,28*28).float()
        x=self.clayer1(x)
        x=self.gelu(x)
        x=self.clayer2(x)
        x = self.gelu(x)
        x=self.clayer3(x)
        x=self.softmax(x)
        return x


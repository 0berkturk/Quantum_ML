import pennylane as qml
import torch
import torchvision
import torch.nn as nn
from pennylane import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import qiskit


class feedforward(nn.Module):
    def __init__(self,in_dim,mlp_dim,out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim,out_dim)
        )
    def forward(self,input):
        b,c,w,h=input.shape
        input=input.reshape(b,-1)
        return self.net(input)

class Quantum_feedforward1(nn.Module): ## This is working very well with n_qubit=2
    def __init__(self,size,n_qubits,n_layer):
        super().__init__()
        self.size=size
        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def embedding_Strong_Entg_1(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] #qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.qlayer = qml.QNode(embedding_Strong_Entg_1, self.dev, interface="torch")
        self.qlayer1 = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)

        self.clayer1=nn.Linear(size**2,2**n_qubits)
        #self.clayer2 = nn.Linear(n_qubits, 2)
        self.softmax=nn.Softmax(dim=1)
        self.gelu = nn.GELU()

    def forward(self,x):
        b,c,h,w=x.shape
        x=x.reshape(b,h*w*c).float()
        print(x.shape)
        x=self.clayer1(x)
        x=self.qlayer1(x)
        #x = self.clayer2(x)
        x=self.softmax(x)
        return x

class Quantum_feedforward2(nn.Module): ## This is working very well with n_qubit=2
    def __init__(self,size,n_qubits,n_layer):
        super().__init__()
        self.size=size
        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def embedding_Strong_Entg_1(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] #qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.qlayer = qml.QNode(embedding_Strong_Entg_1, self.dev, interface="torch")
        self.qlayer1 = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)

        self.clayer1=nn.Linear(size**2,2**n_qubits)
        #self.clayer2 = nn.Linear(n_qubits, 2)
        self.softmax=nn.Softmax(dim=1)
        self.gelu = nn.GELU()

    def forward(self,x):
        b,c,h,w=x.shape
        x=x.reshape(b,h*w*c).float()
        print(x.shape)
        x=self.clayer1(x)
        x=self.qlayer1(x)
        #x = self.clayer2(x)
        x=self.softmax(x)
        return x

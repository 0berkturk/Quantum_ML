import torch.nn as nn
import torch.nn as nn
import pennylane as qml
import torch
import math

class feedforward(nn.Module):
    def __init__(self,in_dim,mlp_dim,out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim,out_dim)
        )
    def forward(self,input):
        return self.net(input)


class Qfeedforward(nn.Module):  ## satisfy indim and n_seq to run this.
    def __init__(self,in_dim,mlp_dim,out_dim,n_layer,n_seq):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim

        n_qubits = int(math.log2(in_dim))
        n_qubits2 = int(math.log2(n_seq))

        def Q_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        def Q_circuit2(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits2),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits2))
            return qml.probs(wires=range(n_qubits2))

        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.Q_circuit = qml.QNode(Q_circuit,self.dev,interface='torch')
        self.circuit1 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes) for _ in range(n_seq)])

        self.weight_shapes2 = {"weights": (n_layer, n_qubits2,3)}
        self.dev2 = qml.device("default.qubit", wires=n_qubits)
        self.Q_circuit2 = qml.QNode(Q_circuit2,self.dev,interface='torch')
        self.circuit2 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit2, self.weight_shapes2) for _ in range(in_dim)])

    def forward(self,x):
        b,n_seq,in_dim = x.shape
        if (self.in_dim==self.out_dim):
            M = torch.Tensor([]).to('cuda')
            for i in range(n_seq):
                M = torch.cat((M, self.circuit1[i](x[:,i,:])), dim=1)
            x=M.reshape(b,n_seq,in_dim)
            M = torch.Tensor([]).to('cuda')
            for j in range(in_dim):
                M = torch.cat((M, self.circuit2[j](x[:,:,j])), dim=1)
            x=M.reshape(b,n_seq,in_dim)
        return x

class feedforward2(nn.Module):   ## this code can work with any sizes i.e. embed_dim and patch size etc.
    def __init__(self,in_dim,mlp_dim,out_dim,n_layer,n_seq):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim

        n_qubits = int(math.log2(in_dim))

        def Q_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.Q_circuit = qml.QNode(Q_circuit,self.dev,interface='torch')
        self.circuit1 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes) for _ in range(n_seq)])
        self.circuit2 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes) for _ in range(n_seq)])

    def forward(self,x):
        b,n_seq,in_dim = x.shape
        if (self.in_dim==self.out_dim):
            M = torch.Tensor([]).to('cuda')
            for i in range(n_seq):
                M = torch.cat((M, self.circuit1[i](x[:,i,:])), dim=1)
            x=M.reshape(b,n_seq,in_dim)

            M = torch.Tensor([]).to('cuda')
            for j in range(n_seq):
                M = torch.cat((M, self.circuit2[j](x[:,j,:])), dim=1)
            x=M.reshape(b,n_seq,in_dim)
        return x


class feedforward3(nn.Module):  ## this cannot work with any sizes.
    def __init__(self,in_dim,mlp_dim,out_dim,n_layer,n_seq):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim

        n_qubits = int(math.log2(in_dim))
        total_pixel = in_dim*n_seq  ## satisfy this
        n_qubits2 = int(math.log2(total_pixel))  ### satisfy this
        total_circuit_number= int(total_pixel/n_qubits2)   ## satisfy this
        self.total_circuit_number=total_circuit_number

        def Q_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        def Q_circuit2(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits2),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits2))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits2)]

        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.Q_circuit = qml.QNode(Q_circuit,self.dev,interface='torch')
        self.circuit1 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes) for _ in range(n_seq)])


        self.weight_shapes2 = {"weights": (n_layer, n_qubits2,3)}
        self.dev2 = qml.device("default.qubit", wires=n_qubits2)
        self.Q_circuit2 = qml.QNode(Q_circuit2,self.dev2,interface='torch')
        self.circuit2 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit2, self.weight_shapes2) for _ in range(total_circuit_number)])

    def forward(self,x):
        b,n_seq,in_dim = x.shape
        if (self.in_dim==self.out_dim):
            M = torch.Tensor([]).to('cuda')
            for i in range(n_seq):
                M = torch.cat((M, self.circuit1[i](x[:,i,:])), dim=1)
            x=M
            M = torch.Tensor([]).to('cuda')
            for j in range(self.total_circuit_number):
                M = torch.cat((M, self.circuit2[j](x)), dim=1)
            x=M.reshape(b,n_seq,in_dim)

        return x

class feedforward4(nn.Module):  ## this code can work with any sizes i.e. embed_dim and patch size etc.
    def __init__(self,in_dim,mlp_dim,out_dim,n_layer,n_seq):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim

        n_qubits = int(math.log2(in_dim))

        def Q_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.Q_circuit = qml.QNode(Q_circuit,self.dev,interface='torch')
        self.circuit1 = nn.ModuleList([qml.qnn.TorchLayer(self.Q_circuit, self.weight_shapes) for _ in range(n_seq)])

    def forward(self,x):
        b,n_seq,in_dim = x.shape
        if (self.in_dim==self.out_dim):
            M = torch.Tensor([]).to('cuda')
            for i in range(n_seq):
                M = torch.cat((M, self.circuit1[i](x[:,i,:])), dim=1)
            x=M.reshape(b,n_seq,in_dim)
        return x


class feedforward5(nn.Module):  ## this code can work with any sizes i.e. embed_dim and patch size etc.
    def __init__(self,in_dim,mlp_dim,out_dim,n_layer,n_seq):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim

        n_qubits = int(math.log2(in_dim*n_seq))

        def Q_circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits)) #[qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (n_layer, n_qubits,3)}
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.Circuit = qml.QNode(Q_circuit,self.dev,interface='torch')  #diff_method="parameter-shift"
        self.Circuit_Matrix = qml.qnn.TorchLayer(self.Circuit, self.weight_shapes)

    def forward(self,x):
        b,n_patch_1,embed_dim = x.shape
        x = x.reshape(b, -1)
        output = self.Circuit_Matrix(x).reshape(b, n_patch_1, -1)
        return output

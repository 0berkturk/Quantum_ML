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
import math
from data_load import *
from function import *
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
### take each row one by one and apply different q circuits. and combine all of them

batch_size=16
n=10000000000

size=8
train_loader, test_loader =load_mnsit_10(batch_size,size,n)

n_qubit = 5

#model_c = feedforward(size**2,768,10)
model = Quantum_feedforward1(size,10,1)
model.to(device)
opt = torch.optim.Adam(model.parameters(),lr=1e-1)
loss = torch.nn.CrossEntropyLoss()
epochs=10
print("started")
time1=time.time()

for epoch in range(epochs):
    k=0
    running_loss = 0
    print("new epoch")
    model.train()
    for xs, ys in train_loader:
        ys=ys.type(torch.LongTensor).to('cuda')
        print(ys)
        xs=xs.to('cuda')
        k+=1
        opt.zero_grad()
        probs = model(xs.float())
        loss_evaluated = loss(probs, ys)
        loss_evaluated.backward()
        opt.step()
        running_loss += loss_evaluated


        max_probs, preds = torch.max(probs, dim=1)

        print("preds",preds)
        print("probs",probs)
        accuracy = torch.sum(preds == ys).item() / len(ys)
        print("acc",accuracy)
        print(" ")

    print(running_loss)

    model.eval()
    acc=0
    k=0
    for xs, ys in test_loader:
        ys = ys.type(torch.LongTensor).to('cuda')
        xs = xs.to('cuda')
        k+=1
        probs = model(xs.float())
        max_probs, preds = torch.max(probs, dim=1)
        accuracy = torch.sum(preds == ys).item() / len(ys)
        acc+=accuracy

    accuracy=acc/k
    print(f"Accuracy: {accuracy * 100}%")

    avg_loss = running_loss / k
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

time2=time.time()
print(time2-time1)
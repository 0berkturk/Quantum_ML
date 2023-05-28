import math
from data_load import *
from vit import *
import os.path
import time

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
print(device)

batch_size = 32
train_loader, test_loader = load_cifar_10(batch_size)

epochs = 1
image_size_y = 32
image_size_x = 32

patch_size = 16
embed_dim = 128
n_patch = (image_size_y//patch_size)*(image_size_x//patch_size)+4
n_qubit = int(math.log2(embed_dim*n_patch))  # embed_dim*n_patch must be the power of 2^n.
print(n_patch,n_qubit)

channel=3
mlp_dim=4*embed_dim
num_heads=4
out_dim=10
depth=1
n_layer=2

model = vit(image_size_y,image_size_x,patch_size,embed_dim,channel,mlp_dim,num_heads,out_dim,depth,n_layer,n_qubit,n_patch)

name_trained_model =" fhfgf" #'Q_VIT4_epoch-1.pt'
if (os.path.isfile(name_trained_model)):
    checkpoint = torch.load(name_trained_model) if torch.cuda.is_available() else torch.load(name_trained_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict']) if os.path.isfile(name_trained_model) else print(" ")
    print("trained model is uploaded")

model.to(device)
opt = torch.optim.Adam(model.parameters(),lr=1e-4)
loss = torch.nn.CrossEntropyLoss()

print("started")
time1=time.time()

batch_loss_list=[]
loss_list=[]
batch_acc_list=[]
acc_list=[]
test_acc_list=[]
test_loss_list=[]
info_list=[]
for epoch in range(epochs):
    k=0
    running_loss = 0
    running_acc = 0
    print("new epoch")
    model.train()
    len1=len(train_loader)
    for xs, ys in train_loader:
        ys=ys.type(torch.LongTensor).to('cuda')
        xs=xs.to('cuda')
        k+=1
        opt.zero_grad()
        outputs=model(xs.float())

        loss_evaluated = loss(outputs, ys)
        loss_evaluated.backward()
        opt.step()
        loss_evaluated=loss_evaluated.cpu().detach().numpy()
        running_loss += loss_evaluated

        probs = outputs
        max_probs, preds = torch.max(probs, dim=1)
        accuracy = torch.sum(preds == ys).item() / len(ys)
        running_acc+=accuracy

        batch_loss_list = np.append(batch_loss_list,loss_evaluated)
        loss_list = np.append(loss_list,(running_loss/k))
        batch_acc_list = np.append(batch_acc_list, accuracy)
        acc_list= np.append(acc_list, (running_acc/k))

        print(ys)
        print(preds)
        #print(probs)
        print(loss_evaluated)
        print(running_loss/k)
        print(accuracy)
        print(running_acc/k)
        print(k/len1)

        print(" ")

    model.eval()
    k=0
    running_loss = 0
    running_acc = 0
    for xs, ys in test_loader:
        ys = ys.type(torch.LongTensor).to('cuda')
        xs = xs.to('cuda')
        k += 1
        opt.zero_grad()
        outputs = model(xs.float())

        loss_evaluated = loss(outputs, ys)
        loss_evaluated = loss_evaluated.cpu().detach().numpy()
        running_loss += loss_evaluated

        probs = outputs
        max_probs, preds = torch.max(probs, dim=1)
        accuracy = torch.sum(preds == ys).item() / len(ys)
        running_acc += accuracy

        test_loss_list = np.append(test_loss_list, running_loss / k)
        test_acc_list = np.append(test_acc_list, running_acc / k)
    print(running_loss / k)
    print(running_acc / k)

time2=time.time()
print(time2-time1)
import numpy as np
np.save("batch_loss_list.npy", batch_loss_list)
np.save("loss_list.npy", loss_list)
np.save("acc_list.npy", acc_list)
np.save("batch_acc_list.npy", batch_acc_list)
np.save("test_loss_list.npy", test_loss_list)
np.save("test_acc_list.npy", test_acc_list)

try:
    state_dict = model.module.state_dict()  # To unwrap DataParallel model.
except AttributeError:
    state_dict = model.state_dict()
state = {
    'epoch': epoch,
    'state_dict': state_dict,
    'optimizer': opt.state_dict()
}
torch.save(state, "Q_VIT4" +
           "_epoch-{}.pt".format(epoch + 1))
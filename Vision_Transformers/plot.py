import numpy as np
import torch
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

numberofparticle=1

print("Training Results")
loss_training = np.load("loss_list.npy")
print("lost_list.npy file ", loss_training,len(loss_training),"\n")

batch_loss_training = np.load("batch_loss_list.npy")
print("batch_loss_list file ", batch_loss_training,len(batch_loss_training),"\n")

acc_list = np.load("acc_list.npy")
print("acc_list file ", acc_list,len(acc_list),"\n")

batch_acc_list = np.load("batch_acc_list.npy")
print("batch_acc_list file ", batch_acc_list,len(batch_acc_list),"\n")

#####################################
print(" ")
print("Test Results")
test_loss_list = np.load("test_loss_list.npy")
print("test_loss_list.npy file ", test_loss_list,len(test_loss_list),"\n")

test_acc_list = np.load("test_acc_list.npy")
print("test_acc_list file ", test_acc_list,len(test_acc_list),"\n")

## detailed



def plot_loss(list_epochs, loss_training,batch_loss_training):
    plt.plot(list_epochs, loss_training, color='b',label="Total loss")
    plt.scatter(list_epochs,batch_loss_training,color='r',label="Loss of a Batch Data (32 image)",s=0.1)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    # plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Number of Steps')
    plt.savefig("loss_plot.png",dpi=300)
    plt.show()


def plot_acc(list_epochs, acc_list,batch_acc_list):
    plt.plot(list_epochs, acc_list, color='b',label="Total Accuracy")
    plt.scatter(list_epochs , batch_acc_list, color='r',label="Accuracy of a Batch Data (32 image)",s=0.1)

    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. Number of Steps')
    plt.savefig("acc_plot.png",dpi=300)
    plt.show()


def plot_loss_test(list_epochs, loss_training):
    plt.plot(list_epochs, loss_training, color='b',label="Total loss")

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    # plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Number of Steps')
    plt.savefig("tloss_plot.png",dpi=300)
    plt.show()


def plot_acc_test(list_epochs, acc_list):
    plt.plot(list_epochs, acc_list, color='b',label="Total Accuracy")

    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. Number of Steps')
    plt.savefig("tacc_plot.png",dpi=300)
    plt.show()


epoch_list=[]
for i in range(len(loss_training)):
    epoch_list.append(i+1)



#plot_loss_training(epoch_list,loss_training)
plot_loss(epoch_list,loss_training,batch_loss_training)
plot_acc(epoch_list,acc_list,batch_acc_list)

epoch_list=[]
for i in range(len(test_loss_list)):
    epoch_list.append(i+1)
plot_loss_test(epoch_list,test_loss_list)
plot_acc_test(epoch_list,test_acc_list)
a = int(len(test_loss_list)/3)
print(test_acc_list[a-1])
print(test_acc_list[2*a-1])
print(test_acc_list[3*a-1])
print(test_acc_list[3*a])
print(test_acc_list[3*a+1])

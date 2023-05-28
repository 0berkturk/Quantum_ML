import torch
from pennylane import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf

def load_cifar_10(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train/255.0, x_test/255.0
    #print(x_train.shape) #(50000, 32, 32, 3)
    #print(y_train.shape)
    #plt.imshow(np.transpose(x_train[0],(0, 1, 2)))
    #plt.show()
    x_train=np.transpose(x_train, (0,3, 1, 2))
    x_test = np.transpose(x_test, (0,3, 1, 2))

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train.flatten())
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test.flatten())


    train_loader = TensorDataset(x_train, y_train)
    test_loader=TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_mnsit_10(batch_size,size,n):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    x_train = tf.image.resize(x_train, (size, size)).numpy()
    x_test = tf.image.resize(x_test, (size, size)).numpy()

    x_train = torch.from_numpy(x_train.reshape(-1,1, size, size))
    y_train = torch.from_numpy(y_train)
    n=int(n/4)
    x_test = torch.from_numpy(x_test.reshape(-1,1, size, size))
    y_test = torch.from_numpy(y_test)


    train_loader = TensorDataset(x_train, y_train)
    test_loader=TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def data_filter(x,y):
    x1= [img for i,img in enumerate(x) if y[i]==1]
    y1= [0 for i in range(len(x1))]

    x2 = [img for i, img in enumerate(x) if y[i] == 8]
    y2 = [1 for i in range(len(x2))]

    return np.concatenate((x1,x2),0), np.concatenate((y1,y2),0)

def load_mnsit_2(batch_size,size,n):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    x_train = tf.image.resize(x_train, (size, size)).numpy()
    x_test = tf.image.resize(x_test, (size, size)).numpy()


    x_train, y_train = data_filter(x_train, y_train)


    x_train = torch.from_numpy(x_train.reshape(-1,1, size, size))
    y_train = torch.from_numpy(y_train)


    x_test, y_test = data_filter(x_test, y_test)
    x_test = torch.from_numpy(x_test.reshape(-1,1, size, size))
    y_test = torch.from_numpy(y_test)

    train_loader = TensorDataset(x_train, y_train)
    test_loader = TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader

def load_fashion_10(batch_size,size,n):
    (x_train, y_train), (x_test, y_test) =tf.keras.datasets.fashion_mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    x_train = torch.from_numpy(x_train.reshape(-1,1, size, size))
    y_train = torch.from_numpy(y_train)
    n=int(n/4)
    x_test = torch.from_numpy(x_test.reshape(-1,1, size, size))
    y_test = torch.from_numpy(y_test)


    train_loader = TensorDataset(x_train, y_train)
    test_loader=TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def data_filter(x,y):
    x1= [img for i,img in enumerate(x) if y[i]==4]
    y1= [0 for i in range(len(x1))]

    x2 = [img for i, img in enumerate(x) if y[i] ==5]
    y2 = [1 for i in range(len(x2))]

    return np.concatenate((x1,x2),0), np.concatenate((y1,y2),0)

def load_fashion_2(batch_size,size,n):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    x_train = tf.image.resize(x_train, (size, size)).numpy()
    x_test = tf.image.resize(x_test, (size, size)).numpy()


    x_train, y_train = data_filter(x_train, y_train)


    x_train = torch.from_numpy(x_train.reshape(-1,1, size, size))
    y_train = torch.from_numpy(y_train)


    x_test, y_test = data_filter(x_test, y_test)
    x_test = torch.from_numpy(x_test.reshape(-1,1, size, size))
    y_test = torch.from_numpy(y_test)

    train_loader = TensorDataset(x_train, y_train)
    test_loader = TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader
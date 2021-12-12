import numpy as np
import torch
import time
def sigmoid(x):
    A = 1 / (1 + 1 / np.exp(x))
    return A

def sigmoid_derivative(x):
    s = 1 / (1 + 1 / np.exp(x))
    sigmoid_derivative = s * (1 - s)
    return sigmoid_derivative

def img2vector(image):
    '''cnvert the input 3D img picture (H x W x C) to a vector of shape (H*W*C, 1)'''
    vec = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
    return vec

def normalizeRows(x):
    X_norm = x.np.linalg.norm(x,axis=1,keepdims=True)  # compute the norm by each row of the input matrix
    x = x / X_norm
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

def L1(y, y_hat):
    Loss = np.sum(np.abs(y, y_hat))
    return Loss


def L2(y_hat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    Loss = np.sum(np.power(y_hat - y), 2)
    return Loss


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_happy_dataset():
    """
    Loads the Happy House dataset

    Returns:
    X_train_orig -- training set features
    Y_train_orig -- training set labels
    X_test_orig -- test set features
    Y_test_orig -- test set labels
    classes -- list of classes
    """
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_signs_dataset():
    """
    Loads the SIGNS dataset

    Returns:
    X_train_orig -- training set features
    Y_train_orig -- training set labels
    X_test_orig -- test set features
    Y_test_orig -- test set labels
    classes -- list of classes
    """
    return load_dataset()


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches *
                                  mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches *
                                  mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Convert to tensors if they're numpy arrays
    if isinstance(W1, np.ndarray):
        W1 = torch.from_numpy(W1).float()
    if isinstance(b1, np.ndarray):
        b1 = torch.from_numpy(b1).float()
    if isinstance(W2, np.ndarray):
        W2 = torch.from_numpy(W2).float()
    if isinstance(b2, np.ndarray):
        b2 = torch.from_numpy(b2).float()
    if isinstance(W3, np.ndarray):
        W3 = torch.from_numpy(W3).float()
    if isinstance(b3, np.ndarray):
        b3 = torch.from_numpy(b3).float()
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    # PyTorch equivalents
    Z1 = torch.matmul(W1, X) + b1
    A1 = F.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = F.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = torch.matmul(W3, A2) + b3

    return Z3


def predict(X, parameters):
    """
    Makes predictions using trained parameters

    Arguments:
    X -- input data
    parameters -- trained model parameters

    Returns:
    prediction -- predicted class
    """
    # Forward propagation
    z3 = forward_propagation_for_predict(X, parameters)

    # Get prediction
    prediction = torch.argmax(z3).item()

    return prediction

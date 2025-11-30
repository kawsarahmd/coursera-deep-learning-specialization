import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def predict(X, parameters):
    """
    Predict using the trained parameters.

    Arguments:
    X -- input data of shape (12288, number of examples)
    parameters -- python dictionary containing trained parameters

    Returns:
    prediction -- predictions for the given data
    """
    # Convert to PyTorch tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()

    # Convert parameters to tensors
    W1 = torch.from_numpy(parameters["W1"]).float() if isinstance(parameters["W1"], np.ndarray) else parameters["W1"]
    b1 = torch.from_numpy(parameters["b1"]).float() if isinstance(parameters["b1"], np.ndarray) else parameters["b1"]
    W2 = torch.from_numpy(parameters["W2"]).float() if isinstance(parameters["W2"], np.ndarray) else parameters["W2"]
    b2 = torch.from_numpy(parameters["b2"]).float() if isinstance(parameters["b2"], np.ndarray) else parameters["b2"]
    W3 = torch.from_numpy(parameters["W3"]).float() if isinstance(parameters["W3"], np.ndarray) else parameters["W3"]
    b3 = torch.from_numpy(parameters["b3"]).float() if isinstance(parameters["b3"], np.ndarray) else parameters["b3"]

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    # Forward propagation
    with torch.no_grad():
        z3 = forward_propagation(X, params)
        prediction = torch.argmax(z3, dim=0)

    return prediction.numpy()
    

def create_placeholders(n_x, n_y):
    """
    Note: Placeholders are not needed in PyTorch.
    This function is kept for backward compatibility but does nothing.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    None, None

    Tips:
    - PyTorch uses dynamic computation graphs, so placeholders are not required.
    """
    # Not needed in PyTorch
    return None, None


def initialize_parameters():
    """
    Initializes parameters to build a neural network with PyTorch. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    torch.manual_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = torch.empty(25, 12288)
    nn.init.xavier_uniform_(W1)
    b1 = torch.zeros(25, 1)

    W2 = torch.empty(12, 25)
    nn.init.xavier_uniform_(W2)
    b2 = torch.zeros(12, 1)

    W3 = torch.empty(6, 12)
    nn.init.xavier_uniform_(W3)
    b3 = torch.zeros(6, 1)
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def compute_cost(z3, Y):
    """
    Computes the cost

    Arguments:
    z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector, same shape as z3

    Returns:
    cost - Tensor of the cost function
    """

    # Convert to PyTorch tensors if needed
    if not isinstance(z3, torch.Tensor):
        z3 = torch.from_numpy(z3).float()
    if not isinstance(Y, torch.Tensor):
        Y = torch.from_numpy(Y).float()

    # Transpose to fit PyTorch requirement (batch_size, num_classes)
    logits = z3.t()
    labels = Y.t()

    ### START CODE HERE ### (1 line of code)
    cost = F.cross_entropy(logits, labels.argmax(dim=1))
    ### END CODE HERE ###

    return cost







def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

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

    # Convert to PyTorch tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()

    # Forward propagation
    Z1 = torch.matmul(W1, X) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = torch.relu(Z2)
    Z3 = torch.matmul(W3, A2) + b3

    return Z3


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer PyTorch neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    torch.manual_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Make parameters require gradients
    for key in parameters:
        parameters[key].requires_grad = True

    # Backpropagation: Define the PyTorch optimizer. Use Adam.
    ### START CODE HERE ### (1 line)
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    ### END CODE HERE ###

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Convert to PyTorch tensors
            minibatch_X = torch.from_numpy(minibatch_X).float()
            minibatch_Y = torch.from_numpy(minibatch_Y).float()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward propagation: Build the forward propagation
            ### START CODE HERE ### (1 line)
            z3 = forward_propagation(minibatch_X, parameters)
            ### END CODE HERE ###

            # Cost function
            ### START CODE HERE ### (1 line)
            cost = compute_cost(z3, minibatch_Y)
            ### END CODE HERE ###

            # Backward propagation
            cost.backward()

            # Update parameters
            optimizer.step()

            epoch_cost += cost.item() / num_minibatches

        # Print the cost every epoch
        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    import matplotlib.pyplot as plt
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Convert parameters to numpy for compatibility
    parameters_numpy = {}
    for key in parameters:
        parameters_numpy[key] = parameters[key].detach().numpy()

    print ("Parameters have been trained!")

    # Calculate the correct predictions
    with torch.no_grad():
        X_train_tensor = torch.from_numpy(X_train).float()
        Y_train_tensor = torch.from_numpy(Y_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        Y_test_tensor = torch.from_numpy(Y_test).float()

        train_predictions = forward_propagation(X_train_tensor, parameters)
        test_predictions = forward_propagation(X_test_tensor, parameters)

        train_correct = (torch.argmax(train_predictions, dim=0) == torch.argmax(Y_train_tensor, dim=0)).float()
        test_correct = (torch.argmax(test_predictions, dim=0) == torch.argmax(Y_test_tensor, dim=0)).float()

        train_accuracy = torch.mean(train_correct)
        test_accuracy = torch.mean(test_correct)

    print ("Train Accuracy:", train_accuracy.item())
    print ("Test Accuracy:", test_accuracy.item())

    return parameters_numpy
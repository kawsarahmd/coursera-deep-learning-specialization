#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from numpy import genfromtxt
import h5py
import matplotlib.pyplot as plt
import PIL
from PIL import Image


# Weight names for loading pre-trained weights
WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
    'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x5_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'dense_layer'
]

conv_shape = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1],
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1,],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1],
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1],
    'inception_5b_1x1_conv': [256, 736, 1, 1],
}


def load_weights():
    """Load pre-trained weights from CSV files"""
    # Set weights path
    dirPath = './weights'
    if not os.path.exists(dirPath):
        print(f"Warning: weights directory '{dirPath}' not found. Skipping weight loading.")
        return {}

    fileNames = [f for f in os.listdir(dirPath) if not f.startswith('.')]
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            weight_path = paths.get(name + '_w')
            bias_path = paths.get(name + '_b')

            if weight_path and bias_path:
                conv_w = genfromtxt(weight_path, delimiter=',', dtype=None)
                conv_w = np.reshape(conv_w, conv_shape[name])
                # Convert from TensorFlow format (H, W, in_C, out_C) to PyTorch format (out_C, in_C, H, W)
                conv_w = np.transpose(conv_w, (3, 2, 0, 1))
                conv_b = genfromtxt(bias_path, delimiter=',', dtype=None)
                weights_dict[name] = [torch.from_numpy(conv_w).float(),
                                     torch.from_numpy(conv_b).float()]
        elif 'bn' in name:
            w_path = paths.get(name + '_w')
            b_path = paths.get(name + '_b')
            m_path = paths.get(name + '_m')
            v_path = paths.get(name + '_v')

            if all([w_path, b_path, m_path, v_path]):
                bn_w = genfromtxt(w_path, delimiter=',', dtype=None)
                bn_b = genfromtxt(b_path, delimiter=',', dtype=None)
                bn_m = genfromtxt(m_path, delimiter=',', dtype=None)
                bn_v = genfromtxt(v_path, delimiter=',', dtype=None)
                weights_dict[name] = [torch.from_numpy(bn_w).float(),
                                     torch.from_numpy(bn_b).float(),
                                     torch.from_numpy(bn_m).float(),
                                     torch.from_numpy(bn_v).float()]
        elif 'dense' in name:
            dense_w_path = dirPath + '/dense_w.csv'
            dense_b_path = dirPath + '/dense_b.csv'

            if os.path.exists(dense_w_path) and os.path.exists(dense_b_path):
                dense_w = genfromtxt(dense_w_path, delimiter=',', dtype=None)
                dense_w = np.reshape(dense_w, (128, 736))
                # Transpose for PyTorch (input_features, output_features) -> (output_features, input_features)
                dense_w = np.transpose(dense_w, (0, 1))  # Keep as is since PyTorch Linear expects (out, in)
                dense_b = genfromtxt(dense_b_path, delimiter=',', dtype=None)
                weights_dict[name] = [torch.from_numpy(dense_w).float(),
                                     torch.from_numpy(dense_b).float()]

    return weights_dict


def load_weights_from_FaceNet(model):
    """
    Load pre-trained weights into the FaceNet model

    Arguments:
    model -- PyTorch FaceRecoModel instance
    """
    weights_dict = load_weights()

    if not weights_dict:
        print("No weights loaded. Model will use random initialization.")
        return

    # Create a mapping from weight names to model layers
    layer_mapping = {
        'conv1': model.conv1,
        'bn1': model.bn1,
        'conv2': model.conv2,
        'bn2': model.bn2,
        'conv3': model.conv3,
        'bn3': model.bn3,
        'dense_layer': model.fc,
    }

    # Load weights for each layer
    for name, weights in weights_dict.items():
        if name in layer_mapping:
            layer = layer_mapping[name]
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = weights[0]
                layer.bias.data = weights[1]
            elif isinstance(nn.BatchNorm2d):
                layer.weight.data = weights[0]
                layer.bias.data = weights[1]
                layer.running_mean.data = weights[2]
                layer.running_var.data = weights[3]
            elif isinstance(layer, nn.Linear):
                layer.weight.data = weights[0]
                layer.bias.data = weights[1]

    print("Weights loaded successfully from FaceNet pre-trained model.")


def load_dataset():
    """Load the happy face dataset"""
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


def img_to_encoding(image_path, model, target_size=(96, 96)):
    """
    Convert an image to its encoding using the FaceNet model

    Arguments:
    image_path -- path to the image file
    model -- PyTorch FaceRecoModel instance
    target_size -- target size for resizing (default: (96, 96) for old version, (160, 160) for new)

    Returns:
    embedding -- normalized embedding vector of shape (1, 128)
    """
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)

    # Normalize pixel values
    img = np.around(img / 255.0, decimals=12)

    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    # Convert to PyTorch format (channels first)
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).float()

    # Set model to evaluation mode
    model.eval()

    # Get embedding
    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.numpy()

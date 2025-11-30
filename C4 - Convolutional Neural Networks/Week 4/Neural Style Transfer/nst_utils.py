### Part of this code is due to the MatConvNet team and is used to load the parameters of the pretrained VGG19 model in the notebook ###

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # For PyTorch, we'll use (1, 3, 1, 1) format
    MEANS_TORCH = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    STYLE_IMAGE = 'images/stone_style.jpg'
    CONTENT_IMAGE = 'images/content300.jpg'
    OUTPUT_DIR = 'output/'


def load_vgg_model(weights_path=None):
    """
    Returns a VGG19 model for Neural Style Transfer using PyTorch.

    This function loads a pre-trained VGG19 model and extracts the convolutional
    layers for style transfer. Unlike the original TensorFlow implementation that
    loaded from a .mat file, this uses torchvision's pre-trained VGG19.

    Arguments:
    weights_path -- Optional path to custom weights (not used, kept for compatibility)

    Returns:
    model -- A dictionary containing the VGG19 convolutional layers
    """
    # Load pre-trained VGG19
    vgg = models.vgg19(pretrained=True).features

    # Set to evaluation mode
    vgg.eval()

    # Freeze all parameters
    for param in vgg.parameters():
        param.requires_grad = False

    # Create a dictionary mapping layer names to their outputs
    # VGG19 feature layers structure:
    # 0: conv1_1, 1: relu, 2: conv1_2, 3: relu, 4: maxpool
    # 5: conv2_1, 6: relu, 7: conv2_2, 8: relu, 9: maxpool
    # 10: conv3_1, 11: relu, 12: conv3_2, 13: relu, 14: conv3_3, 15: relu, 16: conv3_4, 17: relu, 18: maxpool
    # 19: conv4_1, 20: relu, 21: conv4_2, 22: relu, 23: conv4_3, 24: relu, 25: conv4_4, 26: relu, 27: maxpool
    # 28: conv5_1, 29: relu, 30: conv5_2, 31: relu, 32: conv5_3, 33: relu, 34: conv5_4, 35: relu, 36: maxpool

    layer_names = {
        0: 'conv1_1',
        2: 'conv1_2',
        5: 'conv2_1',
        7: 'conv2_2',
        10: 'conv3_1',
        12: 'conv3_2',
        14: 'conv3_3',
        16: 'conv3_4',
        19: 'conv4_1',
        21: 'conv4_2',
        23: 'conv4_3',
        25: 'conv4_4',
        28: 'conv5_1',
        30: 'conv5_2',
        32: 'conv5_3',
        34: 'conv5_4'
    }

    return vgg, layer_names


def get_vgg_layers(vgg_model, layer_names, input_image):
    """
    Extract feature maps from specified VGG layers

    Arguments:
    vgg_model -- The VGG19 features model
    layer_names -- Dictionary mapping layer indices to names
    input_image -- Input image tensor

    Returns:
    outputs -- Dictionary of layer outputs
    """
    outputs = {}
    x = input_image

    for idx, layer in enumerate(vgg_model):
        x = layer(x)
        if idx in layer_names:
            outputs[layer_names[idx]] = x

    return outputs


def generate_noise_image(content_image, noise_ratio=CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image

    Arguments:
    content_image -- Content image tensor of shape (1, 3, height, width)
    noise_ratio -- Ratio of noise to add (default from CONFIG)

    Returns:
    input_image -- Noisy image tensor
    """
    # Get the dimensions
    _, channels, height, width = content_image.shape

    # Generate random noise
    noise_image = torch.rand(1, channels, height, width) * 40 - 20  # Range: [-20, 20]

    # Create weighted average
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)

    Arguments:
    image -- Input image as numpy array or PIL Image

    Returns:
    image -- Preprocessed image tensor
    """
    if isinstance(image, np.ndarray):
        # If it's a numpy array
        if len(image.shape) == 3:
            # Add batch dimension
            image = np.expand_dims(image, axis=0)

        # Convert to tensor (assuming image is in HWC format)
        if image.shape[-1] == 3:
            # Convert from HWC to CHW
            image = np.transpose(image, (0, 3, 1, 2))

        image = torch.from_numpy(image).float()

    # Subtract mean for VGG preprocessing
    # VGG was trained on ImageNet with these mean values
    mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    image = image - mean

    return image


def save_image(path, image):
    """
    Save the generated image

    Arguments:
    path -- Path to save the image
    image -- Image tensor of shape (1, 3, height, width) or (3, height, width)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clone the image to avoid modifying the original
    image = image.clone()

    # Add mean back
    mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    image = image + mean

    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = image.squeeze(0)

    # Clip values to valid range
    image = torch.clamp(image, 0, 255)

    # Convert to numpy and change to HWC format
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Save using PIL
    Image.fromarray(image).save(path)


def load_image(image_path, target_size=(400, 300)):
    """
    Load and preprocess an image for NST

    Arguments:
    image_path -- Path to the image file
    target_size -- Tuple of (width, height) for resizing

    Returns:
    image -- Preprocessed image tensor of shape (1, 3, height, width)
    """
    img = Image.open(image_path)
    # Resize: PIL uses (width, height), but we want (width, height)
    img = img.resize(target_size)
    img = np.array(img).astype(np.float32)

    # Add batch dimension and convert to tensor
    img = np.expand_dims(img, axis=0)  # Shape: (1, height, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))  # Shape: (1, 3, height, width)
    img = torch.from_numpy(img)

    # Subtract VGG mean
    mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    img = img - mean

    return img


def tensor_to_image(tensor):
    """
    Convert a tensor to a displayable image

    Arguments:
    tensor -- Image tensor

    Returns:
    image -- NumPy array suitable for display
    """
    # Clone and detach
    image = tensor.clone().detach()

    # Add mean back
    mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    image = image + mean

    # Remove batch dimension
    if len(image.shape) == 4:
        image = image.squeeze(0)

    # Clip and convert
    image = torch.clamp(image, 0, 255)
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    return image


def clip_0_1(image):
    """
    Clip image tensor values between 0 and 1 (after denormalization)

    Arguments:
    image -- Image tensor

    Returns:
    clipped_image -- Clipped image tensor
    """
    # This is used in the newer version for normalized images
    return torch.clamp(image, 0.0, 1.0)


# For compatibility with the newer notebook version
def preprocess_image(image_path, target_size=(400, 400)):
    """
    Load and preprocess image for the newer NST implementation

    Arguments:
    image_path -- Path to image file
    target_size -- Target size (width, height)

    Returns:
    image -- Preprocessed tensor in [0, 1] range
    """
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img).astype(np.float32) / 255.0

    # Add batch dimension and convert to CHW format
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    img = np.transpose(img, (0, 3, 1, 2))  # (1, C, H, W)

    return torch.from_numpy(img).float()

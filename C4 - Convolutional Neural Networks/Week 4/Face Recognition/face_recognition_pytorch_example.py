"""
Face Recognition with PyTorch - Complete Example
=================================================

This script demonstrates the complete Face Recognition system using PyTorch,
converted from the original TensorFlow/Keras implementation.

Key Components:
1. Triplet Loss Implementation
2. FaceNet Model (Inception Architecture)
3. Face Verification
4. Face Recognition

Usage:
    python face_recognition_pytorch_example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# Import the converted modules
from inception_blocks_v2 import faceRecoModel
from fr_utils import img_to_encoding


# ==============================================================================
# Exercise 1: Triplet Loss
# ==============================================================================

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by the FaceNet paper.

    The triplet loss encourages:
    - Anchor and Positive to be close
    - Anchor and Negative to be far apart (by at least margin alpha)

    Arguments:
    y_true -- Not used (required for compatibility)
    y_pred -- Tuple of (anchor, positive, negative) embeddings
              Each embedding shape: (batch_size, 128)
    alpha -- Margin value (default: 0.2)

    Returns:
    loss -- Scalar tensor representing the triplet loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the distance between anchor and positive
    # ||f(A) - f(P)||^2
    pos_dist = torch.sum(torch.square(anchor - positive), dim=-1)

    # Step 2: Compute the distance between anchor and negative
    # ||f(A) - f(N)||^2
    neg_dist = torch.sum(torch.square(anchor - negative), dim=-1)

    # Step 3: Subtract the two distances and add alpha
    # ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha
    basic_loss = pos_dist - neg_dist + alpha

    # Step 4: Take the maximum of basic_loss and 0, then sum
    # max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0)
    loss = torch.sum(torch.maximum(basic_loss, torch.tensor(0.0)))

    return loss


def test_triplet_loss():
    """Test the triplet loss function"""
    print("="*70)
    print("Testing Triplet Loss")
    print("="*70)

    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Create dummy data
    y_true = (None, None, None)  # Not used
    y_pred = (
        torch.randn(3, 128),  # Anchor
        torch.randn(3, 128),  # Positive
        torch.randn(3, 128)   # Negative
    )

    # Compute loss
    loss = triplet_loss(y_true, y_pred)

    print(f"Triplet Loss: {loss.item():.4f}")
    print(f"Expected: ~527.26 (may vary due to random initialization)\n")

    # Test edge cases
    # Case 1: Perfect match (A == P, A != N)
    perfect_anchor = torch.ones(2, 128)
    perfect_positive = torch.ones(2, 128)
    perfect_negative = torch.zeros(2, 128)
    loss_perfect = triplet_loss(None, (perfect_anchor, perfect_positive, perfect_negative), alpha=0.5)
    print(f"Perfect match loss: {loss_perfect.item():.4f}")
    print(f"(Should be close to 0 since d(A,P)=0 and d(A,N)=128, so loss is clamped to 0)\n")


# ==============================================================================
# Exercise 2: Face Verification
# ==============================================================================

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the image is the claimed identity.

    Arguments:
    image_path -- Path to an image file
    identity -- String, name of the person to verify
    database -- Dictionary mapping names to their encodings (vectors)
    model -- PyTorch FaceRecoModel instance

    Returns:
    dist -- L2 distance between the image encoding and the identity encoding
    door_open -- True if the door should open, False otherwise
    """
    # Step 1: Compute the encoding for the image
    encoding = img_to_encoding(image_path, model, target_size=(160, 160))

    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print(f"It's {identity}, welcome in!")
        door_open = True
    else:
        print(f"It's not {identity}, please go away")
        door_open = False

    return float(dist), door_open


# ==============================================================================
# Exercise 3: Face Recognition
# ==============================================================================

def who_is_it(image_path, database, model):
    """
    Implements face recognition by finding who is the person on the image.

    Arguments:
    image_path -- Path to an image file
    database -- Dictionary containing image encodings along with the person's name
    model -- PyTorch FaceRecoModel instance

    Returns:
    min_dist -- The minimum distance found between the image encoding and database encodings
    identity -- String, the name prediction for the person on image_path
    """
    # Step 1: Compute the target encoding for the image
    encoding = img_to_encoding(image_path, model, target_size=(160, 160))

    # Step 2: Find the closest encoding in the database
    # Initialize min_dist to a large value
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target encoding and the current encoding
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than min_dist, update min_dist and identity
        if dist < min_dist:
            min_dist = dist
            identity = name

    # Check if the person is in the database
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print(f"it's {identity}, the distance is {min_dist:.6f}")

    return float(min_dist), identity


# ==============================================================================
# Main Demo
# ==============================================================================

def main():
    """
    Main demonstration of Face Recognition system
    """
    print("\n" + "="*70)
    print("PyTorch Face Recognition System")
    print("="*70 + "\n")

    # Test triplet loss
    test_triplet_loss()

    # Create model
    print("="*70)
    print("Creating FaceNet Model")
    print("="*70)

    try:
        # Initialize model for 160x160 images (newer version)
        # For the older version, use (3, 96, 96)
        model = faceRecoModel(input_shape=(3, 160, 160))
        model.eval()
        print(f"Model created successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    except Exception as e:
        print(f"Error creating model: {e}\n")
        return

    # Build database (demo with dummy encodings)
    print("="*70)
    print("Building Face Database")
    print("="*70)

    # In a real scenario, you would load actual images and compute encodings
    # For demonstration, we'll create dummy encodings
    database = {}

    # Create some example encodings (in practice, use img_to_encoding)
    print("Note: Using dummy encodings for demonstration.")
    print("In practice, use img_to_encoding() with real images.\n")

    # Example of how to create real encodings (commented out for demo):
    # if os.path.exists("images/danielle.png"):
    #     database["danielle"] = img_to_encoding("images/danielle.png", model, (160, 160))
    # if os.path.exists("images/younes.jpg"):
    #     database["younes"] = img_to_encoding("images/younes.jpg", model, (160, 160))

    # For demo purposes, create dummy encodings
    np.random.seed(42)
    database["person1"] = np.random.randn(1, 128)
    database["person2"] = np.random.randn(1, 128)
    database["person3"] = np.random.randn(1, 128)

    print(f"Database created with {len(database)} people")
    for name in database.keys():
        print(f"  - {name}")
    print()

    # Demo verification (with dummy data)
    print("="*70)
    print("Face Verification Demo")
    print("="*70)
    print("Note: This is a simulation using dummy data.")
    print("To run with real images, uncomment the image loading code above.\n")

    # Example verification code (would work with real images):
    # if os.path.exists("images/camera_0.jpg"):
    #     dist, door_open = verify("images/camera_0.jpg", "younes", database, model)
    #     print(f"Distance: {dist:.4f}")
    #     print(f"Door open: {door_open}\n")

    # Demo recognition
    print("="*70)
    print("Face Recognition Demo")
    print("="*70)
    print("Note: This is a simulation using dummy data.")
    print("To run with real images, uncomment the image loading code above.\n")

    # Example recognition code (would work with real images):
    # if os.path.exists("images/camera_0.jpg"):
    #     min_dist, identity = who_is_it("images/camera_0.jpg", database, model)
    #     print(f"Minimum distance: {min_dist:.4f}")
    #     print(f"Identified as: {identity}\n")

    print("="*70)
    print("Summary")
    print("="*70)
    print("✓ Triplet loss implemented and tested")
    print("✓ FaceNet model (Inception architecture) created")
    print("✓ Face verification function ready")
    print("✓ Face recognition function ready")
    print("\nTo use with real images:")
    print("1. Ensure images are in the 'images/' directory")
    print("2. Uncomment the image loading code in this script")
    print("3. Run the script again")
    print("\nFor pre-trained weights, place them in the 'weights/' directory")
    print("and they will be loaded automatically.\n")


if __name__ == "__main__":
    main()

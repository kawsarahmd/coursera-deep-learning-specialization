# PyTorch Version - Coursera Deep Learning Specialization

This folder contains the PyTorch converted code from the original TensorFlow-based Coursera Deep Learning Specialization.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### What's Included

**Course 2: Improving Deep Neural Networks**
- Week 3: TensorFlow Introduction (converted to PyTorch)
  - `tf_utils.py` - Utility functions
  - `improv_utils.py` - Model training utilities
  - Jupyter notebooks

**Course 4: Convolutional Neural Networks**
- Week 1: CNN Basics
- Week 2: ResNets, Keras Tutorial, Transfer Learning
- Week 3: YOLO Car Detection, U-Net Image Segmentation
- Week 4: Face Recognition, Neural Style Transfer

**Course 5: Sequence Models**
- Week 1: Jazz Improvisation with LSTM (partial)

## Key Differences from TensorFlow

1. **Imports**: Uses `torch`, `torch.nn`, `torch.nn.functional` instead of TensorFlow/Keras
2. **Models**: Defined as `nn.Module` classes with `forward()` methods
3. **Training**: Explicit training loops with `optimizer.step()`
4. **Data Format**: NCHW (batch, channels, height, width) instead of NHWC
5. **Tensors**: PyTorch tensors instead of TensorFlow tensors

## Running Notebooks

1. Install dependencies: `pip install -r requirements.txt`
2. Start Jupyter: `jupyter notebook`
3. Navigate to the desired course/week
4. Open and run the notebook

## Running Python Scripts

```python
# Example: Using C2 Week 3 utilities
from C2_Improving_Deep_Neural_Networks.Week_3 import improv_utils

# Load data and train model
X_train, Y_train, X_test, Y_test, classes = improv_utils.load_dataset()
parameters = improv_utils.model(X_train, Y_train, X_test, Y_test)
```

## Notes

- All converted code uses PyTorch 2.5+
- Some notebooks are partially converted (work in progress)
- Refer to original notebooks for assignment instructions
- Dataset paths remain the same as original structure

## File Count

- Python files: 37
- Jupyter notebooks: 25

## Support

For issues or questions:
1. Check the main repository documentation
2. Refer to PyTorch documentation: https://pytorch.org/docs/
3. See PYTORCH_MIGRATION_GUIDE.md in the main repository

---

**Note**: This is an ongoing conversion. Some advanced topics (RNNs, Attention, Transformers in C5) are still being converted.

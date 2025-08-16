# Convolutional Neural Network (CNN) for Image Classification

This project implements Convolutional Neural Networks for image classification using TensorFlow/Keras. It includes two different implementations:
1. CNN on MNIST digit classification dataset
2. Modified CNN on Fashion-MNIST dataset with enhanced architecture

## Features

- **Multiple Datasets**: MNIST digits and Fashion-MNIST clothing items
- **Different CNN Architectures**: Basic and enhanced CNN models
- **Comprehensive Evaluation**: Training/validation curves, confusion matrix, and classification reports
- **Visualization**: Weight visualization and performance plots
- **Reproducible Results**: Fixed random seeds for consistent outputs

## Requirements

```bash
tensorflow>=2.0
numpy
matplotlib
scikit-learn
seaborn
```

## Model Architectures

### Architecture 1: Basic CNN (MNIST)
```
Input (28×28×1)
├── Conv2D(16, 3×3) + ReLU
├── MaxPooling2D(2×2)
├── Conv2D(64, 3×3) + ReLU
├── MaxPooling2D(2×2)
├── Conv2D(32, 3×3) + ReLU
├── Flatten
├── Dense(64) + ReLU
├── Dense(32) + ReLU
└── Dense(10) + Softmax
```

### Architecture 2: Enhanced CNN (Fashion-MNIST)
```
Input (28×28×1)
├── Conv2D(32, 3×3) + ReLU
├── MaxPooling2D(2×2)
├── Conv2D(64, 3×3) + ReLU
├── MaxPooling2D(2×2)
├── Conv2D(128, 3×3) + ReLU
├── MaxPooling2D(2×2)
├── Flatten
├── Dense(256) + ReLU
├── Dense(128) + ReLU
└── Dense(10) + Softmax
```

## Datasets

### MNIST Dataset
- **Description**: Handwritten digits (0-9)
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale

### Fashion-MNIST Dataset
- **Description**: Fashion items (T-shirt, Trouser, Pullover, etc.)
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale
- **Classes**: 10 different clothing categories

## Installation & Usage

1. **Clone or download the project**
```bash
# If using git
git clone https://github.com/0xafraidoftime/CNN-for-Image-Classification
cd CNN-for-Image-Classification
```

2. **Install dependencies**
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn jupyter
```

3. **Run the code**

   **Option A: Using Jupyter Notebook (Recommended)**
   ```bash
   jupyter notebook
   # Open 'Convolution Neural Network.ipynb' in the browser
   ```

   **Option B: Using Python script**
   ```bash
   python convolution_neural_network.py
   ```

   **Option C: Using Google Colab**
   - Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/)
   - All dependencies are pre-installed in Colab

## Model Training

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 25
- **Batch Size**: 64
- **Validation Split**: 10%

### Data Preprocessing
- Reshape images to (28, 28, 1)
- Normalize pixel values to [0, 1] range
- Convert labels to one-hot encoding

## Evaluation Metrics

The project provides comprehensive evaluation including:

1. **Training History Plots**
   - Training vs Validation Loss
   - Training vs Validation Accuracy

2. **Model Performance**
   - Final training/validation accuracy and loss
   - Test accuracy and loss

3. **Detailed Analysis**
   - Classification report (precision, recall, F1-score)
   - Confusion matrix heatmap
   - First layer weight visualization

## Expected Results

### MNIST (Basic Architecture)
- Training Accuracy: ~99%
- Test Accuracy: ~98-99%

### Fashion-MNIST (Enhanced Architecture)
- Training Accuracy: ~92-95%
- Test Accuracy: ~88-91%

*Note: Fashion-MNIST is more challenging than MNIST due to the complexity of fashion items*

## Project Structure

```
├── Convolution Neural Network.ipynb    # Jupyter Notebook with interactive cells
├── convolution_neural_network.py       # Python script version
└── README.md                           # Project documentation
```

### Code Structure (Both Files)
```
├── Import libraries and setup
├── MNIST Implementation
│   ├── Data loading and preprocessing
│   ├── Model definition (Basic CNN)
│   ├── Training and evaluation
│   └── Visualization and analysis
└── Fashion-MNIST Implementation
    ├── Data loading and preprocessing
    ├── Model definition (Enhanced CNN)
    ├── Training and evaluation
    └── Visualization and analysis
```

## Key Features Explained

### Reproducibility
- Fixed random seeds (42) for NumPy, Python random, and TensorFlow
- Ensures consistent results across runs

### Model Visualization
- Displays learned filters from the first convolutional layer
- Shows feature maps as grayscale images

### Performance Monitoring
- Real-time training progress
- Validation split to monitor overfitting
- Comprehensive metrics for model evaluation

## Modifications Made

The second implementation includes several improvements:
- **Dataset**: Changed from MNIST to Fashion-MNIST
- **Architecture**: Enhanced with more filters and neurons
- **Kernel Configuration**: Different kernel sizes and counts
- **Deeper Network**: Additional layers for better feature extraction

## Contributing

Feel free to fork this project and submit pull requests for:
- Additional datasets
- New architectures
- Performance improvements
- Bug fixes

## Notes

- **Dual Format**: Available as both Jupyter Notebook (`.ipynb`) and Python script (`.py`)
- **Interactive Experience**: Jupyter Notebook provides cell-by-cell execution and inline plots
- **Google Colab Ready**: The notebook was originally created in Google Colab
- **Matplotlib Integration**: Includes `%matplotlib inline` magic command for notebook environments
- Comments explain each major section
- All visualizations are automatically generated
- Model summary is displayed before training

## Jupyter Notebook Benefits

- **Interactive Execution**: Run code cells individually
- **Inline Visualizations**: Plots appear directly below code cells
- **Easy Experimentation**: Modify parameters and re-run specific sections
- **Documentation**: Mix code, results, and markdown explanations
- **Google Colab Integration**: Seamless cloud-based execution

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

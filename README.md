
# Lumber Defect Classifier

This project classifies lumber defects using texture features extracted from grayscale image sections. It implements a `BayesianDefectClassifier` for defect classification based on statistical and co-occurrence matrix-based texture features. The program is available in both **Python** and **C++**.

[Identifying and Locating Surface Defects in Wood: An Automated System](https://www.semanticscholar.org/paper/Identifying-and-Locating-Surface-Defects-in-Wood%3A-Conners-McMillin/bbb3e359bdc32e5771f4d193726893e10913fe1e)

## Features

- Extracts statistical features: Mean, Variance, Skewness, Kurtosis.
- Extracts texture features from the co-occurrence matrix: Inertia, Cluster Shade, Cluster Prominence, Local Homogeneity, Energy, and Entropy.
- Implements a Bayesian Classifier for defect classification.
- Available in **Python** and **C++** for flexibility and ease of use.

## Requirements

### C++ Version
- CMake (version 3.10 or higher)
- OpenCV (version 4.x recommended)
- C++17 compatible compiler

### Python Version
- Python 3.8 or higher
- NumPy
- OpenCV-Python

## Build Instructions

### C++ Version
1. **Clone the Repository**
   ```bash
   git clone https://github.com/MohammadrezaMC2/LumberDefectClassifier.git
   cd LumberDefectClassifier
   ```

2. **Create a Build Directory**
   ```bash
   mkdir build
   cd build
   ```

3. **Run CMake and Build**
   ```bash
   cmake ..
   make
   ```
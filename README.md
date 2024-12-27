# Lumber Defect Classifier

This project classifies lumber defects using texture features extracted from grayscale image sections. It implements a `BayesianDefectClassifier` for defect classification based on statistical and co-occurrence matrix-based texture features.

## Features

- Extracts statistical features: Mean, Variance, Skewness, Kurtosis.
- Extracts texture features from the co-occurrence matrix: Inertia, Cluster Shade, Cluster Prominence, Local Homogeneity, Energy, and Entropy.
- Implements a Bayesian Classifier for defect classification.

## Requirements

- CMake (version 3.10 or higher)
- OpenCV (version 4.x recommended)
- C++17 compatible compiler

## Build Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MohammadrezaMC2/LumberDefectClassifier.git
   cd LumberDefectClassifier

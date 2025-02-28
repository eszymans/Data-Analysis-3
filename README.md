# k-NN Classification Analysis

## Overview

This project is a group assignment for university studies. It focuses on implementing and analyzing the k-nearest neighbors (k-NN) algorithm, a classic method for data classification.

## Features

- **k-NN Algorithm Implementation**
  - Implements the k-NN classification algorithm
  - Considers tie-breaking and data normalization.
  - Demonstrates the algorithm’s functionality using a modified Iris dataset.
- **Classification and Performance Evaluation**
  - Uses a dataset split into a training set (105 samples) and a test set (45 samples).
  - Performs classification for k values ranging from 1 to 15.
  - Computes and visualizes classification accuracy for each k.
  - Displays a confusion matrix for the best-performing k.
- **Feature Pair Analysis**
  - Repeats classification using only two features at a time.
  - Generates accuracy plots and confusion matrices for each feature pair.

## Output

The program generates:

- Accuracy plots for k = 1 to 15 (using all four features and each feature pair).
- A confusion matrix for the best-performing k.

## Dataset

The dataset is a modified version of the Iris dataset, containing 150 samples from three species:

- **Setosa**
- **Versicolor**
- **Virginica**
  
Each sample includes four numerical features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

## License

This project is open-source and available under the MIT License.

## Authors

This project was developed as a group assignment by:

- Edyta Szymańska
- Alicja Bartczak


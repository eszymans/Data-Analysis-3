import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import textwrap

def normalize_data(data, data_for_minmax):
    for column in data.columns:
        max_value = data_for_minmax[column].max()
        min_value = data_for_minmax[column].min()
        for row in data.index:
            data.loc[row, column] = (data.iloc[row, column] - min_value) / (max_value - min_value)
    return data

# Load data
test_data = pd.read_csv("data3_test.csv", header=None)
training_data = pd.read_csv("data3_train.csv", header=None)

# Separate features and labels
normalized_test_data = test_data.iloc[:, :-1]
test_labels = test_data.iloc[:, 4]
normalized_training_data = training_data.iloc[:, :-1]
training_labels = training_data.iloc[:, 4]

# Normalize data
normalized_test_data = normalize_data(normalized_test_data, training_data)
normalized_training_data = normalize_data(normalized_training_data, training_data)

# KNN and accuracy calculation
k_values = range(1, 16)
accuracies_all_features = []

# kNN for all four features
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(normalized_training_data, training_labels)
    test_predictions = knn.predict(normalized_test_data)
    accuracy = accuracy_score(test_labels, test_predictions)
    accuracies_all_features.append(accuracy * 100)

best_k = np.argmax(accuracies_all_features) + 1  # Corrected to find the index of the highest accuracy
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(normalized_training_data, training_labels)
best_test_predictions = best_knn.predict(normalized_test_data)
conf_matrix_all_features = confusion_matrix(test_labels, best_test_predictions)  # Confusion matrix
print(accuracies_all_features)
print(conf_matrix_all_features)


# For all pairs of two features
def knn_for_two_features(test_data, training_data, test_labels, training_labels):
    accuracies = []
    k_values = range(1, 16)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_data, training_labels)
        test_predictions = knn.predict(test_data)
        accuracy = accuracy_score(test_labels, test_predictions)
        accuracies.append(accuracy * 100)
    return accuracies

def find_best_k_matrix (accuracies, test_data, training_data, test_labels, training_labels):
    best_k = np.argmax(accuracies) + 1
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(training_data, training_labels)
    test_predictions = best_knn.predict(test_data)
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    return conf_matrix, best_k

# Features 0 and 1
accuracies_01 = knn_for_two_features(normalized_test_data.iloc[:, [0, 1]], normalized_training_data.iloc[:, [0, 1]], test_labels, training_labels)
conf_matrix_01, best_k_01 = find_best_k_matrix(accuracies_01, normalized_test_data.iloc[:, [0, 1]], normalized_training_data.iloc[:, [0, 1]], test_labels, training_labels)
print(f"Best k for features [0, 1]: {best_k_01}")
print("Confusion Matrix for features [0, 1]:")
print(conf_matrix_01)

# Features 0 and 2
accuracies_02 = knn_for_two_features(normalized_test_data.iloc[:, [0, 2]], normalized_training_data.iloc[:, [0, 2]], test_labels, training_labels)
conf_matrix_02, best_k_02 = find_best_k_matrix(accuracies_02, normalized_test_data.iloc[:, [0, 2]], normalized_training_data.iloc[:, [0, 2]], test_labels, training_labels)
print(f"Best k for features [0, 2]: {best_k_02}")
print("Confusion Matrix for features [0, 2]:")
print(conf_matrix_02)

# Features 0 and 3
accuracies_03 = knn_for_two_features(normalized_test_data.iloc[:, [0, 3]], normalized_training_data.iloc[:, [0, 3]], test_labels, training_labels)
conf_matrix_03, best_k_03 = find_best_k_matrix(accuracies_03, normalized_test_data.iloc[:, [0, 3]], normalized_training_data.iloc[:, [0, 3]], test_labels, training_labels)
print(f"Best k for features [0, 3]: {best_k_03}")
print("Confusion Matrix for features [0, 3]:")
print(conf_matrix_03)

# Features 1 and 2
accuracies_12 = knn_for_two_features(normalized_test_data.iloc[:, [1, 2]], normalized_training_data.iloc[:, [1, 2]], test_labels, training_labels)
conf_matrix_12, best_k_12 = find_best_k_matrix(accuracies_12, normalized_test_data.iloc[:, [1, 2]], normalized_training_data.iloc[:, [1, 2]], test_labels, training_labels)
print(f"Best k for features [1, 2]: {best_k_12}")
print("Confusion Matrix for features [1, 2]:")
print(conf_matrix_12)

# Features 1 and 3
accuracies_13 = knn_for_two_features(normalized_test_data.iloc[:, [1, 3]], normalized_training_data.iloc[:, [1, 3]], test_labels, training_labels)
conf_matrix_13, best_k_13 = find_best_k_matrix(accuracies_13, normalized_test_data.iloc[:, [1, 3]], normalized_training_data.iloc[:, [1, 3]], test_labels, training_labels)
print(f"Best k for features [1, 3]: {best_k_13}")
print("Confusion Matrix for features [1, 3]:")
print(conf_matrix_13)

# Features 2 and 3
accuracies_23 = knn_for_two_features(normalized_test_data.iloc[:, [2, 3]], normalized_training_data.iloc[:, [2, 3]], test_labels, training_labels)
conf_matrix_23, best_k_23 = find_best_k_matrix(accuracies_23, normalized_test_data.iloc[:, [2, 3]], normalized_training_data.iloc[:, [2, 3]], test_labels, training_labels)
print(f"Best k for features [2, 3]: {best_k_23}")
print("Confusion Matrix for features [2, 3]:")
print(conf_matrix_23)

# Print accuracies to verify
print("Accuracies for features [0, 1]:", accuracies_01)
print("Accuracies for features [0, 2]:", accuracies_02)
print("Accuracies for features [0, 3]:", accuracies_03)
print("Accuracies for features [1, 2]:", accuracies_12)
print("Accuracies for features [1, 3]:", accuracies_13)
print("Accuracies for features [2, 3]:", accuracies_23)

# 1. Znalezienie minimalnej i maksymalnej wartości dokładności
min_accuracy = min(min(accuracies_all_features),
                   min(accuracies_01),
                   min(accuracies_02),
                   min(accuracies_03),
                   min(accuracies_12),
                   min(accuracies_13),
                   min(accuracies_23))
max_accuracy = max(max(accuracies_all_features),
                   max(accuracies_01),
                   max(accuracies_02),
                   max(accuracies_03),
                   max(accuracies_12),
                   max(accuracies_13),
                   max(accuracies_23))

# Dodanie marginesu (np. 2% różnicy między max i min)
margin = 0.02 * (max_accuracy - min_accuracy)
y_min = min_accuracy - margin
y_max = max_accuracy + margin

# 2. Wykres słupkowy dla wszystkich cech
plt.figure(figsize=(12, 8))
plt.bar(k_values, accuracies_all_features, color='lightcoral', edgecolor='black')
plt.title('Dokładność klasyfikacji k-NN dla wszystkich cech', fontsize=25)
plt.xlabel('Liczba sąsiadów (k)', fontsize=22)
plt.ylabel('Dokładność (%)', fontsize=22)
plt.xticks(k_values)
plt.ylim(y_min, y_max)  # Wspólny zakres osi y
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.xticks(range(1, 16), fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# 3. Wykresy dla par cech
feature_descriptions = {
    0: "długość działki kielicha [cm]",
    1: "szerokość działki kielicha [cm]",
    2: "długość płatka [cm]",
    3: "szerokość płatka [cm]"
}

feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
accuracies = [accuracies_01, accuracies_02, accuracies_03, accuracies_12, accuracies_13, accuracies_23]

for i in range(len(feature_pairs)):
    pair = feature_pairs[i]
    acc = accuracies[i]
    feature_1 = feature_descriptions[pair[0]]
    feature_2 = feature_descriptions[pair[1]]

    long_title = f'Dokładność klasyfikacji dla cech: {feature_1} i {feature_2} przy użyciu algorytmu k-NN'
    wrapped_title = "\n".join(textwrap.wrap(long_title, width=70))

    plt.figure(figsize=(12, 8))
    plt.bar(k_values, acc, color='lightblue', edgecolor='black')
    plt.title(wrapped_title, fontsize=25)  # Użycie zawiniętego tytułu
    plt.xlabel('Liczba sąsiadów (k)', fontsize=22)
    plt.ylabel('Dokładność (%)', fontsize=22)
    plt.xticks(k_values)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xticks(range(1, 16), fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

#function to find covariance matrix
def covariance_matrix(data):
    mean_vector = np.mean(data, axis=0)
    z_matrix = data - mean_vector
    cov_matrix = np.dot(z_matrix.T, z_matrix) / (data.shape[0] - 1)
    return cov_matrix

# Load data
data = pd.read_csv('face feature vectors.csv').drop('Unnamed: 0', axis=1)
data.rename(columns={'Unnamed: 1':'Gender'}, inplace=True)

# Train-Test-Split
def train_test_split(data, class_column, test_size = 40):
    classes = data[class_column].unique()
    test, train = pd.DataFrame(), pd.DataFrame()
    for c in classes:
        currentClass = data[data[class_column] == c]
        test = pd.concat([test, currentClass.iloc[:test_size, :]], ignore_index=True)
        train = pd.concat([train, currentClass.iloc[test_size:, :]], ignore_index=True)
    train_X, train_y = train.iloc[:, 1:], train.iloc[:, 0]
    test_X, test_y = test.iloc[:, 1:], test.iloc[:, 0]
    return train_X, train_y, test_X, test_y
    
train_X, train_y, test_X, test_y = train_test_split(data, 'Gender', 5)

# Bayes Classifier
# fit function
def fit_bayes_classifier(train_X, train_y):
    classes = train_y.unique()
    cov_mats, cov_dets = {}, {}
    inv_cov_mats = {}
    for c in classes:
        cov_mats[c] = covariance_matrix(np.array(train_X[train_y == c]))
        cov_dets[c] = sp.Matrix(cov_mats[c]).det()
        inv_cov_mats[c] = np.linalg.inv(cov_mats[c])
    return cov_dets, inv_cov_mats

# Classifier function
def bayes_classifier(train_X, train_y, test_X, cov_dets, inv_cov_mats):
    classes = train_y.unique()
    dimensions = train_X.shape[1]
    class_probabilities = {}

    for c in classes:
        apriori = len(train_y[train_y == c]) / len(train_y)
        mean_vector = np.array(train_X[train_y == c].mean())
        req_matrix = test_X - mean_vector
        numerator = np.exp(-0.5 * np.dot(np.dot(req_matrix.T, inv_cov_mats[c]), req_matrix))
        denominator = np.power(2 * np.pi, dimensions / 2) * np.power(cov_dets[c], 0.5)
        class_probabilities[c] = apriori * (numerator / denominator)

    return max(zip(class_probabilities.values(), class_probabilities.keys()))[1]
    
print("Runnning... Please wait...\n")
# Prediction
print("Predictions : ")
pred_y = []
cov_dets, inv_cov_mats = fit_bayes_classifier(train_X, train_y)
for i in range(len(test_X)):
    pred_y.append(bayes_classifier(train_X, train_y, test_X.iloc[i, :], cov_dets, inv_cov_mats))
comparison = pd.DataFrame({'Actual': test_y, 'Predicted': pred_y})
print(comparison)

# Accuracy
accuracies = {}
print("\nClass Accuracies:")
for c in test_y.unique():
    accuracies[c] = len(comparison[(comparison['Actual'] == c) & (comparison['Predicted'] == c)]) / len(comparison[comparison['Actual'] == c])
    print(c, ":", accuracies[c])
accuracies['Overall'] = len(comparison[comparison['Actual'] == comparison['Predicted']]) / len(comparison)
print("\nOverall Accuracy:", accuracies['Overall'])

# Accuracy plots
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.title('Accuracy of each class and overall')
plt.show()
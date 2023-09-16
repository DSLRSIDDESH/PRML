import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#function to find covariance matrix
def covariance_matrix(data):
    mean_vector = np.mean(data, axis=0)
    z_matrix = data - mean_vector
    cov_matrix = np.dot(z_matrix.T, z_matrix) / (data.shape[0] - 1)
    return cov_matrix
    
# Load data
iris_data = pd.read_csv('iris.csv').drop('Id', axis=1)

# Test-Train-Split
def train_test_split(data, class_column, train_size = 40):
    classes = data[class_column].unique()
    test, train = pd.DataFrame(), pd.DataFrame()
    for c in classes:
        currentClass = data[data[class_column] == c]
        test = pd.concat([test, currentClass.iloc[train_size:, :]], ignore_index=True)
        train = pd.concat([train, currentClass.iloc[:train_size, :]], ignore_index=True)
    train_X, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = train_test_split(iris_data, 'Species', 40)

# Bayes Classifier
def bayes_classifier(train_X, train_y, test_X):
    classes = train_y.unique()
    dimensions = train_X.shape[1]
    class_probabilities = {}
    
    for c in classes:
        apriori = len(train_y[train_y == c]) / len(train_y)
        mean_vector = np.array(train_X[train_y == c].mean())
        cov_matrix = covariance_matrix(np.array(train_X[train_y == c]))
        cov_matrix_det = np.linalg.det(cov_matrix)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        req_matrix = test_X - mean_vector
        numerator = np.exp(-0.5 * np.dot(np.dot(req_matrix.T, inv_cov_matrix), req_matrix))
        denominator = np.sqrt(np.power(2 * np.pi, dimensions) * cov_matrix_det)
        class_probabilities[c] = apriori * (numerator / denominator)

    return max(zip(class_probabilities.values(), class_probabilities.keys()))[1]

pred_y = []
for i in range(len(test_X)):
    pred_y.append(bayes_classifier(train_X, train_y, test_X.iloc[i, :]))
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

# Accuracy Plot
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.title('Accuracy of each class')
plt.show()
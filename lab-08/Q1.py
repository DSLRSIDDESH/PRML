import numpy as np
import pandas as pd

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def covariance_matrix(data, mean_vector):
        z_matrix = data - mean_vector
        cov_matrix = np.dot(z_matrix.T, z_matrix) / (data.shape[0] - 1)
        return cov_matrix

def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2)**2))

# PCA
class PCA:
    def __init__(self, n_components=0):
        self.d = n_components

    def fit(self, X):
        self.mean_vector = np.mean(X, axis = 0)
        self.cov_mat = covariance_matrix(X, self.mean_vector)
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.cov_mat)
        index = np.argsort(self.eigen_values)[::-1]
        self.sorted_eigen_values = self.eigen_values[index]
        if self.d > 0 and self.d < 1:
            self.total_variance = np.sum(self.sorted_eigen_values)
            self.selected_eigen_values = []
            cum_variance = 0
            i = 0
            while cum_variance < self.d * self.total_variance:
                cum_variance += self.sorted_eigen_values[i]
                self.selected_eigen_values.append(self.sorted_eigen_values[i])
                i += 1
            self.selected_eigen_values = np.array(self.selected_eigen_values)
            self.d = len(self.selected_eigen_values)
        self.sorted_eigen_vectors = self.eigen_vectors[index]
        self.final_eigen_vectors = self.sorted_eigen_vectors[:, :self.d]

    def transform(self, X):
       return np.dot(X, self.final_eigen_vectors)

# KNN Classifier
class KNN():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def knn(self, test_point):
        distances = []
        for i in range(self.X_train.shape[0]):
            train_point = self.X_train[i, :]
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, self.y_train[i]))
        distances.sort()
        return distances[:self.k]
    
    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            test_point = X_test[i, :]
            k_nearest_neighbours = self.knn(test_point)
            nearest_labels = pd.DataFrame([label for _,label in k_nearest_neighbours])
            y_pred.append(nearest_labels.mode()[0][0])
        return np.array(y_pred)

# Load Data
dataset = pd.read_csv("gender.csv")
dataset

# Test Train Split
def train_test_split(dataset):
    classes = dataset.iloc[:, 1].unique()
    test = pd.DataFrame()
    train = pd.DataFrame()
    for c in classes:
        class_data = dataset[dataset.iloc[:, 1] == c]
        train = pd.concat([train, class_data.iloc[10:]])
        test = pd.concat([test, class_data.iloc[:10]])
    X_train, X_test = train.iloc[:, 2:].values, test.iloc[:, 2:].values
    y_train, y_test = train.iloc[:, 1].values, test.iloc[:, 1].values
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(dataset)

print("Running PCA...")
# Principal Component Analysis
pca = PCA(0.95)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# KNN Classifier with PCA
knn = KNN(5)
knn.fit(X_train_pca, y_train)
# Prediction
y_pred = knn.predict(X_test_pca)
final_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(final_df)
# Accuracy
print("Accuracy: ", accuracy(y_test, y_pred) * 100, "%")
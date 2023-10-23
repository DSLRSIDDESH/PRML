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
    def __init__(self, cumvar_threshold=0.95):
        self.cumvar_threshold = cumvar_threshold
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None
        self.n_components = None
        
    def fit(self, X):
        mean_vector = np.mean(X, axis=0)
        cov = covariance_matrix(X, mean_vector)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / eigenvalues.sum()
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        self.n_components = np.argmax(self.cumulative_variance_ratio >= self.cumvar_threshold) + 1
        self.components = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        return np.dot(X, self.components)

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
dataset = pd.read_csv("face.csv")

# Train Test Split
def train_test_split(dataset):
    classes = dataset.iloc[:, -1].unique()
    test, train = pd.DataFrame(), pd.DataFrame()
    for c in classes:
        class_data = dataset[dataset.iloc[:, -1] == c]
        test = pd.concat([test, class_data.iloc[:2]], ignore_index=True)
        train = pd.concat([train, class_data.iloc[2:]], ignore_index=True)
    X_train, X_test = train.iloc[:, :-1].values, test.iloc[:, :-1].values
    y_train, y_test = train.iloc[:, -1].values, test.iloc[:, -1].values
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(dataset)

# Principal Component Analysis
print("Running PCA...")
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# KNN Classifier and Accuracy
knn = KNN(5)
knn.fit(X_train_pca, y_train.reshape(-1, 1))
# Predict
y_pred = knn.predict(X_test_pca)
final_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(final_df)
print("Accuracy:", accuracy(y_test, y_pred) * 100, "%")
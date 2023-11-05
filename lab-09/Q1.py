#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd

print('Running...')


# ### Required Functions

# In[2]:


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


# In[3]:


def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2)**2))


# ### Class LDA

# In[4]:


class LDA():
    def __init__(self, n_components = 0):
        self.n_components = n_components
        self.sw = None
        self.sb = None
        self.classes = None

    def mean_vector(self, x):
        return np.mean(x, axis=0)

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        classes = np.unique(y_train)
        self.wc_scatter_matrices = []
        self.mean_vectors = []

        # Calculating mean vectors for each class
        for c in classes:
            class_data_set = X_train[y_train == c]
            m_v = self.mean_vector(class_data_set)
            self.mean_vectors.append(m_v)
        
        # Calculating within class scatter matrices
        self.sw = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i in range(len(X_train)):
            row = X_train[i].reshape(X_train.shape[1], 1)
            m = self.mean_vector(X_train[y_train == y_train[i]])
            self.sw += np.dot((row - m), (row - m).T)

        # Calculating between class scatter matrices
        self.sb = np.zeros((X_train.shape[1], X_train.shape[1]))
        m_v = self.mean_vector(X_train)
        for i in range(len(X_train)):
            row = X_train[i].reshape(X_train.shape[1], 1)
            m_i = self.mean_vector(X_train[y_train == y_train[i]])
            N_i = len(X_train[y_train == y_train[i]])
            self.sb += N_i * np.dot((m_i - m), (m_i - m).T)
        
        # Calculating linear discriminants
        self.linear_discriminants = np.dot(np.linalg.inv(self.sw), self.sb)
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.linear_discriminants)
        self.eig_pairs = [(np.abs(self.eigen_values[i]), self.eigen_vectors[:,i]) for i in range(len(self.eigen_values))]
        self.eig_pairs = sorted(self.eig_pairs, key=lambda k: k[0], reverse=True)

        self.selected_eigen_vectors = []
        for i in range(self.n_components):
            self.selected_eigen_vectors.append(self.eig_pairs[i][1].reshape(X_train.shape[1],1))
        self.final_matrix = np.hstack(self.selected_eigen_vectors)
    
    def transform(self, X_train):
        return np.dot(X_train, self.final_matrix)


# ### Class KNN Classifier

# In[5]:


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


# ### Load Data

# In[6]:


data = pd.read_csv("gender.csv")
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.rename(columns = {'Unnamed: 1': 'class'}, inplace = True)


# ### Train Test Split

# In[8]:


def train_test_split(dataset):
    classes = dataset['class'].unique()
    test, train = pd.DataFrame(), pd.DataFrame()
    for c in classes:
        class_data = dataset[dataset['class'] == c]
        test = pd.concat([test, class_data.iloc[:10]], ignore_index=True)
        train = pd.concat([train, class_data.iloc[10:]], ignore_index=True)
    X_train, X_test = train.iloc[:, 1:].values, test.iloc[:, 1:].values
    y_train, y_test = train.iloc[:, 0].values, test.iloc[:, 0].values
    return X_train, X_test, y_train, y_test


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(data)


# ### Linear Discriminant Analysis

# In[10]:


lda = LDA(n_components=1)
lda.fit(X_train, y_train)


# In[11]:


X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)


# ### KNN Classifier

# In[12]:


knn = KNN(k = 5)
knn.fit(X_train_lda, y_train)


# In[13]:


y_pred = knn.predict(X_test_lda)
final_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(final_df)


# In[14]:


print("Accuracy: ", accuracy(y_test, y_pred) * 100, "%")



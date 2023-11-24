# CS21B2019
# DEVARAKONDA SLR SIDDESH
# ### Import Libraries

# In[89]:


import numpy as np
import matplotlib.pyplot as plt


# In[90]:


def accuracy(y_test, y_pred):
    return np.sum(y_pred == y_test) / len(y_test)


# In[91]:


def decision_boundary(epoch, X, y, weights, bias):
        plt.figure()
        plt.rcParams['figure.figsize'] = [4, 3.2]
        plt.scatter(X[:, 0], X[:, 1], c = y)
        x1 = np.linspace(-3.2, 3.2, 100)
        x2 = -(weights[0] * x1 + bias) / weights[1]
        plt.plot(x1, x2, 'r')
        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.title("Interation: " + str(epoch))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show(block = False)
        plt.pause(0.1)
        plt.close()


# ### Perceptron Implementation

# In[92]:


class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.converged = False
        self.iterations = 0

    def activation(self, z):
        return 1 if z >= 0 else 0

    def forward_prop(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def backward_prop(self, X, y, y_hat):
        self.weights += self.learning_rate * (y - y_hat) * X
        self.bias += self.learning_rate * (y - y_hat)
    
    def fit(self, X, y):
        self.n_features = X.shape[1]

        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for epoch in range(1, self.epochs + 1):
            misClassified = 0
            for i in range(len(X)):
                y_hat = self.forward_prop(X[i])
                if y_hat != y[i]:
                    misClassified += 1
                    self.backward_prop(X[i], y[i], y_hat)
            if misClassified == 0:
                self.converged = True
                self.iterations = epoch - 1
                break
            print(f"Iteration: {epoch} | Weights: {self.weights} | Bias: {self.bias}")
            decision_boundary(epoch, X, y, self.weights, self.bias)
    
    def predict(self, X):
        y_hat = []
        for i in range(len(X)):
            y_hat.append(self.forward_prop(X[i]))
        return np.array(y_hat)
    
    def get_convergence(self):
        return self.converged, self.iterations
    
    def get_params(self):
        return self.weights, self.bias


# ### SVM Implementation

# In[93]:


class SVM:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, z):
        return 1 if z >= 0 else 0
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for epoch in range(1, self.epochs + 1):
            for i in range(len(X)):
                if y[i] * (np.dot(self.weights, X[i]) + self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * 1 / epoch * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * 1 / epoch * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]
            print(f"Iteration: {epoch} | Weights: {self.weights} | Bias: {self.bias}")
            decision_boundary(epoch, X, y, self.weights, self.bias)
    
    def predict(self, X):
        y_hat = []
        for i in range(len(X)):
            z = np.dot(X[i], self.weights) + self.bias
            y_hat.append(self.activation(z))
        return np.array(y_hat)


# ### Load Data

# In[94]:


# X and Y for AND gate
X = np.array([[2, 2], [-1, -3], [-1, 2], [0, -1], [1, 3], [-1, -2], [1, -2], [-1, -1]])
Y = np.array([1, 0, 1, 0, 1, 0, 0, 1])


# ### Training (Perceptron)

# In[95]:


p1 = Perceptron(learning_rate=0.01)
p2 = Perceptron(learning_rate=0.5)


# In[96]:


p1.fit(X, Y)


# In[97]:


p2.fit(X, Y)


# ### Convergence (Perceptron)

# In[98]:


print("Iterations required to converge the perception algorithm : ", p1.iterations)
print("Iterations required to converge the perception algorithm : ", p2.iterations)


# ### Prediction (Perceptron)

# In[99]:


y_pred_p1 = p1.predict(X)
y_pred_p2 = p2.predict(X)
print("Actual values :", Y)
print("\nPredicted values (learning rate = 0.01):", y_pred_p1)
print("Predicted values (learning rate = 0.5)   :", y_pred_p2)


# ### Accuracy (Perceptron)

# In[100]:


acc_p1 = accuracy(Y, y_pred_p1)
print("Accuracy (learning rate = 0.01):", acc_p1 * 100, "%")
acc_p2 = accuracy(Y, y_pred_p2)
print("Accuracy (learning rate = 0.5) :", acc_p2 * 100, "%")


# ### Training (SVM)

# In[101]:


svm = SVM(learning_rate=0.01, epochs=100)
svm.fit(X, Y)


# ### Prediction (SVM)

# In[102]:


y_pred_svm = svm.predict(X)
print("Predicted values:", y_pred_svm)
print("Actual values   :", Y)


# ### Accuracy (SVM)

# In[103]:


acc = accuracy(Y, y_pred_svm)
print("Accuracy:", acc * 100, "%")


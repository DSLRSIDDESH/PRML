# CS21B2019
# DEVARAKONDA SLR SIDDESH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

image_dir = ""

X = []  
y = []  

for image_file in os.listdir(image_dir):
    if image_file.startswith("poly"):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if int(image_file[4]) <= 7:
            label = 1  
        else:
            label = 2  
        x1 = np.mean(image)
        x2 = np.var(image)
        X.append([x1, x2])
        y.append(label)

X = np.array(X)
y = np.array(y)

w = np.zeros(X.shape[1])
b = 0

def perceptron(x):
    return 1 if np.dot(w, x) + b > 0 else 2

learning_rate = 0.01
num_iterations = 1000
for _ in range(num_iterations):
    for i in range(len(X)):
        prediction = perceptron(X[i])
        if prediction != y[i]:
            if prediction == 1:
                w -= learning_rate * X[i]
                b -= learning_rate
            else:
                w += learning_rate * X[i]
                b += learning_rate

def plot_decision_boundary(X, y, w, b):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    Z = np.array([perceptron([x1, x2]) for x1, x2 in np.c_[xx1.ravel(), xx2.ravel()]])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='o')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], label='Class 2', marker='x')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

plot_decision_boundary(X, y, w, b)

image_dir_svm = "C:\\Users\\vardh\\Desktop\\5th Sem\\PRML\\Lab10\\Images"
class1_images_svm = range(1, 8)
class2_images_svm = range(8, 15)

X_svm = []  
y_svm = []  

for i in class1_images_svm:
    image_path = os.path.join(image_dir_svm, f"poly{i}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x1 = np.mean(image)
    x2 = np.var(image)
    X_svm.append([x1, x2])
    y_svm.append(1)

for i in class2_images_svm:
    image_path = os.path.join(image_dir_svm, f"poly{i}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x1 = np.mean(image)
    x2 = np.var(image)
    X_svm.append([x1, x2])
    y_svm.append(2)

X_svm = np.array(X_svm)
y_svm = np.array(y_svm)

X_svm, y_svm = shuffle(X_svm, y_svm, random_state=42)

learning_rate_svm = 0.01
C_svm = 1.0  

w_svm = np.zeros(X_svm.shape[1])
b_svm = 0

def hinge_loss_svm(w, b, X, y):
    loss = 1 - y * (np.dot(X, w) + b)
    return np.maximum(0, loss)

num_iterations_svm = 1000
for _ in range(num_iterations_svm):
    for i in range(len(X_svm)):
        if y_svm[i] * (np.dot(X_svm[i], w_svm) + b_svm) >= 1:
            w_svm -= learning_rate_svm * (2 * 1 / num_iterations_svm * w_svm)
        else:
            w_svm -= learning_rate_svm * (2 * 1 / num_iterations_svm * w_svm - np.dot(X_svm[i], y_svm[i]))
            b_svm -= learning_rate_svm * y_svm[i]

def plot_svm_decision_boundary(X, y, w, b):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    Z_svm = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z_svm = Z_svm.reshape(xx.shape)
    plt.contourf(xx, yy, Z_svm, levels=[-1, 0, 1], colors=('blue', 'white', 'red'), alpha=0.4)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], c='red', label='Class 2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.show()

plot_svm_decision_boundary(X_svm, y_svm, w_svm, b_svm)
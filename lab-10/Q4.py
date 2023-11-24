# CS21B2019
# DEVARAKONDA SLR SIDDESH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.read_csv('iris.csv')
modified_df = {'Petal Length': iris_df['PetalLengthCm'], 'Sepal Width': iris_df['SepalWidthCm'], 'Species': iris_df['Species']}
modified_df = pd.DataFrame(data=modified_df)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
modified_df['Species'] = label_encoder.fit_transform(modified_df['Species'])
features = modified_df.drop('Species', axis=1).to_numpy()
labels = np.array(modified_df['Species'])

features = features[labels != 0]  
labels = labels[labels != 0]  

n_classes = 3  
n_features = features.shape[1]
weights_perceptron = np.zeros((n_classes, n_features))
bias_perceptron = np.zeros(n_classes)
learning_rate_perceptron = 0.01
epochs_perceptron = 100

for epoch_perceptron in range(epochs_perceptron):
    for feature, target_perceptron in zip(features, labels):
        for class_perceptron in range(n_classes):
            update_perceptron = learning_rate_perceptron * (int(target_perceptron == class_perceptron) - np.dot(feature, weights_perceptron[class_perceptron]) - bias_perceptron[class_perceptron])
            weights_perceptron[class_perceptron] += update_perceptron * feature
            bias_perceptron[class_perceptron] += update_perceptron

plt.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1], label="Versicolor")
plt.scatter(features[labels == 2][:, 0], features[labels == 2][:, 1], label="Virginica")
plt.scatter(features[labels == 0][:, 0], features[labels == 0][:, 1], label="Setosa")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Length")

x_min_perceptron, x_max_perceptron = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min_perceptron, y_max_perceptron = features[:, 1].min() - 1, features[:, 1].max() + 1
xx_perceptron, yy_perceptron = np.meshgrid(np.arange(x_min_perceptron, x_max_perceptron, 0.01), np.arange(y_min_perceptron, y_max_perceptron, 0.01))

for class_perceptron in range(n_classes):
    Z_perceptron = np.dot(np.c_[xx_perceptron.ravel(), yy_perceptron.ravel()], weights_perceptron[class_perceptron]) + bias_perceptron[class_perceptron]
    Z_perceptron = Z_perceptron.reshape(xx_perceptron.shape)
    plt.contourf(xx_perceptron, yy_perceptron, Z_perceptron, alpha=0.3)

plt.legend()
plt.title("Multi-Class Perceptron Decision Boundaries")
plt.show()

features_svm = modified_df.drop('Species', axis=1).to_numpy()
labels_svm = np.array(modified_df['Species'])

features_svm = features_svm[labels_svm != 2]
labels_svm = labels_svm[labels_svm != 2]
labels_svm[labels_svm == 0] = -1  

weights_svm = np.zeros(features_svm.shape[1])
learning_rate_svm = 0.01
epochs_svm = 1000

for epoch_svm in range(epochs_svm):
    for i_svm, feature_svm in enumerate(features_svm):
        if labels_svm[i_svm] * np.dot(feature_svm, weights_svm) <= 1:
            weights_svm = weights_svm + learning_rate_svm * (labels_svm[i_svm] * feature_svm - 2 * weights_svm)

plt.figure(figsize=(8, 6))
plt.scatter(features_svm[:, 0], features_svm[:, 1], c=labels_svm, cmap=plt.cm.Paired, marker='o', edgecolors='k')

ax_svm = plt.gca()
xlim_svm = ax_svm.get_xlim()
ylim_svm = ax_svm.get_ylim()

xx_svm, yy_svm = np.meshgrid(np.linspace(xlim_svm[0], xlim_svm[1], 50),
                             np.linspace(ylim_svm[0], ylim_svm[1], 50))
Z_svm = np.dot(np.c_[xx_svm.ravel(), yy_svm.ravel()], weights_svm)
Z_svm = Z_svm.reshape(xx_svm.shape)

plt.contour(xx_svm, yy_svm, Z_svm, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.title('SVM Decision Boundary')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')

plt.show()
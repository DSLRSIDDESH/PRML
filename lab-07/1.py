# Find and plot the decision boundary between class ω1 and ω2.

import numpy as np

w1 = np.array([[1,6],[3,4],[3,8],[5,6]])
w2 = np.array([[3,0],[1,-2],[3,-4],[5,-2]])

pw1, pw2 = 0.5, 0.5

# Calculate mean of each class
mean_w1 = np.mean(w1, axis=0)
mean_w2 = np.mean(w2, axis=0)


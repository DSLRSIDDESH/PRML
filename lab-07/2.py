# Find and plot the decision boundary between class ω1 and ω2.

import numpy as np


# ω1 = [1,-1; 2,-5; 3,-6; 4,-10; 5,-12; 6,-15]
# ω2 = [-1,1; -2,5; -3,6; -4,10, -5,12; -6, 15]

w1 = np.array([[1,-1],[2,-5],[3,-6],[4,-10],[5,-12],[6,-15]])
w2 = np.array([[-1,1],[-2,5],[-3,6],[-4,10],[-5,12],[-6,15]])

pw1, pw2 = 0.3, 0.7

# Calculate mean of each class
mean_w1 = np.mean(w1, axis=0)
mean_w2 = np.mean(w2, axis=0)


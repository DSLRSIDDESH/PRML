import numpy as np

def quadratic_form(matrix, vector):
    return np.sqrt(np.dot(vector, np.dot(matrix, vector.T)))

matrix = [
            [1, 0.135, 0.195, 0.137, 0.157],
            [0.135, 1, 0.2, 0.309, 0.143],
            [0.195, 0.2, 1, 0.157, 0.122],
            [0.137, 0.309, 0.157, 1, 0.195],
            [0.157, 0.143, 0.122, 0.195, 1]
        ]

matrix = np.array(matrix)
hq_ht_diff_T = np.array([0.5, 0.5, -0.5, -0.25, -0.25])

distance = quadratic_form(matrix, hq_ht_diff_T)
print(f'Quadratic form distance: {round(distance, 4)}')
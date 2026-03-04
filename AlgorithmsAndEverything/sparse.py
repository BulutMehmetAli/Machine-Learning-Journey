import numpy as np
from scipy.sparse import csr_matrix

# Define a dense matrix
dense_matrix = np.array([[0, 0, 3], [0, 5, 0], [0, 0, 0]])
# Convert the dense matrix to the sparse matrix
sparse_matrix = csr_matrix(dense_matrix)
# Print the sparse matrix
print(sparse_matrix)
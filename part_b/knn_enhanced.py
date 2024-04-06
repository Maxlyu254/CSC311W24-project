from utils import *
import numpy as np

if __name__ == "__main__":
  sparse_matrix = load_train_sparse("../data").toarray()
  val_data = load_valid_csv("../data")
  test_data = load_public_test_csv("../data")

  print("Sparse matrix:")
  print(sparse_matrix)
  print("Shape of sparse matrix:")
  print(sparse_matrix.shape)
    
  # Assuming 'sparse_matrix' is your sparse matrix
  # Replace NaN with zero for counting non-zero entries
  sparse_matrix[sparse_matrix >= 0] = 1
  sparse_matrix[np.isnan(sparse_matrix)] = 0
  print(np.mean(sparse_matrix))
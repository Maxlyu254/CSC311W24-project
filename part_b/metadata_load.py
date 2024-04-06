from utils import *
from datetime import datetime
import numpy as np
import csv
import os

def load_csv_to_matrix(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        for row in csv_reader:
            data.append(row)
    return np.array(data)

def transform_metadata(metadata):

  row_num = metadata.shape[0]
  transformed_meta = [None] * row_num
  date_origin = datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
  for row in metadata:
    # identify row id
    row_id = int(row[0])
    # identify gender
    is_female = int(int(row[1]) == 1)
    is_male = int(int(row[1]) == 2)
    is_unspecified = int(int(row[1]) == 0)
    # identify age
    if not row[2] == '':
      birth = datetime.strptime(row[2][0 : 19], "%Y-%m-%d %H:%M:%S")
      age = date_origin.year - birth.year
    else:
      age = -1
    # identify premium student
    if row[3] == '1.0':
      premium = 1
    elif row[3] == '0.0':
      premium = 0
    else:
      premium = 0.5
    tsfmd_row = [is_unspecified, is_female, is_male, age, premium]
    transformed_meta[row_id] = tsfmd_row

  # calculate and apply average age
  avg_age = sum(row[3] for row in transformed_meta if row[3] >= 0) // row_num
  for row in transformed_meta:
    if row[3] < 0:
      row[3] = avg_age
     
  transformed_meta = np.matrix(transformed_meta)
  return transformed_meta
   

if __name__ == "__main__":
  root_dir = "../data"
  sparse_matrix = load_train_sparse(root_dir).toarray()
  val_data = load_valid_csv(root_dir)
  test_data = load_public_test_csv(root_dir)

  student_metadata = load_csv_to_matrix(os.path.join(root_dir, "student_meta.csv"))

  print("Sparse matrix:")
  print(sparse_matrix)
  print("Shape of sparse matrix:")
  print(sparse_matrix.shape)

  print("student metadata:")
  print(student_metadata)
  print("Shape of metadata:")
  print(student_metadata.shape)

  tsfmd_meta = transform_metadata(student_metadata)
  print("transformed metadata: ")
  print(tsfmd_meta)

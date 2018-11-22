import numpy as np
from sklearn.model_selection import train_test_split

def main():
  # Load data
  data     = np.genfromtxt('datasets/9_gold_exp_advantage_matrix.csv', delimiter=',', skip_header=1)
  
  # Split data into training and test set and save to two separate csv files
  train_data, test_data  = train_test_split(data, test_size=0.2)
  np.savetxt("datasets/gpmxpm_train_data.csv", train_data, delimiter=",",fmt='%i')
  np.savetxt("datasets/gpmxpm_test_data.csv", test_data, delimiter=",",fmt='%i')
  
  
main()
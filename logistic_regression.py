import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_regression(train_data, train_labels):
  # Add column of 1's to train data
  ones_column = np.ones((train_data.shape[0],1))
  train_data  = np.concatenate((ones_column, train_data), axis = 1)
  
  model = LogisticRegression()
  model.fit(train_data, train_labels)
  return model


def predictor(model, test_data):
  # Add column of 1's to test data
  ones_column = np.ones((test_data.shape[0],1))
  test_data  = np.concatenate((ones_column, test_data), axis = 1)
  
  # Predict labels and weights
  predicted_labels, weights = model.predict(test_data), model.coef_[0]
  return predicted_labels, weights
  


def classification_accuracy(pred_labels, actual_labels):
  pred_len = len(pred_labels)
  actu_len = len(actual_labels)
  
  if pred_len != actu_len:
    print('Predicted and actual labels should have the same length')
    return -1
  

  result = [int(pred_labels[x] == actual_labels[x]) for x in range(pred_len)]
  accuracy = np.mean(result)  
  return accuracy
  



def main():
  #data   = np.genfromtxt('pro_team_composition.csv', delimiter=',', skip_header=1, usecols=(range(10)))
  #labels = np.genfromtxt('pro_team_composition.csv', delimiter=',', dtype=str, skip_header=1, usecols=(10))
  
  # Load data and labels, and split data into 80/20 for training and testing
  data   = np.genfromtxt('data_after_clustering.csv', delimiter=',', usecols=(range(10)))
  labels = np.genfromtxt('data_after_clustering.csv', delimiter=',', dtype=str, usecols=(10))  

  train_data, test_data     = train_test_split(data, test_size=0.2)
  train_labels, test_labels = train_test_split(labels, test_size=0.2)
  
  model = logistic_regression(train_data, train_labels)
  predicted_labels, _ = predictor(model, test_data) 
  
  accuracy = classification_accuracy(predicted_labels, test_labels)
  print(accuracy)
  
  
main()

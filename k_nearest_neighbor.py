import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def cross_validation(model, train_data, train_labels):
  accuracy_rates = []
  cv = KFold(n_splits=5, shuffle=True, random_state=42)
  for train, test in cv.split(train_data):
    # Split data into train and test data/labels
    trainX, testX = train_data[train], train_data[test]
    trainY, testY = train_labels[train], train_labels[test]
    
    # Fit the model on the train data
    model.fit(trainX, trainY)
    #prediction = model.predict(testX)
    
    # get the classification error and save to error rate list
    accuracy_rates.append(model.score(testX, testY))
  
  return np.mean(accuracy_rates)

def best_k(k_array, train_data, train_labels):
  # Compute train accuracy for all k in k_array
  k_accuracies = []
  for k in k_array:
    print("Computing accuracy for k=" + str(k))
    kNN = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_validation(kNN, train_data, train_labels)
    k_accuracies.append(accuracy)
    print("Accuracy = " + str(accuracy))
    
  # get K with best train accuracy
  best_accuracy = k_accuracies.index(max(k_accuracies))
  
  # Plot the accuracies for the different k values
  plt.figure(figsize=(12,7))
  plt.plot(k_array, k_accuracies, '--bo', color='green')
  plt.rc('axes',titlesize=10)
  plt.xticks(k_array)
  plt.xlabel("k-parameter")
  plt.ylabel("Accuracy score")
  plt.title("Accuracy score for different kNN-classifiers")
  plt.show()
  
  return k_array[best_accuracy]
  

def knn_probability(k, train_data, train_labels):
  # Define and train model with train data
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(train_data, train_labels)
  
  return model


def predictor(model, test_data):
  # Predict probabilities on test data
  #predicted_probs = model.predict(test_data)
  predicted_probs = model.predict_proba(test_data)
  
  return predicted_probs
  


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
  # Load data and labels, and split data into 80/20 for training and testing
  train_data = np.genfromtxt('data-retriever/datasets/gpmxpm_train_data.csv', delimiter=',')
  test_data  = np.genfromtxt('data-retriever/datasets/gpmxpm_test_data.csv', delimiter=',')
  
  trainX, trainY = preprocessing.scale(train_data[:,1:]), train_data[:,0]
  testX, testY   = preprocessing.scale(test_data[:,1:]), test_data[:,0]

  np.set_printoptions(suppress=True)


  

  #k_array = [i for i in range(1000) if i % 2 != 0]
  k_array = [i for i in range(8) if i % 2 != 0]
  print(best_k(k_array, trainX, trainY))

  
  
  """
  for i in range(1000):
    data1 = np.concatenate(([1], features[1000+i,:]))
    print(calculate_win_probability(data1,weights))
    #print(calculate_win_probability(data2,weights))
  """
  
main()
import itertools
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
  plt.figure(figsize=(10,5))
  plt.plot(k_array, k_accuracies, '--bo', color='green')
  plt.rc('axes',titlesize=10)
  plt.xticks(np.arange(0,201,10))
  plt.grid()
  plt.xlabel("k-parameter")
  plt.ylabel("Accuracy score (%)")
  plt.title("Accuracy score for different kNN-classifiers")
  plt.show()
  
  return k_array[best_accuracy]
  

def get_knn_model(k, train_data, train_labels):
  # Define and train model with train data
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(train_data, train_labels)
  
  return model


def predictor(model, test_data, isProbabilistic):
  # Predict probabilities on test data
  if (isProbabilistic):
    return model.predict_proba(test_data)
  
  # Else, return predicted labels for test data
  else:
    return model.predict(test_data)


def classification_accuracy(pred_labels, actual_labels):
  pred_len = len(pred_labels)
  actu_len = len(actual_labels)
  
  if pred_len != actu_len:
    print('Predicted and actual labels should have the same length')
    return -1
  
  result = [int(pred_labels[x] == actual_labels[x]) for x in range(pred_len)]
  accuracy = np.mean(result)  
  return accuracy
  
def radiant_win_probability(model, data):
  preds = predictor(model, data, True)
  radiant_probability = [arr[1] for arr in preds]
  return preds[:,1].flatten()
  #return radiant_probability
  

def main():  
  # k_accuracies = [0.618,0.642,0.654,0.661,0.665,0.668,0.671,0.673,0.674,0.676,0.677,0.678,0.679,0.679,0.680,0.680,0.681,0.682,0.682,0.682,0.683,0.683,0.683,0.684,0.684,0.685,0.684,0.685,0.685,0.686,0.685,0.686,0.686,0.686,0.686,0.686,0.687,0.687,0.687,0.687,0.687,0.688,0.688,0.688,0.688,0.688,0.688,0.688,0.688,0.688,0.689,0.688,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.689,0.690,0.690,0.690,0.689,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.690,0.691,0.691,0.691,0.691,0.691,0.691,0.690,0.691,0.690,0.690,0.690,0.691]
  
  # Load training and test data
  train_data = np.genfromtxt('data-retriever/datasets/gpmxpm_train_data.csv', delimiter=',')
  test_data  = np.genfromtxt('data-retriever/datasets/11_knn_two_samples.csv', delimiter=',')
  
  # Split data into features and labels
  trainX, trainY = preprocessing.scale(train_data[:,1:]), train_data[:,0]
  testX, testY   = preprocessing.scale(test_data[:,1:]), test_data[:,0]

  
  k_array = [i for i in range(202) if i % 2 == 1]
  print(best_k(k_array, trainX, trainY))
  
  
  # Train the kNN model and predict two random samples
  k = 101
  model = get_knn_model(k, trainX, trainY)
  sample_preds = radiant_win_probability(model,testX)
  first_game   = sample_preds[:43]
  second_game  = sample_preds[43:]
  
  minutes = [0,5,10,15,20,25,30,35,40]
  values1 = [first_game[e]*100 for e in minutes]
  values2 = [second_game[e]*100 for e in minutes]
  
  plt.figure(figsize=(14,8))
  plt.plot(range(43), first_game*100, color='#ffa31a')
  plt.scatter(minutes,values1, label='Gold/Experience difference', color='grey')
  plt.axhline(50, color='black')
  plt.title('Live analysis of game that Radiant team loses', fontsize=15)
  plt.xlabel('Timestamp (min)', fontsize=15)
  plt.ylabel('Win probability (%)', fontsize=15)
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(14,8))
  plt.plot(range(44), second_game*100, color='#ffa31a')
  plt.scatter(minutes,values2, label='Gold/Experience difference', color='grey')
  plt.axhline(50, color='black')
  plt.title('Live analysis of game that Radiant team wins', fontsize=15)
  plt.xlabel('Timestamp (min)', fontsize=15)
  plt.ylabel('Win probability (%)', fontsize=15)
  plt.legend()
  plt.show()
  
  
  
  # Train the kNN model and predict on the test data
  preds = predictor(model, testX, False)

  # Get the list of correct and incorrect predictions
  correctPredictions   = preds[np.where(preds == testY)[0]]
  incorrectPredictions = preds[np.where(preds != testY)[0]]

  # Calculate the classification-cases the model produced, where positive = win  
  truePositives  = len(correctPredictions[np.where(correctPredictions == 1)[0]])
  trueNegatives  = len(correctPredictions[np.where(correctPredictions == 0)[0]])
  falsePositives = len(correctPredictions[np.where(incorrectPredictions == 1)[0]])
  falseNegatives = len(correctPredictions[np.where(incorrectPredictions == 0)[0]])
  print("TP: ", truePositives)
  print("TN: ", trueNegatives)
  print("FP: ", falsePositives)
  print("FN: ", falseNegatives)
   
  # Calculate and print the test accuracy
  np.set_printoptions(suppress=True)
  print("Accuracy: ", (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives))
  
  # Predict win probability for 10 samples selected from test data
  test_samples = preprocessing.scale(
      np.array([
                 [1  ,  258   ,  338  ], # radiant win
                 [1  , -1852  , -160  ], # radiant win
                 [5  , -4710  , -5938 ], # dire win
                 [5  ,  4     ,  375  ], # radiant win
                 [15 , -15171 ,  4727 ], # radiant win
                 [25 ,  11074 ,  15811], # radiant win
                 [35 , -23832 , -33072], # dire win
                 [45 ,  38862 ,  594  ], # radiant win
                 [50 ,  8205  , -3212 ], # dire win
                 [75 ,  499   ,  235  ], # radiant win
                 [90 , -40707 , -146  ], # dire win
                 [119,  -38876,  166  ]  # dire win
               ]) + 0.)                  # (convert from int to float list)
  
  pred_samples = predictor(model, test_samples, True)
  print(pred_samples)
  
  

main()
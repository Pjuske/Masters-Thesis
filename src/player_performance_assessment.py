import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def predict_weights(X, Y):
  reg = LinearRegression().fit(X,Y)
  return reg.coef_, reg.intercept_

def perform_linear_regression(train_data):
  features = train_data.shape[1]
  
  # For each feature f, compute the weights for that linear regression problem
  feature_weights = np.empty([15,15])
  for i in range(features):
    # Define the training data to be all features except that specific feature f
    X = np.delete(train_data, (i), axis=1)
    Y = train_data[:,i]
    
    weights, intercept   = predict_weights(X, Y)
    feature_weights[i,:] = [intercept] + list(weights)
    
  return feature_weights


def display_assessment_of_player(model, vector_stats, labels):
  preds = []
  for i in range(len(vector_stats)):
    intercept = model[i,0]
    weights   = model[i,1:]
    value =  intercept + np.dot(weights, np.delete(vector_stats, i))
    preds.append(np.around(value,2))

  # For each feature, display the expected and actual result
  for i in range(len(labels)):
    expected = 'Expected '+ labels[i] + ':'
    actual   = '\nActual   '  + labels[i] + ':'
    print(expected, preds[i], actual, vector_stats[i])

  return preds

def compute_residuals_and_responses(train_datasets, test_datasets):
  features        = 15
  roles           = 6
  wanted_features = [index for index in range(17) if (index != 0 and index != 12)]
  
  all_residuals = []
  all_responses = []
  
  for feat in range(features):
    residuals = np.array([])
    responses = np.array([])
        
    for role in range(roles):
      # Train the model
      train_data = train_datasets[role][:, wanted_features]
      X          = np.delete(train_data, (feat), axis=1)
      Y          = train_data[:,feat]
      model = LinearRegression().fit(X,Y)
    
      # Remove the output feature from the test data
      test_data = np.delete(test_datasets[role], (feat), axis=1)
      
      # Get the actual and predicted output and compute the residuals      
      actual_output    = test_datasets[role][:,feat]
      predicted_output = model.predict(test_data)
    
      rolespecific_residuals = actual_output - predicted_output
      residuals = np.append(residuals, rolespecific_residuals)
      responses = np.append(responses, predicted_output)
      
    all_residuals.append(residuals)
    all_responses.append(responses)
      
  return all_residuals, all_responses

 
def plot_residual_distributions(residuals, feature):
  min_value = int(np.min(residuals))
  max_value = int(np.max(residuals))
  
  plt.hist(residuals, bins=compute_bin_range(min_value, max_value), color='#cc99ff')
  plt.title('Distribution of residuals for feature \'' + feature + '\'', fontsize=14)
  plt.xlabel(feature)
  plt.ylabel('Frequency')
  plt.show()

def plot_residuals_vs_predicted_responses(residuals, responses, feature): 
  plt.scatter(responses, residuals, color='#cc99ff')
  plt.title('Residuals vs predicted responses for \'' + feature + '\'', fontsize=14)
  plt.xlabel('responses')
  plt.ylabel('residuals')
  plt.show()


def get_wins_only(train_data, result_feature):
  return train_data[np.where(train_data[:,result_feature] == 1)[0]]

def get_loss_only(train_data, result_feature):
  return train_data[np.where(train_data[:,result_feature] == 0)[0]]
  
def computer_r2_score(train_datasets, test_datasets):
  r2_score_matrix = []
  wanted_features   = [index for index in range(17) if (index != 0 and index != 12)]
  for (train_dataset, test_dataset) in zip(train_datasets, test_datasets):
    train_dataset = get_wins_only(train_dataset, 12)[:,wanted_features]
    test_dataset = test_dataset[:,wanted_features]
    r2_score_vector = []
    for feature in range(15):
      
      test_data = np.delete(test_dataset, (feature), axis=1)
      X = np.delete(train_dataset, (feature), axis=1)
      Y = train_dataset[:,feature]
      model = LinearRegression().fit(X,Y)
      y_True = test_dataset[:,feature]
      y_Pred = model.predict(test_data)
      
      r2_score_vector.append(np.around(r2_score(y_True, y_Pred),2))
      
    r2_score_matrix.append(r2_score_vector)
    
  for elem in r2_score_matrix:
    print(elem)

def compute_bin_range(min_value, max_value):
  # Compute step size for bins in histogram
  if min_value < 1000 and max_value < 1000:
    return range(min_value, max_value, 1)
  elif min_value < 10000 and max_value < 10000:
    return range(min_value, max_value, 10)
  else:
    return range(min_value, max_value, 1000)


def main():
  # Load training data for each role
  wanted_features   = [index for index in range(17) if (index != 0 and index != 12)]
  train_datasets   = [np.loadtxt('data-retriever/datasets/regression/regression_train_data_1.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_train_data_2.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_train_data_3.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_train_data_4.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_train_data_5.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_train_data_6.csv', delimiter=',', skiprows=1)]
  
  # Load test data for each role
  test_datasets    = [np.loadtxt('data-retriever/datasets/regression/regression_test_data_1.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_2.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_3.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_4.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_5.csv', delimiter=',', skiprows=1),
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_6.csv', delimiter=',', skiprows=1)]
  
  
  test_datasets_win = [get_wins_only(elem, 12) for elem in test_datasets]
  
  test_datasets_loss = [get_loss_only(elem, 12) for elem in test_datasets]
  
  #R2 score for winning games
  computer_r2_score(train_datasets, test_datasets_win)
  
  #R2 score for losing games
  computer_r2_score(train_datasets, test_datasets_loss)

  
  test_datasets = [elem[:, wanted_features] for elem in test_datasets]
  labels = ['kills','deaths','assists','hero healing','last hits','gpm','xpm',
            'tower damage','hero damage','observers placed','sentries placed',
            'team score','enemy team score','team gold advantage','team exp advantage']
  
  # Get the classifier model for each role
  roles = 6
  weights = []
  for i in range(roles):
    # Get only data points of players who won their match
    win_column = 12
    train_data = get_wins_only(train_datasets[i], win_column)[:, wanted_features]

    # Perform linear regression
    weight = perform_linear_regression(train_data)
    weights.append(weight)
  
  # Compute residual and predicted responses for all features
  all_residuals, all_responses = compute_residuals_and_responses(train_datasets, test_datasets)
  
  # Plot the residual distribution for each feature
  for feat in range(len(wanted_features)):
    feat_residuals = all_residuals[feat]
    plot_residual_distributions(feat_residuals, labels[feat])
  
  
  # Plot the standardized residuals versus the standardized predicted responses
  for feat in range(len(wanted_features)):
    feat_residuals = all_residuals[feat]
    responses      = all_responses[feat]
    plot_residuals_vs_predicted_responses(feat_residuals, responses, labels[feat])
  
  
  # Display an assessment of a random player from each role
  for i in range(roles):
    test_data     = test_datasets[i]
    random_player = random.randint(0,len(test_data)-1)
    sample_role = test_data[random_player]
    print('\n----------------------------------------')
    print('Assesment of player', random_player, 'from role', i+1)
    print('----------------------------------------')
    display_assessment_of_player(weights[i], sample_role, labels)

  
main()
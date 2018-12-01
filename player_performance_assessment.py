import random
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def get_wins_only(train_data, result_feature):
  return train_data[np.where(train_data[:,result_feature] == 1)[0]]
  
def compute_bin_range(max_value):
  # Compute step size for bins in histogram
  if max_value < 1000:
    return range(0, max_value, 1)
  elif max_value < 10000:
    return range(0, max_value, 10)
  else:
    return range(0, max_value, 1000)


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
  test_datasets    = [np.loadtxt('data-retriever/datasets/regression/regression_test_data_1.csv', delimiter=',', skiprows=1)[:, wanted_features],
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_2.csv', delimiter=',', skiprows=1)[:, wanted_features],
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_3.csv', delimiter=',', skiprows=1)[:, wanted_features],
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_4.csv', delimiter=',', skiprows=1)[:, wanted_features],
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_5.csv', delimiter=',', skiprows=1)[:, wanted_features],
                      np.loadtxt('data-retriever/datasets/regression/regression_test_data_6.csv', delimiter=',', skiprows=1)[:, wanted_features]]
  
  labels = ['kills','deaths','assists','hero healing','last hits','gpm','xpm',
            'tower damage','hero damage','observers placed','sentries placed',
            'team score','enemy team score','team gold advantage','team exp advantage']

  # Get the classifier model for each role
  roles = 6
  models = []
  for i in range(roles):
    # Get only data points of players who won their match
    win_column = 12
    train_data = get_wins_only(train_datasets[i], win_column)[:, wanted_features]

    # Perform linear regression
    model = perform_linear_regression(train_data)
    models.append(model)
  
  # Display an assessment of a random player from each role
  for i in range(roles):
    test_data     = test_datasets[i]
    random_player = random.randint(0,len(test_data)-1)
    sample_role = test_data[random_player]
    print('\n----------------------------------------')
    print('Assesment of player', random_player, 'from role', i+1)
    print('----------------------------------------')
    display_assessment_of_player(models[i], sample_role, labels)
 
  
  # Plot histogram for each feature to see if they are normal distributed
  """
  sns.set()
  regression_data = np.loadtxt('data-retriever/datasets/10_regression_data.csv', delimiter=',', skiprows=1)[:, wanted_features]
  
  for i in range(len(wanted_features)):
    feature_data = regression_data[:,i]
    #plt.hist(feature_data, bins=range(int(np.min(feature_data)),int(np.max(feature_data))))
    max_value = int(np.max(feature_data))
    
    
    bin_range = compute_bin_range(max_value)
    
    plt.hist(feature_data, bins=bin_range, color='#cc99ff')
    plt.title('Distribution of feature \'' + labels[i] + '\'')
    plt.xlabel(labels[i])
    plt.ylabel('Frequency')
    plt.show()
  """
  
main()
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

def find_outliers(data):
  threshold = 3.5
  
  # Compute sample median for each feature
  sample_median = np.median(data, axis=0)
    
  # Compute difference
  diff = np.sum(data - sample_median, axis=1)
  
  # Compute median of absolute deviations about the median, MAD  
  mad = np.median(np.abs(diff))
  
  # Get modified Z score
  modified_z_scores = 0.6745 * diff / mad
  np.set_printoptions(suppress=True)

  return np.where(np.abs(modified_z_scores) > threshold)[0]

def get_roles_from_ids(data, roles):
  for row in range(len(data)):
    hero_id = data[row, 0]
    data[row, 0] = roles.get(hero_id)
  
  return data

def get_correlation_matrix(features_wanted):
  n_features = features_wanted.shape[1]
  correlation_matrix = np.empty([n_features, n_features])
  
  for i in range(n_features):
    for j in range(n_features):
      # Calculate correlation coefficient for row i, column j and save to matrix
      coefficient = pearsonr(features_wanted[:,i], features_wanted[:,j])[0]
      correlation_matrix[i,j] = np.around(coefficient, 3)

  return correlation_matrix

def main():
  # Define cluster role for each hero id
  roles = {       1:5,   2:4,   3:2,   4:6,   5:2,   6:5,   7:3,   8:5,   9:4, 
           10:5,  11:6,  12:5,  13:4,  14:3,  15:6,  16:3,  17:6,  18:6,  19:4, 
           20:2,  21:4,  22:6,  23:4,  25:6,  26:2,  27:2,  28:3,  29:4, 
           30:1,  31:2,  32:4,  33:4,  34:6,  35:6,  36:4,  37:1,  38:4,  39:6, 
           40:4,  41:6,  42:5,  43:6,  44:6,  45:4,  46:5,  47:6,  48:5,  49:5, 
           50:1,  51:3,  52:4,  53:4,  54:5,  55:4,  56:5,  57:1,  58:4,  59:6, 
           60:4,  61:5,  62:3,  63:6,  64:2,  65:3,  66:1,  67:6,  68:2,  69:4, 
           70:6,  71:3,  72:6,  73:5,  74:6,  75:2,  76:6,  77:5,  78:4,  79:2, 
           80:5,  81:6,  82:5,  83:1,  84:2,  85:3,  86:2,  87:2,  88:3,  89:3, 
           90:1,  91:1,  92:5,  93:6,  94:5,  95:5,  96:4,  97:4,  98:4,  99:4, 
           100:3, 101:3, 102:4, 103:3, 104:4, 105:4, 106:6, 107:3, 108:4, 109:5, 
           110:4, 111:1, 112:1, 113:6, 114:6, 119:2, 120:4}

  # Load data and scale it to mean 0 and variance 1
  data = np.loadtxt('datasets/10_regression_data.csv', skiprows=1, delimiter=',')
  outliers = find_outliers(data)
  print('\nOutliers:\n', outliers)
  
  
  dataWithRoles  = get_roles_from_ids(data, roles)
  wantedFeatures = [index for index in range(17) if (index != 0 and index != 12)]
  wantedData     = dataWithRoles[:,wantedFeatures]
  print('\nCorrelation matrix:\n', get_correlation_matrix(wantedData))
  
  # Split the data set into 6 data sets, one for each role  
  for i in range(1,7):
    roleSpecificData = dataWithRoles[np.where(dataWithRoles[:,0] == i)[0]]
    train_data, test_data = train_test_split(roleSpecificData, test_size=0.1)
    np.savetxt("datasets/regression/regression_train_data_" +str(i) +".csv", train_data, delimiter=",",fmt='%i')
    np.savetxt("datasets/regression/regression_test_data_"  +str(i) +".csv", test_data, delimiter=",",fmt='%i')
    

main()
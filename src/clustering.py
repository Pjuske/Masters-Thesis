import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

def plot_cumulative_variance(eigen_val):
  cumvar = np.cumsum(eigen_val) / np.sum(eigen_val)
  plt.plot(cumvar)
  plt.show()
  
  for i in range(len(cumvar)):
    print('\nCum. variance captured by the ' + str(i+1) + 
          '. PC: ' + str(np.around(cumvar[i],3)))
  
  

def PCA(data):
  # Get eigenvalues (variance) and eigenvectors from covariance matrix
  cov_matrix = np.cov(np.transpose(data))
  eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
  
  # Sort eigenvalues by value to get principal comps with highest eigenvalue
  index = np.flipud(np.argsort(eigen_val))
  eigen_val = eigen_val[index]
  eigen_vec = eigen_vec[:,index]

  return eigen_val, eigen_vec



def clustering():
  pass


def main():
  # Load data and normalize to zero mean and unit variance?
  data      = np.genfromtxt('cluster_data.csv', delimiter=',', skip_header=1)
  norm_data = preprocessing.scale(data)
  
  # Perform PCA and show cumulative variance captured by the PCs
  eigen_val, eigen_vec = PCA(norm_data)
  plot_cumulative_variance(eigen_val)
  
  proj_data = np.dot(norm_data, eigen_vec[:,:2])
  plt.scatter(proj_data[:,0], proj_data[:,1])
  plt.show()
  
  
  # NOTE: kan man ikke bare gange med de 4 PCs?
  # siden kMeans godt kan klare flere dimensioner end 2?
  #clusters = KMeans(5, )
  #clusters.fit(proje)
  
  
  
  
main()
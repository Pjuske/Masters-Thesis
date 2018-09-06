from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def plot3d(data):
  color = ["red", "green"]
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim(min(data[:,1]),max(data[:,1]))
  ax.set_ylim(min(data[:,2]),max(data[:,2]))
  ax.set_zlim(min(data[:,3]),max(data[:,3]))
  
  for i in range(1000): #len(data)):
    radiant_win = int(data[i,0])
    ax.scatter(data[i,1], data[i,2], data[i,3], c=color[radiant_win])
    
    if (i % 5000) == 0:
      print(i)
    
  plt.show()


def PCA(data):
  # Get eigenvalues (variance) and eigenvectors from covariance matrix
  cov_matrix = np.cov(np.transpose(data))
  eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
  
  # Sort eigenvalues by value to get principal comps with highest eigenvalue
  index = np.flipud(np.argsort(eigen_val))
  eigen_val = eigen_val[index]
  eigen_vec = eigen_vec[:,index]

  return eigen_val, eigen_vec

def plot_cumulative_variance(eigen_val):
  cumvar = np.cumsum(eigen_val) / np.sum(eigen_val)
  plt.plot(cumvar)
  plt.show()
  
  for i in range(len(cumvar)):
    print('\nCum. variance captured by the ' + str(i+1) + 
          '. PC: ' + str(np.around(cumvar[i],3)))

def main():
  data = np.genfromtxt('9_gold_exp_advantage_matrix.csv', delimiter=',', skip_header=1)
  norm_data = preprocessing.scale(data[:,1:])
  
  eigen_val, eigen_vec = PCA(norm_data)
  Y = np.dot(norm_data, eigen_vec[:,:2])
  """
  #eigen_val, eigen_vec = PCA(data)
  #Y = np.dot(data, eigen_vec[:,:2])
  plt.figure(figsize=(15,7))
  patches = [mpatches.Patch(color='red', label='Lost game'), mpatches.Patch(color='green', label='Won game')]
  color = ["red", "green"]
  for i in range(2):
    indices = np.where(data[:,0] == i)[0]
    #todo = np.random.choice(indices, size=300)
    #y = Y[todo]
    y = Y[indices]
    plt.scatter(y[:,0], y[:,1], c=color[i], s=2, alpha=0.01)
  plt.legend(handles=patches)
    
  # Define plot layout
  plt.title('PC-plot of colorcoded classes on 1.498.131 match-stamps')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.xlim(-7,7)
  plt.ylim(ymin=-5)
  plt.show()

  
  
  plot3d(data)
  """
  
  plt.figure(figsize=(15,7))
  duration = data[:,1]
  gpm = data[:,2]
  patches = [mpatches.Patch(color='red', label='Lost game'), mpatches.Patch(color='green', label='Won game')]
  color = ['red','green']
  for i in range(2):
      plt.scatter(duration[np.where(data[:,0] == i)[0]], gpm[np.where(data[:,0] == i)[0]], s=8,c=color[i], alpha=0.02)
  plt.legend(handles=patches)
  plt.title('Scatterplot of gold advantage vs. duration with colorcoded classes on 1.498.131 match-stamps')
  plt.xlim(xmax=90)
  plt.ylim(-60000,60000)
  plt.ylabel('Gold advantage')
  plt.xlabel('Duration')  
  plt.show()

  plt.figure(figsize=(15,7))
  duration = data[:,1]
  xpm = data[:,3]
  patches = [mpatches.Patch(color='red', label='Lost game'), mpatches.Patch(color='green', label='Won game')]
  color = ['red','green']
  for i in range(2):
      plt.scatter(duration[np.where(data[:,0] == i)[0]], xpm[np.where(data[:,0] == i)[0]], s=8,c=color[i], alpha=0.02)
  plt.legend(handles=patches)
  plt.title('Scatterplot of exp advantage vs. duration with colorcoded classes on 1.498.131 match-stamps')
  plt.xlim(xmax=90)
  plt.ylim(-60000,60000)
  plt.ylabel('exp advantage')
  plt.xlabel('Duration')  
  plt.show()
  
  
  
  
main()

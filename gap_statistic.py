import math
import random
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from hierarchical_clustering import hierarchical_clustering


def generate_sample(ranges):
  return [random.uniform(ranges[0][i],ranges[1][i]) for i in range(len(ranges[0]))]


def generate_B_sets(B, n_samples, ranges):
  return [np.array([generate_sample(ranges) for i in range(n_samples)]) for i in range(B)]
  

def compute_Dr_and_Wk(clusters, data):
  # Get samples belonging to each cluster
  labels = np.unique(clusters)
  cluster_data = []
  for label in labels:
    indices = np.where(clusters == label)[0]
    cluster_data.append(np.take(data, indices, axis=0))  
  
  # Compute sum of pairwise distances for all samples in each respective cluster
  pairwise_dist_sum = []
  for cluster in cluster_data:
    pairwise_dist_sum.append(np.sum(pdist(cluster, 'sqeuclidean')))
  
  # Compute the pooled within-cluster sum of squares around the cluster means, W_k
  W_k = 0
  for k in range(len(cluster_data)):
    D_r = pairwise_dist_sum[k]
    sample_size = cluster_data[k].shape[0]
    W_k += D_r / (2 * sample_size)
  
  return W_k
  

def gap_k(k, B, data):
  # Compute min and max range for each feature
  n_samples  = data.shape[0]
  n_features = data.shape[1]
  ranges = [[np.min(data[:,x]) for x in range(n_features)],
            [np.max(data[:,x]) for x in range(n_features)]]

  # Calculate log(W_k) for our data
  clusters_Wk = cut_tree(hierarchical_clustering(data, 'ward', 'euclidean'), n_clusters=[k])
  log_Wk      = math.log(compute_Dr_and_Wk(clusters_Wk, data))

  # Generate B Monte Carlo datasets drawn from our reference distribution
  datasets = generate_B_sets(B, n_samples, ranges)
  
  # Calculate log(W_kb) for each dataset and get the expectation
  acc = 0
  for b in range(B):
    clusters = cut_tree(hierarchical_clustering(datasets[b], 'ward', 'euclidean'), n_clusters=[k])
    acc += math.log(compute_Dr_and_Wk(clusters, datasets[b]))
 
  return log_Wk, acc/B

  # Compute the estimated gap statistic
  #gap_k = (acc / B) - log_Wk
  #return gap_k


def monte_carlo_test(B, n_samples, data):
  # Compute min and max range for each feature
  n_samples  = data.shape[0]
  n_features = data.shape[1]
  ranges = [[np.min(data[:,x]) for x in range(n_features)],
            [np.max(data[:,x]) for x in range(n_features)]]
  
  # Generate B Monte Carlo datasets drawn from our reference distribution
  datasets = generate_B_sets(B, n_samples, ranges)
  mc_B = []
  for i in range(1,n_samples):
    # Calculate log(W_kb) for each dataset and get the expectation
    acc = 0
    for b in range(B):
      clusters = cut_tree(hierarchical_clustering(datasets[b], 'ward', 'euclidean'), n_clusters=[i])
      acc += math.log(compute_Dr_and_Wk(clusters, datasets[b]))
    mc_B.append(acc/B)
  print("Done with", B ,"Monte carlo test")


def main():
  hero_ids  = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,0]
  data      = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,1:]
  norm_data = preprocessing.scale(data)
  
  #result = hierarchical_clustering(norm_data, 'ward', 'euclidean')
  
  
  #cuttree = cut_tree(result, n_clusters=[3])
  #sum_of_pairwise_distances(cuttree, norm_data)
  
  #print(cuttree)
  
  
  B = [10,25,50,75,100,150,200]
  test = [monte_carlo_test(B[i], 10, norm_data) for i in range(len(B))]
  plt.plot(test[0])
  plt.plot(test[1])
  plt.plot(test[2])
  plt.plot(test[3])
  plt.plot(test[4])
  plt.plot(test[5])
  plt.plot(test[6])
  plt.show()
  
  #B = 50 
  """
  #gap_k_list = []
  k_max = 25
  
  
  list1 = []
  list2 = []
  for k in range(1,k_max):
    a,b = gap_k(k, B, data)
    list1.append(a)
    list2.append(b)
    #gap_k_list.append(gap_k(k, B, data))
    
  
  plt.plot(range(1,k_max),list1, color='r')
  plt.plot(range(1,k_max),list2)
  plt.title('Plot of expected and observed log(W_k)')
  plt.xlabel('number of clusters k')
  plt.ylabel('obs and exp log(W_k)')
  plt.show()
  """





main()



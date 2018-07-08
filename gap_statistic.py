import math
import random
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree
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
  
  # Compute sum of pairwise distances for all samples in each respective cluster, D_r
  pairwise_dist_sum = []
  for cluster in cluster_data:
    pairwise_dist_sum.append(np.sum(pdist(cluster, 'euclidean')))
  
  # Compute the pooled within-cluster sum around the cluster means, W_k
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
  log_Wkb_list = []
  for b in range(B):
    clusters = cut_tree(hierarchical_clustering(datasets[b], 'ward', 'euclidean'), n_clusters=[k])
    log_Wkb_list.append(math.log(compute_Dr_and_Wk(clusters, datasets[b])))
 
  return log_Wk, log_Wkb_list

  
def compute_sk(B, l, log_Wkb_list):
  # Compute standard deviation sd_k
  sd_k = np.sqrt(np.sum((np.array(log_Wkb_list) - l)**2)/B)
  
  # Compute s_k = sd_k * sqrt(1+(1/B))
  sk = sd_k * np.sqrt(1+(1/B))
  
  return sk


def choose_k(k_max, B, gap_list, l_list, log_Wkb_matrix):
  sk_list = [compute_sk(B, l_list[i], log_Wkb_matrix[i]) for i in range(len(l_list))]
  
  diff_list = np.array(gap_list) - np.array(sk_list)
  
  # Choose number of clusters as the smallest k where Gap(k) >= Gap(k+1)-s_{k+1}
  for i in range(len(diff_list)):
    k = i+1
    if gap_list[i] < diff_list[i+1]:
      continue
    return k


def main():
  data      = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', 
                            delimiter=',', skip_header=1)[:,1:]
  norm_data = preprocessing.scale(data)

  k_max = 10
  B = 100
  gap_list       = []
  log_Wk_list    = []
  log_Wkb_matrix = []
  l_list         = []
  
  for k in range(k_max):
    # Compute the estimated gap statistic
    log_Wk, log_Wkb = gap_k(k+1, B, norm_data)
    gap = sum(log_Wkb)/B - log_Wk  
    
    # Save W_k, W_kb for each b in B, l, and the gap
    log_Wk_list.append(log_Wk)
    log_Wkb_matrix.append(log_Wkb)
    l_list.append(sum(log_Wkb)/B)
    gap_list.append(gap)
    
    
  # Plot of expected and observed value of log(W_k)
  plt.plot(range(1,k_max+1), log_Wk_list, '--bo', label='Observed value', color='green')
  plt.plot(range(1,k_max+1), l_list, '--bo', label='Expected value', color='orange')
  plt.title('Plot of expected and observed log(W_k) per k')
  plt.xticks(np.arange(1,11,1.0))
  plt.xlabel('number of clusters k')
  plt.ylabel('observed and expected log(W_k)')
  plt.legend()
  plt.show()
  
  # Plot of gap between expected and observed value
  plt.plot(range(1,k_max+1),gap_list, '--bo')
  plt.title('Plot of gap between expected and observed log(W_k)')
  plt.xticks(np.arange(1,11,1.0))
  plt.xlabel('number of clusters k')
  plt.ylabel('gap')
  plt.show()

  optimal_k = choose_k(k_max, B, gap_list, l_list, log_Wkb_matrix)
  print('The optimal number of clusters using gap statistic: ', optimal_k)



main()
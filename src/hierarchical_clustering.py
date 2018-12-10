import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


def hierarchical_clustering(data, method, metric):
  hierarchy = linkage(data, method, metric)
  return hierarchy


def plot_dendrogram(data, hero_ids):
  plt.figure()
  plt.figure(figsize=(18,7))
  dendrogram(data, labels=[str(int(i)) for i in hero_ids.tolist()])
    
  plt.xticks(fontsize=8, rotation=90)
  plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
  plt.xlabel('Hero id',  fontsize=12)
  plt.ylabel('Distance', fontsize=12)
  plt.show()

def get_labels(dendrogram, k):
  return fcluster(dendrogram, k, criterion='maxclust')

def get_label_from_hero_id(hero_id, hero_ids, labels):
  index = np.where(hero_ids == hero_id)
  return labels[index]
  
  

def main():  
  # Load hero ids, then load data and normalize it to mean zero and stand deviation one
  hero_ids  = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,0]
  data      = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,1:]
  norm_data = preprocessing.scale(data)
  
  result = hierarchical_clustering(norm_data, 'ward', 'euclidean')
  plot_dendrogram(result, hero_ids)
  
  print(get_labels(result, 6))
  
  
main()
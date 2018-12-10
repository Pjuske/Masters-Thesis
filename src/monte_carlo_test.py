import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.cluster.hierarchy import cut_tree
from hierarchical_clustering import hierarchical_clustering
from gap_statistic import generate_B_sets, compute_Dr_and_Wk


def monte_carlo_test(B, k, data):
  # Compute min and max range for each feature
  n_samples  = data.shape[0]
  n_features = data.shape[1]
  ranges = [[np.min(data[:,x]) for x in range(n_features)],
            [np.max(data[:,x]) for x in range(n_features)]]
  
  # Generate B Monte Carlo datasets drawn from our reference distribution
  datasets = generate_B_sets(B, n_samples, ranges)
  mc_B = []
  for i in range(k):
    # Calculate log(W_kb) for each dataset and get the expectation
    acc = 0
    for b in range(B):
      clusters = cut_tree(hierarchical_clustering(datasets[b], 'ward', 'euclidean'), n_clusters=[i+1])
      acc += math.log(compute_Dr_and_Wk(clusters, datasets[b]))
    mc_B.append(acc/B)
  
  print("Done with a B =", B ,"Monte Carlo test")
  return mc_B



def main():
  data      = np.genfromtxt('data-retriever/datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,1:]
  norm_data = preprocessing.scale(data)
  
  # Run Monte Carlo simulations with different values of B
  simulations = 5
  B = [1,5,10,50,100,200]
  
  for b in B:
    test = [monte_carlo_test(b, 10, norm_data) for i in range(simulations)]
    
    # Plot each simulation in the same plot for a specific B 
    for i in range(len(test)):
      plt.plot([x+1 for x in range(10)], test[i], label='Simulation #'+str(i+1))
      plt.legend()
      plt.title('Monte Carlo simulations for B = '+str(b))
      plt.xticks(np.arange(1,11, 1.0))
      plt.xlabel('Number of clusters k')
      plt.ylabel('Expected value of log(W_k)')
    plt.figure()
  plt.show()
  
  
main()
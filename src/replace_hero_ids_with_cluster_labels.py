import csv
import numpy as np

def replace_heroes_with_labels(hero_comp, cluster_labels):
  teamsize = 5
  clustercounts = [0] * 10
  
  radiant_hc = hero_comp[:5].astype(np.int64)
  dire_hc    = hero_comp[5:].astype(np.int64)
  
  for i in range(teamsize):
    
    radiant_label = cluster_labels[radiant_hc[i]-1]
    dire_label    = 5 + cluster_labels[dire_hc[i]-1]
    
    clustercounts[radiant_label] += 1
    clustercounts[dire_label]    += 1
    
  return clustercounts
  
    
def newdata(data, labels):
  for i in range(len(data)):
    data[i,:] = replace_heroes_with_labels(data[i,:], labels)    
  
  return data


def main():
  data   = np.genfromtxt('pro_team_composition.csv', delimiter=',', skip_header=1, usecols=(range(10)))
  labels = np.genfromtxt('pro_team_composition.csv', delimiter=',', dtype=str, skip_header=1, usecols=(10))
  
  # Get cluster labels and replace hero ids with the cluster labels
  clusterlabels = np.genfromtxt('hero_label_result.csv', delimiter=',').astype(np.int64)
  new_data = newdata(data, clusterlabels)

  # Write data to new file
  writer = csv.writer(open("./%s" % ("data_after_clustering.csv"), 'w', newline=''))
  for i in range(len(new_data)):
    row   = new_data[i].astype(np.int64)
    label = [int(labels[i]=='TRUE')]
    
    writer.writerow(np.concatenate((row, label)))
  
    
main()
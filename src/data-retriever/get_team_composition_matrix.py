import csv
import numpy as np
from itertools import combinations_with_replacement

# Method from "https://stackoverflow.com/questions/37711817/generate-all-
#              possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego
def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in combinations_with_replacement(masks, n): 
      yield sum(c)

def look_up(composition1, composition2, data):
  # En der counter den ene sandsynlighed og en der counter den anden sandsynlighed?
  pass



def main():
  data = np.loadtxt('datasets/5_team_compositions.csv', delimiter=',', skiprows=1)
  
  combinations = np.array(list(partitions(5, 6)))
  radiant_win  = data[:,0]
  team_1       = data[:,1:7]
  team_2       = data[:,7:]
  
  
  match_count_matrix = np.empty((252,252))
  win_rate_matrix    = np.empty((252,252))
  
  file_1 = 'datasets/6_match_count_matrix.csv'
  file_2 = 'datasets/7_win_rate_matrix.csv'
  
  with open("./%s" % (file_1), 'w', newline='', encoding="utf-8") as outf, open("./%s" % (file_2), 'w', newline='', encoding="utf-8") as outg:
    header = ['.'] + [str(e) for e in list(combinations)]
    
    dw1 = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw2 = csv.DictWriter(outg, dict.fromkeys(header).keys(), delimiter=',')
    
    dw1.writeheader()
    dw2.writeheader()
  
    for i in range(len(combinations)):
      print(i, ' done')
      my_dict1={}
      my_dict2={}
      my_dict1['.'] = combinations[i]
      my_dict2['.'] = combinations[i]      
      
      
      for j in range(len(combinations)):
        # Find games where the teams have composition a and b
        a = np.where(np.all(team_1 == combinations[i],axis=1))
        b = np.where(np.all(team_2 == combinations[j],axis=1))
        indices1 = np.intersect1d(a,b)
        
        # Find games where the teams have composition b and a
        B = np.where(np.all(team_1 == combinations[j],axis=1))
        A = np.where(np.all(team_2 == combinations[i],axis=1))
        indices2 = np.intersect1d(B,A)
        
        
        # If matchup exists, calculate total matches and win rate
        if len(indices1) + len(indices2) != 0:
          # Check if the two team compositions are exactly the same
          if list(combinations[i]) == list(combinations[j]):
            match_count_matrix[i,j] = match_count_matrix[i,j] + len(indices1) + len(indices2)
            win_rate_matrix[i,j]    = 0.50
          
          else:
            # Find matches where composition A wins
            compA_win1  = np.take(radiant_win, indices1, axis=0)
            compA_win2  = np.take(1 - radiant_win, indices2, axis=0)
            
            # Calculate total matches won and total matches played
            total_won     = np.count_nonzero(compA_win1) + np.count_nonzero(compA_win2)
            total_matches = len(indices1) + len(indices2)
            
            # Calculate win rate and save the results into the matrices
            win_rate = total_won / total_matches
            
            match_count_matrix[i,j] = total_matches
            win_rate_matrix[i,j]    = win_rate
        
        # Else, set total matches to 0 and winrate to either 50% or -1
        else:
          match_count_matrix[i,j] = 0
          win_rate_matrix[i,j] = 0.50 if list(combinations[i]) == list(combinations[j]) else -1
      
        # Save results to dictionary
        my_dict1[str(combinations[j])] = match_count_matrix[i,j]
        my_dict2[str(combinations[j])] = win_rate_matrix[i,j]
        
      # Write to csv file
      dw1.writerow(my_dict1) 
      dw2.writerow(my_dict2)

      
main()
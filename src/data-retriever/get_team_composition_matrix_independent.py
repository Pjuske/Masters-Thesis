from __future__ import division
import csv
import numpy as np
from itertools import combinations_with_replacement

# Method from "https://stackoverflow.com/questions/37711817/generate-all-
#              possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego
def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in combinations_with_replacement(masks, n): 
      yield sum(c)


def main():
  data = np.loadtxt('datasets/5_team_compositions.csv', delimiter=',', skiprows=1)
  
  combinations = np.array(list(partitions(5, 6)))
  radiant_win  = data[:,0]
  team_1       = data[:,1:7]
  team_2       = data[:,7:]
  
  with open("./%s" % ('datasets/8_independent_matrix.csv'), 'w', newline='', encoding="utf-8") as outf:
    header = ['composition','match_count','win_rate']
    dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()

    for combination in combinations:
      my_dict = {}
      my_dict['composition'] = combination
      # Find games where the teams have composition a and b
      a = np.where(np.all(team_1 == combination,axis=1))
      b = np.where(np.all(team_2 == combination,axis=1))
      indices_same_comb = np.intersect1d(a,b)
      
      a_without_double_comb = [x for x in a[0] if x not in indices_same_comb]
      b_without_double_comb = [x for x in b[0] if x not in indices_same_comb]
      
      # Calculate number of matches for that combination
      count = len(a_without_double_comb) + len(b_without_double_comb) + len(indices_same_comb)
      my_dict['match_count'] = count
      
      # Save win rate as -1
      if count == 0:
        my_dict['win_rate'] = -1 
        
      else:
        # Calculate win rate for that combination
        compA_win     = np.take(radiant_win, a_without_double_comb, axis=0)
        compB_win     = np.take(1 - radiant_win, b_without_double_comb, axis=0)
        compSame_win  = len(indices_same_comb) * 0.5 
        
        total_won     = np.count_nonzero(compA_win) + np.count_nonzero(compB_win) + compSame_win
        win_rate      = total_won / count
        my_dict['win_rate'] = win_rate
      
      dw.writerow(my_dict)      
  
main()
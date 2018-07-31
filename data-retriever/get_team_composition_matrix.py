import numpy as np
from itertools import combinations_with_replacement
from sympy.solvers.diophantine import partition

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
  
  
  composition_matrix = np.empty((252,252))
  
  for i in range(len(combinations)):
    for j in range(len(combinations)):
      a = np.where(np.all(team_1 == combinations[i],axis=1))
      b = np.where(np.all(team_2 == combinations[j],axis=1))
      
      # Get indices of games where team_1 has composition a and team_2 has composition b
      indices = np.intersect1d(a,b)
      
      # Check if that match up exists, else save win rate as -1
      if indices != 0:
        # Compute win rate and save (number of matches, win rate) into result
        win_rate = np.take(radiant_win, indices, axis=0).sum() / len(indices)
  
        result = ((a == b).sum(), np.around(win_rate))
        composition_matrix[i,i] = result
      
      else:
        composition_matrix[i,i] = (0, -1)
     
  print(composition_matrix)
      
main()
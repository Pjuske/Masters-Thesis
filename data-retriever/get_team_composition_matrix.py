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
  possible_combinations = np.array(list(partitions(5, 6)))
  print(possible_combinations.shape)
  
  # TODO: make a look-up function.
  # Get winrate and count of a specific composition vs another composition
  
  # Bjarke siger: kig begge veje men pas meget på. Matrix bliver ikke symmetrisk
  for i in range(len(possible_combinations)):
    for j in range(len(possible_combinations)):
        
      
main()
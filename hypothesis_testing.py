import math
import numpy as np
import scipy.stats as stats

"""
Let X and Y be random variables describing the gold or exp advantage of radiant and dire team.
Now, our null-hypothesis is that mean(X) = mean(Y).
That is, there is no significant difference between the gold/exp advantage of radiant and dire team.
"""
def hypothesis_testing(X, Y, alpha):
  # Calculate t-statistic
  numerator   = np.mean(X) - np.mean(Y)
  denominator = np.sqrt( (np.std(X)**2) / len(X) + (np.std(Y)**2) / len(Y) )
  t_statistic = numerator / denominator
  
  # Compute degrees of freedom v
  numerator   = (np.std(X)**2 / len(X) + np.std(Y)**2 / len(Y))**2
  denominator = (np.std(X)**4 / (len(X)**2 * (len(X)-1))) + (np.std(Y)**4 / (len(Y)**2 * (len(Y)-1)))
  dof         = math.floor(numerator / denominator)
  
  # Compute p-value
  p = stats.t.cdf(-abs(t_statistic), dof) * 2
  print(p)
  
  # Reject null hypothesis if p < alpha
  alpha = 0.05
  return (p < alpha)
  

"""
Take a vector with gpm and xpm values for winning and losing team.
Returns the result of the hypothesis testing for both gold and experience advantage.
"""
def test_hypothesis(radiant_win, gpm, xpm, alpha):  
  winner_gpm, winner_xpm = [], []
  loser_gpm, loser_xpm   = [], []  

  # Check which team won and calculate their advantage
  for i in range(len(radiant_win)):
    if radiant_win[i] == 1 :
      winner_gpm.append(gpm[i])
      winner_xpm.append(xpm[i])
      loser_gpm.append(gpm[i] * (-1))
      loser_xpm.append(xpm[i] * (-1))
    else:
      winner_gpm.append(gpm[i] * (-1))
      winner_xpm.append(xpm[i] * (-1))
      loser_gpm.append(gpm[i])
      loser_xpm.append(xpm[i])      
      
  gpm_hypothesis = hypothesis_testing(winner_gpm, loser_gpm, alpha)
  print('\ngpm calculation:', gpm_hypothesis)
  xpm_hypothesis = hypothesis_testing(winner_xpm, loser_xpm, alpha)
  print('\nxpm calculation:', xpm_hypothesis)
  return gpm_hypothesis and xpm_hypothesis
  

def main():
  data = np.genfromtxt('data-retriever/datasets/9_gold_exp_advantage_matrix.csv',delimiter=',',skip_header=1)
  alpha = 0.05
  
  result = []
  for i in range(15):
    duration = data[:, 1]
    winning_team  = data[np.where(duration == i)[0]][:,1]
    gpm_advantage = data[np.where(duration == i)[0]][:,2]
    xpm_advantage = data[np.where(duration == i)[0]][:,3]
    
    result.append(test_hypothesis(winning_team, gpm_advantage, xpm_advantage, alpha))
  
  print(result)
  
  
  
  
main()
  
  
  
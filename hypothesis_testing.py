import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

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
  
  # Reject null hypothesis if p < alpha
  alpha = 0.05
  return p#(p < alpha)
  

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
  #print('\ngpm calculation:', gpm_hypothesis)
  xpm_hypothesis = hypothesis_testing(winner_xpm, loser_xpm, alpha)
  #print('\nxpm calculation:', xpm_hypothesis)
  return [gpm_hypothesis, xpm_hypothesis]
  

def main():
  data = np.genfromtxt('data-retriever/datasets/9_gold_exp_advantage_matrix.csv',delimiter=',',skip_header=1)
  alpha = 0.05
  
  result = []
  for i in range(int(np.max(data[:,1]))):
    duration        = data[:, 1]
    
    match_count   = data[np.where(duration == i)[0]][:,0]
    #winning_team  = data[np.where(duration == i)[0]][:,1]
    gpm_advantage = data[np.where(duration == i)[0]][:,2]
    xpm_advantage = data[np.where(duration == i)[0]][:,3]
    
    radiant_matches = list(match_count).count(1)
    dire_matches    = list(match_count).count(0)
    
    # Skip if there are not enough matches
    if (radiant_matches < 20 or dire_matches < 20):
        continue
    
    result.append(test_hypothesis(match_count, gpm_advantage, xpm_advantage, alpha))
  
  gpm_hypotheses = [result[i][0] for i in range(len(result))]
  xpm_hypotheses = [result[i][1] for i in range(len(result))]

  return result, gpm_hypotheses, xpm_hypotheses 

result, gpm_hypotheses, xpm_hypotheses = main()

alpha=0.05
plt.figure(figsize=(12,6))
def minhandler(number):
    if(number>0):
        return number
    else:
        return sys.float_info.min

fail_to_reject = list(np.where(np.array(gpm_hypotheses) > alpha)[0]) + list(np.where(np.array(xpm_hypotheses) > alpha)[0])
fail_to_reject_values = [gpm_hypotheses[i] for i in np.where(np.array(gpm_hypotheses) > alpha)[0]] + [xpm_hypotheses[i] for i in np.where(np.array(xpm_hypotheses) > alpha)[0]]

xpm_hypotheses = [minhandler(i) for i in range(len(xpm_hypotheses))]
gpm_hypotheses = [minhandler(i) for i in range(len(gpm_hypotheses))]

plt.plot(gpm_hypotheses, label='gpm p-values', linewidth=3.5, color='#53E53C')
plt.plot(xpm_hypotheses, label='xpm p-values', linewidth=3.0, color='orange')
plt.plot([alpha for i in range(len(xpm_hypotheses))], label=('alpha = ' + str(alpha)), color='black')
plt.scatter(fail_to_reject, fail_to_reject_values, label='Failed to reject hypothesis', color='red')
plt.title('p-values from hypothesis-testing gold and exp advantage per minute')
plt.xticks(np.arange(0,len(result),2.0))
plt.yticks(np.arange(0,1.1,0.1))
plt.xlabel('Game minute')
plt.ylabel('p-value')
plt.legend()
plt.show()


print(gpm_hypotheses)
print(xpm_hypotheses)
"""
log_fail_to_reject = list(np.where(np.array([math.log(gpm_hypotheses[i]) for i in range(len(gpm_hypotheses))]) > math.log(alpha))[0]) + list(np.where(np.array([math.log(xpm_hypotheses[i]) for i in range(len(xpm_hypotheses))]) > math.log(alpha))[0])
log_fail_to_reject_values = [[math.log(gpm_hypotheses[i]) for i in range(len(gpm_hypotheses))][i] for i in np.where(np.array([math.log(gpm_hypotheses[i]) for i in range(len(gpm_hypotheses))]) > math.log(alpha))[0]] + [[math.log(xpm_hypotheses[i]) for i in range(len(xpm_hypotheses))][i] for i in np.where(np.array([math.log(xpm_hypotheses[i]) for i in range(len(xpm_hypotheses))]) > math.log(alpha))[0]]

plt.figure(figsize=(12,6))
plt.plot([math.log(gpm_hypotheses[i]) for i in range(len(gpm_hypotheses))], label='log(gpm p-values)', linewidth=3.5, color='#53E53C')
plt.plot([math.log(xpm_hypotheses[i]) for i in range(len(xpm_hypotheses))], label='log(xpm p-values)', linewidth=3.0, color='orange')
plt.plot([math.log(alpha) for i in range(len(xpm_hypotheses))], label=('alpha = log(' + str(alpha) + ')'), color='black')
plt.scatter(log_fail_to_reject, log_fail_to_reject_values, label='Failed to reject hypothesis', color='red')
plt.title('log(p-values) from hypothesis-testing gold and exp advantage per minute')
plt.xticks(np.arange(0,len(result),2.0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Game minute')
plt.ylabel('log(p-value)')
plt.legend()
plt.show()
"""

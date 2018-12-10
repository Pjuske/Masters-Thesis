import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load data and find the duration of the longest game
data = np.loadtxt('datasets/9_gold_exp_advantage_matrix.csv',delimiter=',', skiprows=1)
max_duration = int(np.max(data[:,1]))

# For each duration, find out how many matches are at least that long
durations_win = data[np.where(data[:,0]==1)[0]][:,1]
durations_loss = data[np.where(data[:,0]==0)[0]][:,1]

# Plot the histogram showing number of data points for different timestamps
sns.set()
plt.figure(figsize=(8,6))
plt.hist((durations_win,durations_loss),bins=range(0,max_duration+1),
         histtype='barstacked',label=('Win', 'Loss'), color=('#cc99ff','#cc99ff'))
plt.title('Number of data points for different timestamps', fontsize=14)
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# Plot the same data points in log-sclae
plt.figure(figsize=(8,6))
plt.hist((durations_win,durations_loss),bins=range(0,max_duration+1),log=True,
         histtype='barstacked',label=('Win', 'Loss'), color=('#cc99ff','#cc99ff'))
plt.title('Number of data points for different timestamps in log-scale', fontsize=14)
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

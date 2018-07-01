import numpy as np
import csv

header = ['hero_id','kills','deaths','assists','hero_healing',
          'last_hits','gold_per_min','xp_per_min','tower_damage',
          'hero_damage','obs_placed','sen_placed']


def main():
  data   = np.genfromtxt('detailed_player_data.csv',delimiter=',', skip_header=1)
  new_data = data[data[:,0].argsort()]
  
  
  with open("./%s" % ('mean_player_data.csv'), 'w', newline='',  encoding="utf-8") as f:
    # Write header into csv file    
    dw = csv.DictWriter(f, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()
    
    current_hero_id = 1
    acc_data = [0]*11
    occurences = 0
    for i in range(len(new_data)):
      if new_data[i,0] == current_hero_id:
          acc_data = acc_data + new_data[i, 1:]
          occurences += 1
      else:
          mean_data = acc_data / (occurences - 1)
          print(mean_data, '\n')
          f.write(','.join([str(current_hero_id)] + [str(np.around(e,3)) for e in mean_data.tolist()]) + '\n')
          occurences = 0
        
          print('Current hero_id finished:', current_hero_id)
        
          acc_data = new_data[i, 1:]
          current_hero_id = int(new_data[i,0])
          
    
    # Save the last hero id as well
    mean_data = acc_data / (occurences - 1)
    print(mean_data, '\n')
    f.write(','.join([str(current_hero_id)] + [str(np.around(e,3)) for e in mean_data.tolist()]) + '\n')
    
  
  
  
main()
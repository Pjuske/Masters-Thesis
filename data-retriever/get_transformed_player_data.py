import numpy as np
import json
import csv
import pandas

def CustomParser(data):
    j1 = json.loads(data)
    return j1



def main():
  #data = json.loads('detailed_match_data_final_without_blank_cells.csv') #np.genfromtxt('detailed_match_data_final_without_blank_cells.csv',skip_header=1, delimiter=',')
  #print(data.shape)
  
  columns = ['player0','player1','player2','player3','player4',
             'player5','player6','player7','player8','player9']
  
  feats_wanted = ['hero_id','kills','deaths','assists','hero_healing',
                  'last_hits','gold_per_min','xp_per_min','tower_damage',
                  'hero_damage','obs_placed','sen_placed']


  
  
  # Reads the file the same way that you did
  csv_file = csv.DictReader(open('detailed_match_data_final_players.csv', 'r'), delimiter=';')

  
  # Created a list and adds the rows to the list
  json_list = []
  for row in csv_file:
      json_list.append(row)
  
  
  
  with open("./%s" % ('detailed_player_data.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file    
    header = feats_wanted
    dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()

    for i in range(len(json_list)):
      if (i % 3000 == 0):
          print(i, 'iterations done...\n')
      
      try:
        for player in columns:
          row = json_list[i]
          player_data = json.loads(row[player].replace("\'", "\""))
      
          my_dict={}
          for feature in feats_wanted:
            my_dict[feature] = player_data[feature]
            
      
          dw.writerow(my_dict)
     
      # Ignore JSONDecodeError
      except ValueError:
        print('Decoding JSON has failed for row ',i)      
      
      

main()
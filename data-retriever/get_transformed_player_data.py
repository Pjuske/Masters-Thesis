import numpy as np
import json
import csv

def CustomParser(data):
    j1 = json.loads(data)
    return j1



def main():
  playerColumns = ['player0','player1','player2','player3','player4',
                   'player5','player6','player7','player8','player9']
 
  team_feats    = ['radiant_score','dire_score','radiant_gold_adv','radiant_xp_adv']
  player_feats  = ['hero_id','kills','deaths','assists','hero_healing',
                   'last_hits','gold_per_min','xp_per_min','tower_damage',
                   'hero_damage','obs_placed','sen_placed']


  # Reads the detaled match data, ignoring those with missing values
  csv_file = csv.DictReader(open('datasets/2_detailed_match_data_without_blanks.csv', 'r'), delimiter=';')

  
  # Created a list and adds the rows to the list
  json_list = []
  for row in csv_file:
      json_list.append(row)
  
  
  with open("./%s" % ('datasets/10_regression_data.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file    
    header = player_feats + team_feats
    dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()

    for i in range(len(json_list)):
      if (i % 3000 == 0):
          print(i*10, 'iterations done...\n')
      
      try:
        row = json_list[i]
        my_dict={}
        
        for player in playerColumns:  
          # Parse player data and add it to dicitonary
          player_data = json.loads(row[player].replace("\'", "\""))
          for feature in player_feats:
            my_dict[feature] = player_data[feature]
            
          # Add team data to dictionary
          for feature in team_feats:
            my_dict[feature] = json.loads(row[feature])  
          
          # Write row
          dw.writerow(my_dict)
            
     
      # Ignore JSONDecodeError
      except ValueError:
        print('Decoding JSON has failed for row ',i)      
            

main()
import numpy as np
import json
import csv
import ast

def CustomParser(data):
    j1 = json.loads(data)
    return j1


def get_team_scores(radiant_score, dire_score, player_slot):
  # If player is on team Dire, return their score for the feature 'my_team_score'
  if player_slot > 4:
    return dire_score, radiant_score
  else:
    return radiant_score, dire_score
  

def get_final_gpm_xpm(gpm_data, xpm_data, player_slot):
  # Get the gold/xpm difference when the game ended
  gpm = gpm_data[-1]
  xpm = xpm_data[-1]
  
  # If player is on team Dire, negate the gpm and xpm values
  if player_slot > 4:
    return gpm*(-1),xpm*(-1)
  else:
    return gpm, xpm


def main():
  playerColumns = ['player0','player1','player2','player3','player4',
                   'player5','player6','player7','player8','player9']
  
  player_feats  = ['hero_id','kills','deaths','assists','hero_healing',
                   'last_hits','gold_per_min','xp_per_min','tower_damage',
                   'hero_damage','obs_placed','sen_placed']
  
  team_feats    = ['my_team_score','enemy_team_score','my_team_gold_adv','my_team_xp_adv']


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
          print(i, 'iterations done...\n')
      
      try:
        my_dict = {}
        row     = json_list[i]
        
        for player in playerColumns:  
          # Parse player data and add it to dicitonary
          player_data = json.loads(row[player].replace("\'", "\""))
          player_slot = player_data['player_slot']
          
          for feature in player_feats:
            my_dict[feature] = player_data[feature]
            
          # Add team data to dictionary
          my_score, enemy_score       = get_team_scores(json.loads(row['radiant_score']),json.loads(row['dire_score']),player_slot)
          my_gold_avg, my_xpm_avg     = get_final_gpm_xpm(json.loads(row['radiant_gold_adv']),json.loads(row['radiant_xp_adv']),player_slot)
          
          my_dict['my_team_score']    = my_score
          my_dict['enemy_team_score'] = enemy_score
          my_dict['my_team_gold_adv'] = my_gold_avg
          my_dict['my_team_xp_adv']   = my_xpm_avg
               
          # Write row
          dw.writerow(my_dict)
            
     
      # Ignore JSONDecodeError
      except ValueError:
        print('Decoding JSON has failed for row ',i)      
            

main()
import numpy as np
import urllib.request
import time
import json
import csv


features        = ['radiant_win','duration','skill',
                   'radiant_score','dire_score',
                   'barracks_status_radiant','barracks_status_dire',
                   'tower_status_radiant','tower_status_dire',
                   'radiant_gold_adv','radiant_xp_adv',
                   'patch','region']

players         = ['player0','player1','player2','player3','player4',
                   'player5','player6','player7','player8','player9']

player_features = ['player_slot','hero_id','kills','deaths','assists','hero_healing',
                   'last_hits','gold_per_min','xp_per_min','tower_damage','hero_damage']



def format_player_data(players_data, features):
  formatted_data = []
  
  # For each player, retrieve the features wanted
  for player in players_data:
    my_dict = {}
    for feature in features:
      my_dict[feature] = player[feature]
      
    formatted_data.append(my_dict)
  
  return formatted_data  


def main():
  # Load file containing match ids that we want detailed match data for
  filename = 'match_ids_for_latest_matches.csv'
  data = np.genfromtxt(filename, dtype=np.dtype('i8'), skip_header=1)
  
  with open("./%s" % ('detailed_match_data.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file    
    header = features + players
    dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()
    
    for i in range(len(data)):
      # Avoid exceeding the rate limit of 60 calls per minute
      # See https://www.opendota.com/api-keys
      if (i % 50 == 0 and i != 0):
        print(str(i) + '/' + str(i) + ' iterations done...\n')
        time.sleep(70)
      
      # Get detailed match data
      match_id = data[i]
      url = "https://api.opendota.com/api/matches/" + str(match_id)
      detailed_data = json.loads(urllib.request.urlopen(url).read())

      # Build a json object consisting of the wanted features
      my_dict={}
      for feature in features:
        my_dict[feature] = detailed_data[feature]

      # Include data about player features as well
      player_data = format_player_data(detailed_data['players'], player_features)
      for i in range(len(player_data)):
        feature = players[i]
        my_dict[feature] = player_data[i]
      
      dw.writerow(my_dict)
    
main()
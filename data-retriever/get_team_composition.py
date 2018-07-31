import csv
import json
import numpy as np
from sklearn import preprocessing
from hierarchical_clustering import hierarchical_clustering, get_labels, get_label_from_hero_id

players      = ['player0','player1','player2','player3','player4',
                'player5','player6','player7','player8','player9']

roles_wanted = ['radiant_role1','radiant_role2','radiant_role3',
                'radiant_role4','radiant_role5','radiant_role6',
                'dire_role1','dire_role2','dire_role3','dire_role4',
                'dire_role5','dire_role6']

def main():  
  # Get dendrogram and labels for each hero id
  hero_ids  = np.genfromtxt('datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,0]
  data      = np.genfromtxt('datasets/4_detailed_player_data_mean.csv', delimiter=',', skip_header=1)[:,1:]
  norm_data = preprocessing.scale(data)
  result = hierarchical_clustering(norm_data, 'ward', 'euclidean')
  labels = get_labels(result, 6)  
  
  # Open csv file containing match data
  csv_file = csv.DictReader(open('datasets/22_detailed_match_data.csv', 'r'), delimiter=';')  
  
  # Created a list and adds the rows to the list
  json_list = []
  for row in csv_file:
    json_list.append(row)


  with open("./%s" % ('datasets/5_team_compositions.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file    
    header = ['radiant_win'] + roles_wanted
    dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
    dw.writeheader()

    for i in range(len(json_list)):
      row                 = json_list[i]
      roles               = 6
      count_radiant_roles = [0] * roles
      count_dire_roles    = [0] * roles
   
      # Create a dictionary and add information about who won the game
      my_dict={}
      my_dict['radiant_win'] = 1 if (row['radiant_win']==True) else 0
      
      
      try:
        for player in range(len(players)):
          # Extract hero_id from each player and get their role
          player_data = json.loads(row[players[player]].replace("\'", "\"").replace("None","0"))          
          hero_id = player_data['hero_id']
          cluster_label = int(get_label_from_hero_id(hero_id, hero_ids, labels))-1
          
          # Count how many occurences of each role there is in each team          
          if (player < 5):
            # radiant team
            count_radiant_roles[cluster_label] +=1          
          else:
            #dire team
            count_dire_roles[cluster_label] += 1          
          
        # Save list of roles into result
        for i in range(roles):
          my_dict[roles_wanted[i]] = count_radiant_roles[i]
        for i in range(roles):
          my_dict[roles_wanted[i+roles]] = count_dire_roles[i]      
        
        # Write result into file
        dw.writerow(my_dict) 
      
      # Ignore JSONDecodeError
      except ValueError:
        print('Decoding JSON has failed for row ',i)

main()
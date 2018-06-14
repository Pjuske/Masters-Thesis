import numpy as np
import urllib.request
import time
import json
import csv


# TODO: make it work for fields containing gigantic string lengths, such as 'players'
#features = ['match_id','barracks_status_dire','barracks_status_radiant',
#            'dire_score','duration','picks_bans','radiant_gold_adv',
#            'radiant_score','radiant_win','radiant_xp_adv','tower_status_dire',
#            'tower_status_radiant','players','patch','region','skill']

features = ['match_id','barracks_status_dire','barracks_status_radiant',
            'dire_score','duration','radiant_score','radiant_win']

def main():
  # Load file containing match ids that we want detailed match data for
  filename = 'match_ids_for_latest_matches.csv'
  data = np.genfromtxt(filename, dtype=np.dtype('i8'), skip_header=1)
  
  with open("./%s" % ('detailed_match_data.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file    
    dw = csv.DictWriter(outf, dict.fromkeys(features).keys(), delimiter=',')
    #dw = csv.DictWriter(outf, dict.fromkeys(['players']).keys())
    dw.writeheader()
    
    #for i in range(len(data)):
    for i in range(3):
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
        
      dw.writerow(my_dict)
      
      
    
    # TODO make a new file where it's possible to take a subset
    # of the fields within the detailed match data...
    
    

  
    
main()
import urllib.request
import time
import json
import csv


def main():
  
  iterations = 2000
  less_than_match_id = None
  
  with open("./%s" % ('match_ids_for_latest_matches.csv'), 'w', newline='') as outf:
    dw = csv.DictWriter(outf, dict.fromkeys(['match_id']).keys())
    dw.writeheader()
    
    for i in range(iterations):
      # Avoid exceeding the rate limit of 60 calls per minute
      # See https://www.opendota.com/api-keys
      if (i % 50 == 0 and i != 0):
        print(str(i) + '/' + str(iterations) + ' iterations done...\n')
        time.sleep(75)
  
      # Get list of randomly sampled public matches
      if (i == 0):
        url = 'https://api.opendota.com/api/publicMatches'
      else:
        url = 'https://api.opendota.com/api/publicMatches?less_than_match_id=' + str(less_than_match_id)
      
      data = json.loads(urllib.request.urlopen(url).read())
      less_than_match_id = data[-1]['match_id']
      
      for row in data:    
        # Select 'Ranked' as lobby type and 'Normal/Captains Mode' as game mode in high skilled games
        if (row['avg_mmr'] != None and row['avg_mmr'] >= 3400):
          if row['lobby_type']==7 and (row['game_mode']==2 or row['game_mode']==22):
            match_id = json.loads('{"match_id": ' + str(row['match_id']) + '}')          
            dw.writerow(match_id)


main()

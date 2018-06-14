import numpy as np
import urllib.request
import time
import json
import csv


def main():
  # Load file containing match ids that we want detailed match data for
  filename = 'match_ids_for_latest_matches.csv'
  data = np.genfromtxt(filename, dtype=np.dtype('i8'), skip_header=1)
  
  with open("./%s" % ('detailed_match_data.csv'), 'w', newline='',  encoding="utf-8") as outf:
    # Write header into csv file
    url = "https://api.opendota.com/api/matches/" + str(data[1])
    header = json.loads(urllib.request.urlopen(url).read()).keys()
    dw = csv.DictWriter(outf, header)
    dw.writeheader()
    
    for i in range(len(data)):
      # Avoid exceeding the rate limit of 60 calls per minute
      # See https://www.opendota.com/api-keys
      if (i % 50 == 0 and i != 0):
        print(str(i) + '/' + str(i) + ' iterations done...\n')
        time.sleep(70)
      
      # Get detailed match data and write into csv file
      match_id = data[i]
      url = "https://api.opendota.com/api/matches/" + str(match_id)
      detailed_data = json.loads(urllib.request.urlopen(url).read())
      dw.writerow(detailed_data)
    
    
    # TODO make a new file where it's possible to take a subset
    # of the fields within the detailed match data...
  
    
main()
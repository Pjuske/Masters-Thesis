import numpy as np
import urllib.request


def main():
  # TODO: load file containing match ids that we want detailed match data for
  match_id_file = 'match_ids_for_latest_matches.csv'
  data = np.loadtxt(match_id_file, delimiter=',')
  
  for match_id in range(len(data)):  #Example of match_id = 3921060078
    url = "https://api.opendota.com/api/matches/" + match_id 
    data = urllib.request.urlopen(url).read()
    
    
    # TODO make a new file where it's possible to take a subset
    # of the fields within the detailed match data...
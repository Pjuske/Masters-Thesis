import ast
import csv
import numpy as np
import matplotlib.pyplot as plt




def main():
  # Read detailed match data
  with open('datasets/2_detailed_match_data_without_blanks.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';', quotechar='|')
    next(reader, None)
    
    with open("./%s" % ('datasets/9_gold_exp_advantage_matrix.csv'), 'w', newline='',  encoding="utf-8") as outf:
      # Write header into csv file    
      header = ['radiant_win', 'minute', 'radiant_gold_adv', 'radiant_xp_adv']
      dw = csv.DictWriter(outf, dict.fromkeys(header).keys(), delimiter=',')
      dw.writeheader()
    
      for row in reader:
        # Get radiant win, radiant gold advantage, and radiant exp advantage
        radiant_win = 1 if row[0] == 'TRUE' else 0
        gold_adv    = ast.literal_eval(row[10])
        exp_adv     = ast.literal_eval(row[11])
        
        for i in range(len(gold_adv)):
          my_dict={}
          my_dict['radiant_win'] = radiant_win
          my_dict['minute'] = i
          my_dict['radiant_gold_adv'] = gold_adv[i]
          my_dict['radiant_xp_adv']   = exp_adv[i]
          
          dw.writerow(my_dict) 
        
main()
from utils import *
from matplotlib import pyplot as plt
team_file = getNewDataset()
multi = 0
listErrorBar=['results/ErrorBar-K5.xlsx','results/ErrorBar-K10.xlsx']
for num in range(len(listErrorBar)):
   list_charts =[]
   for x in range(len(team_file.listFileTest) + 1):
      print(x)
      if (x != 0):
         tmp_name = team_file.train
         team_file.train = team_file.listFileTest[x - 1]
         team_file.listFileTest[x - 1] = tmp_name
         list_charts.append(error_bars(listErrorBar[num],fileToNameCustom(team_file.train)))
      else:
         list_charts.append(error_bars(listErrorBar[num],fileToNameCustom(team_file.train)))
#    multiple_charts(list_charts,"ErrorBar_"+str(num))
#    list_charts = []

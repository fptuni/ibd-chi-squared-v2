from utils import *

print("========= Stimulates Start =========")
print("Algorithms use: ")
print("Chi-Squared")
team_file = getNewDataset()
import time
import xlsxwriter
#Set up here
num_feats = [30]
kfold = [5,10]
nTimes = [10]
# folder_name ="Got-" +str(len(num_feats*len(kfold*len(nTimes)))) +  time.strftime("-results-%Hh%Mm%Ss-%d%B%Y/")
folder_name = "results/"
import os
import shutil
# define the name of the directory to be created
path = folder_name
try:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
resultsSetForErrorBars_TrainAll = ["Train"]
resultsSetForErrorBarsAll = ["Performance"]
resultsSetForErrorBars_TestResultAll = ["Test"]
resultsSetForErrorBars_MetricsAll = ["Metrics"]
for var_times in range(len(nTimes)):
    for var_k_fold in range(len(kfold)):
        for var_num_feats in range(len(num_feats)):
            file_excel_name =folder_name+"TableData-Run" + str(nTimes[var_times]) + "Times-" + str(num_feats[var_num_feats]) + "Feats-Fold" + str(kfold[var_k_fold])+"-" + time.strftime(
                "%Hh%Mm%Ss-%d%B%Y") + ".xlsx"
            workbook = xlsxwriter.Workbook(file_excel_name)
            for x in range(len(team_file.listFileTest) + 1):
                if (x != 0):
                    tmp_name = team_file.train
                    team_file.train = team_file.listFileTest[x - 1]
                    team_file.listFileTest[x - 1] = tmp_name
                    resultsSetForErrorBars, resultsSetForErrorBars_TestResult, resultsSetForErrorBars_Metrics, resultsSetForErrorBars_Train = subteam2(team_file.train, team_file.resultColName,
                             team_file.listFileTest, nTimes[var_times], num_feats[var_num_feats], kfold[var_k_fold], workbook)
                    for x in range(len(resultsSetForErrorBars)):
                        resultsSetForErrorBarsAll.append(resultsSetForErrorBars[x])
                    for x in range(len(resultsSetForErrorBars_TestResult)):
                        resultsSetForErrorBars_TestResultAll.append(resultsSetForErrorBars_TestResult[x])
                    for x in range(len(resultsSetForErrorBars_Metrics)):
                        resultsSetForErrorBars_MetricsAll.append(resultsSetForErrorBars_Metrics[x])
                    for x in range(len(resultsSetForErrorBars_Train)):
                        resultsSetForErrorBars_TrainAll.append(resultsSetForErrorBars_Train[x])
                else:
                    resultsSetForErrorBars,resultsSetForErrorBars_TestResult,resultsSetForErrorBars_Metrics,resultsSetForErrorBars_Train = subteam2(team_file.train, team_file.resultColName,
                             team_file.listFileTest, nTimes[var_times], num_feats[var_num_feats], kfold[var_k_fold], workbook)
                    for x in range(len(resultsSetForErrorBars)):
                        resultsSetForErrorBarsAll.append(resultsSetForErrorBars[x])
                    for x in range(len(resultsSetForErrorBars_TestResult)):
                        resultsSetForErrorBars_TestResultAll.append(resultsSetForErrorBars_TestResult[x])
                    for x in range(len(resultsSetForErrorBars_Metrics)):
                        resultsSetForErrorBars_MetricsAll.append(resultsSetForErrorBars_Metrics[x])
                    for x in range(len(resultsSetForErrorBars_Train)):
                        resultsSetForErrorBars_TrainAll.append(resultsSetForErrorBars_Train[x])
            workbook.close()
            print('Data is written to excel File.')
        file_excel_name_errorbars = folder_name +"ErrorBar-K"+str(kfold[var_k_fold]) + ".xlsx"
        exel_exports_error_bars_format(resultsSetForErrorBarsAll, resultsSetForErrorBars_TestResultAll,
                                       resultsSetForErrorBars_MetricsAll, resultsSetForErrorBars_TrainAll,file_excel_name_errorbars)
        resultsSetForErrorBars_TrainAll = ["Train"]
        resultsSetForErrorBarsAll = ["Performance"]
        resultsSetForErrorBars_TestResultAll = ["Test"]
        resultsSetForErrorBars_MetricsAll = ["Metrics"]

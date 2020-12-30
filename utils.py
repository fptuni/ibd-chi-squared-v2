import os

import pandas as pd
import xlsxwriter
from mpmath import *
from numpy.random import default_rng
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from utils import *
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

# creating the DataFrame
def exportResults(file_train_name, list_test_file_name, result_list_in_orders, workbook):
    number_of_rows = len(result_list_in_orders)
    number_of_test = len(list_test_file_name)
    x = file_train_name.split('/')
    file_train_name_sheet = x[len(x) - 1].split('.')[0]
    if (len(list_test_file_name) != 5):
        print("Sorry! We are not implement auto numbber of tex")
    worksheet = workbook.add_worksheet(name=file_train_name_sheet)
    # Increase the cell size of the merged cells to highlight the formatting.
    worksheet.set_column('A:C')
    worksheet.set_column('A:B', 60)
    worksheet.set_column('C:C', 40)
    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})
    merge_format_2 = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'font_color': 'white',
        'valign': 'vcenter',
        'fg_color': 'green'})
    merge_format_3 = workbook.add_format({
        'font_color': 'green'
    })
    merge_format_4 = workbook.add_format({
        'bottom': 1,
        'left': 1
    })
    merge_format_2.set_text_wrap()
    merge_format.set_text_wrap()
    # Merge 3 cells.
    worksheet.merge_range('A1:A' + str(number_of_rows), "Train by " + file_train_name, merge_format_2)
    # Merge 3 cells over two rows.
    count_test = 0
    for x in range(number_of_test):
        first_row = x * 9 + 1
        last_row = x * 9 + 9
        worksheet.conditional_format('C' + str(last_row) + ':C' + str(last_row),
                                     {'type': 'no_blanks', 'format': merge_format_4})
        if (count_test <= (len(list_test_file_name) - 1)):
            worksheet.merge_range('B' + str(first_row) + ':B' + str(last_row),
                                  "Test on " + list_test_file_name[count_test], merge_format)
            count_test += 1

    for x in range(number_of_rows):
        if "Random" in result_list_in_orders[x]:
            worksheet.write('C' + str(x + 1), result_list_in_orders[x])
        else:
            worksheet.write('C' + str(x + 1), result_list_in_orders[x], merge_format_3)


class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName


dirname = os.path.dirname(__file__)


def getNewDataset():
    train = os.path.join(dirname, 'data/ibdfullHS_CDr_x.csv')  # iCDr & UCf &iCDf &CDr&CDf ibdfullHS_iCDf_x
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCr_x.csv'))
    return TeamFile(train, fileListTest, "RS")


def find_if_chi_squared(resultColName, filenameTrain, num_feats):
    print("Feature selection method to be used : " + "Chi-Squared")
    print(str("Train by file ") + str(filenameTrain))
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    X = df[colName]
    y = df[resultColName]
    X_No_First = df.drop(df.columns[0], axis=1)
    X_Data = X_No_First.drop(labels='RS', axis=1)
    X_norm = MinMaxScaler().fit_transform(X_Data)
    print("num_feats = " + str(num_feats))
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X_Data.loc[:, chi_support].columns.tolist()
    importanceFeature = chi_feature
    print("Number feature selected : " + str(len(importanceFeature)))
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    return importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature


def findRandomeFeaturesSets(resultColName, filenameTrain, sizeIF):
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    y = df[resultColName]
    X_No_First = df.drop(df.columns[0], axis=1)
    X_Data = X_No_First.drop(labels='RS', axis=1)
    rng = default_rng()
    # In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
    #len(colName) - 2, , replace=False
    numbers = rng.choice(len(X_Data.columns),size=sizeIF)
    randomeFeatureSameSize = X_Data.columns.take(numbers)
    X_Train_Random = X_Data[randomeFeatureSameSize]
    y_Train_Random = y
    return randomeFeatureSameSize, X_Train_Random, y_Train_Random


def printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, acc_svm, mcc_svm, auc_svm,nTimes):
    print("When do Random ")
    print("ACC = " + str(acc_random / nTimes))
    print("MCC = " + str(mcc_random / nTimes))
    print("AUC = " + str(auc_random / nTimes))
    print("+++++ ")
    print("When we got Importance Features using Chi-Squared")
    print("ACC = " + str(acc_if / nTimes))
    print("MCC = " + str(mcc_if / nTimes))
    print("AUC = " + str(auc_if / nTimes))
    print("+++++ ")
    print("When we got Importance Features using SVM")
    print("ACC = " + str(acc_svm / nTimes))
    print("MCC = " + str(mcc_svm / nTimes))
    print("AUC = " + str(auc_svm / nTimes))
    print("--------------------------------- ")


def sumThenAveragePercisely(accuracy_model_acc):
    return fdiv(fsum(accuracy_model_acc), len(accuracy_model_acc), prec=5)


# Train on one dataset, then test of another dataset
def find_if_using_svm(resultColName, filenameTrain, num_feats):
    print("Feature selection method to be used : " + "SVM")
    print(str("Train by file ") + str(filenameTrain))
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    X = df[colName]
    y = df[resultColName]
    X_No_First = df.drop(df.columns[0], axis=1)
    X_Data = X_No_First.drop(labels='RS', axis=1)
    X_norm = MinMaxScaler().fit_transform(X_Data)
    print("num_feats = " + str(num_feats))
    model = svm.SVC(kernel='linear')
    model.fit(X_norm, y)
    dfscores = pd.DataFrame(model.coef_)
    dfscores= dfscores.T
    dfcolumns = pd.DataFrame(X_Data.columns)
    # concat two dataframes for better visualization
    relevant_features = pd.concat([dfcolumns, dfscores], axis=1)
    relevant_features.columns = ['Specs', 'Score']
    importanceFeature = relevant_features.nlargest(num_feats, 'Score')
    importanceFeature = importanceFeature.drop('Score', 1)
    importanceFeature = importanceFeature.iloc[:, -1]
    # chi_feature = X_No_First.loc[:, model.get_support()].columns.tolist()
    # importanceFeature = chi_feature
    # print("Number feature selected : " + str(len(importanceFeature)))
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    return importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature


def subteam2(filenameTrain, resultColName, fileListTest, nTimes, nlargestFeatures, kfold, workbook):
    importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature = find_if_chi_squared(resultColName,
                                                                                          filenameTrain,
                                                                                          nlargestFeatures)
    svmFeature, x_train_svm, y_train_svm = find_if_using_svm(resultColName, filenameTrain,
                                                                                     len(importanceFeature))
    randomeFeatureSameSize, X_Train_Random, y_Train_Random = findRandomeFeaturesSets(resultColName, filenameTrain,
                                                                                     len(importanceFeature))
    print("Results begin ----------------- ")
    resultSetForExcel = []
    vars_acc = []
    vars_mcc = []
    vars_auc = []
    acc_random_nTimes = 0.0
    mcc_random_nTimes = 0.0
    auc_random_nTimes = 0.0
    acc_if_nTimes = 0.0
    mcc_if_nTimes = 0.0
    auc_if_nTimes = 0.0
    acc_svm_nTimes = 0.0
    mcc_svm_nTimes = 0.0
    auc_svm_nTimes = 0.0
    resultsSetForErrorBars = []
    resultsSetForErrorBars_TestResult = []
    resultsSetForErrorBars_Metrics = []
    resultsSetForErrorBars_Train = []
    for x in range(len(fileListTest)):
        print("Run test on the file name " + fileListTest[x])
        for n in range(nTimes):
            print("Run time " + str((n + 1)) + " time")
            if nTimes == 0:
                break
            # Get data from file to test
            data_yu = pd.read_csv(fileListTest[x])

            # Get the test data same column name with Random for compare
            df_Test = pd.DataFrame(data_yu, columns=randomeFeatureSameSize).fillna(0)
            X_Test_Random = df_Test[randomeFeatureSameSize]
            y_Test_Random = data_yu[resultColName]
            # Train with method Random
            clfRandom = RandomForestClassifier()
            cvRandom = StratifiedKFold(n_splits=kfold)
            for (train, test), i in zip(cvRandom.split(X_Test_Random, y_Test_Random), range(5)):
                clfRandom.fit(X_Test_Random.iloc[train], y_Test_Random.iloc[train])
                y_Predict_Random = clfRandom.predict(X_Test_Random.iloc[test])
                vars_acc.append(metrics.accuracy_score(y_Test_Random.iloc[test], y_Predict_Random))
                vars_mcc.append(metrics.matthews_corrcoef(y_Test_Random.iloc[test], y_Predict_Random))
                vars_auc.append(metrics.roc_auc_score(y_Test_Random.iloc[test], y_Predict_Random))
            acc_random = sum(vars_acc) / len(vars_acc)
            mcc_random = sum(vars_mcc) / len(vars_mcc)
            auc_random = sum(vars_auc) / len(vars_auc)
            # Get the test data same column name with SVM
            df_Test = pd.DataFrame(data_yu, columns=svmFeature).fillna(0)
            X_Test_svm = df_Test[svmFeature]
            y_Test_svm = data_yu[resultColName]
            # Train with method Random
            vars_acc = []
            vars_mcc = []
            vars_auc = []
            clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            Pipeline(steps=[('standardscaler', StandardScaler()),
                            ('svc', SVC(gamma='auto'))])
            cv_svm = StratifiedKFold(n_splits=kfold)
            for (train, test), i in zip(cv_svm.split(X_Test_svm , y_Test_svm), range(5)):
                clf_svm.fit(X_Test_svm.iloc[train], y_Test_svm.iloc[train])
                y_Predict_svm = clf_svm.predict(X_Test_svm.iloc[test])
                vars_acc.append(metrics.accuracy_score(y_Test_svm.iloc[test], y_Predict_svm))
                vars_mcc.append(metrics.matthews_corrcoef(y_Test_svm.iloc[test], y_Predict_svm))
                vars_auc.append(metrics.roc_auc_score(y_Test_svm.iloc[test], y_Predict_svm))
            acc_svm = sum(vars_acc) / len(vars_acc)
            mcc_svm = sum(vars_mcc) / len(vars_mcc)
            auc_svm = sum(vars_auc) / len(vars_auc)
            vars_acc = []
            vars_mcc = []
            vars_auc = []
            df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
            X_Test_IF = df_IF[importanceFeature]
            y_Test_IF = data_yu[resultColName]
            # Train with method FS
            clf = RandomForestClassifier()
            cv = StratifiedKFold(n_splits=kfold)

            for (train, test), i in zip(cv.split(X_Test_IF, y_Test_IF), range(5)):
                clf.fit(X_Test_IF.iloc[train], y_Test_IF.iloc[train])
                y_Predict_IF = clf.predict(X_Test_IF.iloc[test])
                vars_acc.append(metrics.accuracy_score(y_Test_IF.iloc[test], y_Predict_IF))
                vars_mcc.append(metrics.matthews_corrcoef(y_Test_IF.iloc[test], y_Predict_IF))
                vars_auc.append(metrics.roc_auc_score(y_Test_IF.iloc[test], y_Predict_IF))
            acc_if = sum(vars_acc) / len(vars_acc)
            mcc_if = sum(vars_mcc) / len(vars_mcc)
            auc_if = sum(vars_auc) / len(vars_auc)

            acc_random_nTimes += acc_random
            mcc_random_nTimes += mcc_random
            auc_random_nTimes += auc_random
            acc_if_nTimes += acc_if
            mcc_if_nTimes += mcc_if
            auc_if_nTimes += auc_if
            acc_svm_nTimes += acc_svm
            mcc_svm_nTimes += mcc_svm
            auc_svm_nTimes += auc_svm
            if nTimes == 0:
                break
        printResult(acc_random_nTimes, mcc_random_nTimes, auc_random_nTimes, acc_if_nTimes, mcc_if_nTimes,
                    auc_if_nTimes,acc_svm_nTimes, mcc_svm_nTimes, auc_svm_nTimes, nTimes)
        resultSetForExcel.append("Chi-Squared ACC= " + str(round(acc_if_nTimes / nTimes,5)))
        resultSetForExcel.append("Chi-Squared MCC= " + str(round(mcc_if_nTimes / nTimes,5)))
        resultSetForExcel.append("Chi-Squared AUC= " + str(round(auc_if_nTimes / nTimes,5)))
        resultSetForExcel.append("Random ACC = " + str(round(acc_random_nTimes / nTimes,5)))
        resultSetForExcel.append("Random MCC = " + str(round(mcc_random_nTimes / nTimes,5)))
        resultSetForExcel.append("Random AUC = " + str(round(auc_random_nTimes / nTimes,5)))
        resultSetForExcel.append("SVM ACC= " + str(round(acc_svm_nTimes / nTimes,5)))
        resultSetForExcel.append("SVM MCC= " + str(round(mcc_svm_nTimes / nTimes,5)))
        resultSetForExcel.append("SVM AUC= " + str(round(auc_svm_nTimes / nTimes,5)))
        # Begin data for  --- Error bar excels
        # Random
        resultsSetForErrorBars.append(str(round(acc_random_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(mcc_random_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(auc_random_nTimes / nTimes,5)))
        # IF
        resultsSetForErrorBars.append(str(round(acc_if_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(mcc_if_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(auc_if_nTimes / nTimes,5)))
        # SVM
        resultsSetForErrorBars.append(str(round(acc_svm_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(mcc_svm_nTimes / nTimes,5)))
        resultsSetForErrorBars.append(str(round(auc_svm_nTimes / nTimes,5)))

        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))
        resultsSetForErrorBars_TestResult.append(fileToNameCustom(fileListTest[x]))

        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))
        resultsSetForErrorBars_Train.append(fileToNameCustom(filenameTrain))

        resultsSetForErrorBars_Metrics.append("rACC")
        resultsSetForErrorBars_Metrics.append("rMCC")
        resultsSetForErrorBars_Metrics.append("rAUC")
        resultsSetForErrorBars_Metrics.append("ACC")
        resultsSetForErrorBars_Metrics.append("MCC")
        resultsSetForErrorBars_Metrics.append("AUC")
        resultsSetForErrorBars_Metrics.append("sACC")
        resultsSetForErrorBars_Metrics.append("sMCC")
        resultsSetForErrorBars_Metrics.append("sAUC")
        # End data for  --- Error bar excels
        acc_if_nTimes = 0.0
        mcc_if_nTimes = 0.0
        auc_if_nTimes = 0.0
        acc_random_nTimes = 0.0
        mcc_random_nTimes = 0.0
        auc_random_nTimes = 0.0
        acc_svm_nTimes = 0.0
        mcc_svm_nTimes = 0.0
        auc_svm_nTimes = 0.0
    exportResults(filenameTrain, fileListTest, resultSetForExcel, workbook)
    return resultsSetForErrorBars, resultsSetForErrorBars_TestResult, resultsSetForErrorBars_Metrics, resultsSetForErrorBars_Train


def exel_exports_error_bars_format(resultsSetForErrorBars, resultsSetForErrorBars_TestResult,
                                   resultsSetForErrorBars_Metrics, resultsSetForErrorBars_Train,
                                   file_excel_name_errorbars):
    workbook_eb = xlsxwriter.Workbook(file_excel_name_errorbars)
    pd.ExcelWriter(file_excel_name_errorbars, engine='xlsxwriter', options={'strings_to_formulas': False})
    worksheet_eb = workbook_eb.add_worksheet("Data for Error Bars")
    worksheet_eb.add_table('A1:D' + str(len(resultsSetForErrorBars)))
    print('A1:D' + str(len(resultsSetForErrorBars)), {'header_row': True}, {'autofilter': False},
          {'banded_rows': False})
    my_list = np.hstack([[resultsSetForErrorBars_Train, resultsSetForErrorBars, resultsSetForErrorBars_TestResult,
                          resultsSetForErrorBars_Metrics]])
    for col_num, col_data in enumerate(my_list):
        for row_num, row_data in enumerate(col_data):
            worksheet_eb.write(row_num, col_num, row_data)
    workbook_eb.close()

def fileToNameCustom(name):
    x = name.split('/')
    nameWithout = x[len(x) - 1].split('.')[0]
    y = nameWithout.split('_')
    return y[1]


def error_bars(fileResultsName, fileTrainName):
    x = fileResultsName.split('-')
    nameWithout = x[len(x) - 1].split('.')[0]
    data = pd.read_excel(fileResultsName, engine='openpyxl')
    df = pd.DataFrame(data)
    myFrame = pd.DataFrame(data=[], columns=["Train", "Performance", "Test", "Metrics"])
    for x in range(len(df.index)):
        if fileTrainName == str(df[x:x + 1]["Train"][x]):
            df2 = pd.DataFrame([[df[x:x + 1]["Train"][x], df[x:x + 1]["Performance"][x], df[x:x + 1]["Test"][x],
                                 df[x:x + 1]["Metrics"][x]]], columns=["Train", "Performance", "Test", "Metrics"])
            myFrame = myFrame.append(df2, ignore_index=True, sort=False)
    if len(myFrame.index) > 0:
        df_prices = myFrame.groupby(['Metrics']).agg([np.mean, np.std])
        df_half_if = df_prices.iloc[0:3, :]
        df_half_random = df_prices.iloc[3:6, :]
        df_half_svm = df_prices.iloc[6:9, :]
        rd_half = df_half_random["Performance"]
        if_half = df_half_if["Performance"]
        svm_half = df_half_svm["Performance"]
        x_pos_a = [0,1,2]
        x_pos_b = [4,5,6]
        x_pos_c = [8,9,10]
        xpos = [x_pos_a, x_pos_b, x_pos_c]
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 4.67)
        for i in range(len(xpos)):
            autolabel(ax.bar(xpos[i][0], if_half["mean"][i],
                             yerr=if_half["std"][i],
                             align='center',
                             alpha=0.5,
                             color='orange',
                             ecolor='black',
                             capsize=15), ax, if_half["std"][i], round(if_half["mean"][i], 3))
            autolabel(ax.bar(xpos[i][1], rd_half["mean"][i],
                             yerr=rd_half["std"][i],
                             align='center',
                             alpha=0.5,
                             color='#002fff',
                             ecolor='black',
                             capsize=15), ax, rd_half["std"][i], round(rd_half["mean"][i], 3))
            autolabel(ax.bar(xpos[i][2], svm_half["mean"][i],
                             yerr=svm_half["std"][i],
                             align='center',
                             alpha=0.5,
                             color='green',
                             ecolor='black',
                             capsize=15), ax, svm_half["std"][i], round(svm_half["mean"][i], 3))
        ax.set_ylabel('Accuracy value', fontweight='bold')
        ax.set_xticks([0, 1, 2, 4, 5,6,8,9,10])
        labels = ["ACC", "ACC","ACC", "MCC", "MCC","MCC", "AUC", "AUC", "AUC"]
        ax.set_xticklabels(labels)
        # if nameWithout == "K5":
        #     # running 5-fold cross validation repeated 10 times
        #     ax.set_title('Train by file: ' + fileTrainName + " and running 5-fold cross validation repeated 10 times.")
        # else:
        #     ax.set_title('Train by file: ' + fileTrainName + " and running 10-fold cross validation repeated 10 times.")
        ax.yaxis.grid(False)
        # Save the figure and show
        plt.xlabel('Metrics', fontweight='bold')
        from matplotlib.patches import Patch
        custom_lines = [Patch(facecolor='orange', edgecolor='orange', alpha=0.5,
                              label='Color Patch'),
                        Patch(facecolor='blue', edgecolor='blue', alpha=0.5,
                              label='Color Patch'),
                        Patch(facecolor='green', edgecolor='green', alpha=0.5,
                              label='Color Patch')
                        ]
        plt.legend(custom_lines, ['Chi-square-based feature selection', 'Random selection','SVM-based feature selection'],loc='upper right', bbox_to_anchor=(0.5, 0.5))
        plt.tight_layout()
        plt.savefig('results/'+ nameWithout + '_errbar_' + fileTrainName + '.png')
        plt.show()

def autolabel(rects, ax, text, value_display):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(value_display),
                    xy=(rect.get_x() + rect.get_width() / 2, height + text - 0.015),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

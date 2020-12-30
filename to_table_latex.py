from utils import *

def to_table_latex_from_excel(filename):
    data = pd.read_excel( filename,sheet_name = 5, engine='openpyxl')
    df = pd.DataFrame(data)
    trainName = fileToNameCustom(str(df.iloc[:, 0:1].columns))
    print("Train name is " +trainName)
    firstValue = str(pd.DataFrame(data).iloc[:,2:3].columns).split("= ")[1]
    totalValuesList =[]
    firstValue = firstValue.split("']")[0]
    totalValuesList.append(firstValue)
    for x in range(len(pd.DataFrame(data).iloc[:,2:3].values)):
        varValue = str(pd.DataFrame(data).iloc[:, 2:3].values[x]).split("= ")[1]
        varValue = varValue.split("']")[0]
        totalValuesList.append(varValue)
    totalValuesList = totalValuesList
    listTestName = []
    x = 0
    listTestName.append(fileToNameCustom(str(df.iloc[:,1:2].columns)))
    listTestName.append(fileToNameCustom(str(df.iloc[:,1:2].values[x + 8])))
    listTestName.append(fileToNameCustom(str(df.iloc[:, 1:2].values[x + 17])))
    listTestName.append(fileToNameCustom(str(df.iloc[:, 1:2].values[x + 26])))
    listTestName.append(fileToNameCustom(str(df.iloc[:, 1:2].values[x + 35])))
    initString = ""
    for x in range(len(listTestName)):
        initString+=("&"+listTestName[x]+" ")
    print(initString)
    for x in range(9):
        if x == 0 :
            print("Chi\n")
            initString ="& ACC & "+totalValuesList[x]+" & "+str(totalValuesList[x+9])+" & "+str(totalValuesList[x+18])+" & "+str(totalValuesList[x+27])+" & "+str(totalValuesList[x+36])+"\\\\"
            print(str(initString))
        if x == 1 :
            initString ="& & MCC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x ==2 :
            initString ="& & AUC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x == 3 :
            print("Random\n")
            initString ="& ACC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x == 4 :
            initString ="& & MCC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x ==5 :
            initString ="& & AUC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x == 6 :
            print("Svm\n")
            initString ="& ACC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x == 7 :
            initString ="& & MCC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
        if x ==8 :
            initString ="& & AUC & "+totalValuesList[x]+" & "+totalValuesList[x+9]+" & "+totalValuesList[x+18]+" & "+totalValuesList[x+27]+" & "+totalValuesList[x+36]+"\\\\"
            print(str(initString))
    # nValues =


to_table_latex_from_excel("results/TableData-Run10Times-30Feats-Fold10-15h45m26s-30December2020.xlsx")
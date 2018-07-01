# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''


#Module importieren
import pandas as pd
import scipy.stats as stats
import lightgbm
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from bi2018.BI.data_handler import DataHandler



def main():
    #Hier werden alle verschiedenen Methoden aufgerufen, da es sonst wirklich ziemlich unübersichtlich wird
    #Einlesen des Files
    df = readF("train.csv", True) # True wenn Index im File vorhanden, wie hier.
    #df1 = MultiColumnLabelEncoder().fit_transform(df)
    test = readF('test.csv', False)
    #test1 = MultiColumnLabelEncoder().fit_transform(test)
    #with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        #print (df1)
        #print (test)
    #cT = ChiSquare(df) #
    #useChi(cT) #gibt aus, welche Columns "important" sind für "Category"; DESCRIPT is most important
    dh = DataHandler()
    #xf= ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    
    dh.load_data(train=df, test=test)
    data_sets = dh.transform_data()
    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        print(data_sets)
        #exit()
    
    resulttrain= lgbm(data_sets)
    print(resulttrain)
    exit()


    #resulttrain = train(df1, test1)
    #print (resulttrain)


#Data Understanding & Data Preparation von BI_martin.py, dort wird von train.csv die csv "rewritten.csv" erstellt, und hier wieder eingelesen zur Auswertung.
def readF(path, index): #index == True, wenn Index vorhanden
    print('Reading: ', path)
    if (index == True):
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, dtype={"AddressSuffix": str, 'X': float, 'Y': float}) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    else:
        #df = pd.read_csv(path, header = 0, sep='\t' )
        #probably not needed anymore since bi_martin is fixed
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, dtype={"AddressSuffix": str, 'X': float, 'Y': float}, index_col=0) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    print('Transforming', path)
    #df['Date'], df['Time'] = df['Dates'].str.split(' ', 1).str
    df['Year'] = df['Dates'].str[:4]
    df['Month'] = df['Dates'].str[5:7]
    df['Day'] = df['Dates'].str[8:10]
    df['Time'] = df['Dates'].str[11:16]
    df['Season'] = df.apply(get_season, axis=1)
    #Note the axis=1 specifier, that means that the application is done at a row, rather than a column level.
    #df['AddressSuffix'] = df['Address'].str[-2:]
    df['DayOfWeek'] = df['DayOfWeek'].str.upper()
    #df['Address'] = df['Address'].str.upper()
    df['X'] = df['X'].apply(lambda x: 0 if float(x)>=-122.3649 or float(x)<=-122.5136 else x)
    df['Y'] = df['Y'].apply(lambda y: 0 if float(y)<=37.70788 or float(y)>=37.81998 else y)
    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        print (df)    
        df = df.drop('Dates', 1)
        df = df.drop('Address', 1)
        if (path == 'train.csv'):
            df = df.drop('Descript', 1)
            df = df.drop('Resolution', 1)
        print (df)
    print('Success for ', path)
    
    #with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        #print(df.ix[257059]) # --> Einige Zeilen sind abgeschnitten und ergeben nicht immer viel Sinn. So wie diese hier; Excel index + 2 = Python,,, index 257061 = 257059
        #print(df)
        # Abfrage für bestimmten Wert "NONE" in Spalte "Resolution"
        #print(output.loc[output['Resolution'] == 'NONE'])
        #Entfernt alle Einträge "NONE" aus der Spalte "Resolution"
        #print("Hier werden die zu löschenden Inhalte ausgegeben.")
        #print(df.loc[~(df['Resolution'] != 'NONE')])
        #Will suchen nach 'OWNING' im Feld 'Descript'; um das zu tun müssen ggf. Descript Felder in Liste umgewandelt werden. oider einzelnd in CSV ausgelesen werden
        #print(df.loc[output['Descript'].isin('OWNING')])
        #Viele kompakte leicht zu verstehende Informationen auf Code Basis sind hier zu finden -v
        #further use: https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
        #existieren duplicates?
        #print (output.duplicated(subset='Dates', keep=False)) #Keep=False markiert alle Duplikate als True, keep=first, nur den ersten nicht
        #Gebe den Dataframe zurück, da wir nun alle Daten in der CSV wie gewünscht bearbeitet haben
    return df

def get_season(row):
    if 3 <= int(row['Dates'][5:7]) <= 5:
        return "SPRING"
    elif 6 <= int(row['Dates'][5:7]) <= 8:
        return "SUMMER"
    elif 9 <= int(row['Dates'][5:7]) <= 11:
        return "AUTUMN"
    else: return "WINTER"
    
#Entnommen aus: https://stackoverflow.com/a/30267328
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

"""
Feature Extraction
Feature Extraction mit ChiSquare Test, welcher Wert nimmt am meisten Einfluß wenn Null Hypothese gilt
Chi-Square Erklärung 5-min YouTube: https://www.youtube.com/watch?v=VskmMgXmkMQ ;; Besser: https://www.youtube.com/watch?v=WXPBoFDqNVk (12 min)
Quelle: http://www.handsonmachinelearning.com/blog/2AeuRL/chi-square-feature-selection-in-python
"""
class ChiSquare: #Erstellen von chisquare-Klasse um Werte zu speichern
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        self.dfTabular = None
        self.dfExpected = None


    #  alpha is der Wert, der zur Bestimmung ob Null Hypothese angewendet zutrifft oder nicht
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha: # If P is low, Ho (null hypothesis) must go...
            result="{} is IMPORTANT for Prediction. Value of chi2 {}".format(colX, self.chi2)
        else:
            result="{} is NOT an important predictor. Value of chi2 {}".format(colX, self.chi2)

        print(result)

    def TestIndependence(self,colX,colY, alpha=0.001): #changed from 0.05 to 0.001 --> calculated by ML soße: "alpha_range = 10.0**-np.arange(1,7)" ändert Outcome aber NICHT
        X = self.df[colX].astype(str) #Konvertierung zu String der unabhängigen Features
        Y = self.df[colY].astype(str) #Konvertierung zu String des abhängigen Features

        self.dfObserved = pd.crosstab(Y,X) #Anzahl für Observed in Abhängigkeit von Resolution
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof
        #print("Observed")
        #print(self.dfObserved)

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        #print("Expected")
        #print(self.dfExpected)

        self._print_chisquare_result(colX, alpha)



#Feature Selection
def useChi(cT):
    testColumns = ["Date", "Time", "Year", "Month", "Day", "Hour", "Season","Descript","DayOfWeek","PdDistrict", "Resolution", "Address", "AddressSuffix", "X", "Y"]
    for var in testColumns: #Für jede einzelne Column wird  Chi-Square ausgeführt
        cT.TestIndependence(colX=var,colY="Category") #Aufruf des Chi-Square Test mit Resolution als abhängiges Features

def lgbm(data_set):
    params = {}
    params['task'] = 'train'
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'goss'
    params['objective'] = 'multiclass'
    params['numclass'] = '38'
    params['metric'] = 'multi_logloss'
    #params['sub_feature'] = 0.5
    #params['num_leaves'] = 10
    params['min_data'] = 5000
    #params['max_depth'] = 10
    """   'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0"""

    #print("*************************hallo")
    #print(df[df.columns[1]])

    x_train = data_set['train_X']
    y_train = data_set['train_Y']
    x_test = data_set['test_X']
    #y_train = train.iloc[0].values
    #x_train = train.drop(0).values
    #print(x_train)
    #x_train = df.drop(0, axis=1).values

    #http://lightgbm.readthedocs.io/en/latest/Python-Intro.html - how it should work
    #print ("data", np.random.rand(500,10))
    #print ("label", np.random.randint(2, size=500))
    #print('y')
    #print(y_train)
    #print('x')
    #print(x_train)
    #exit() #cuz it doesnt work yet
    lgb_train = lightgbm.Dataset(x_train, y_train)
    lgb_eval = lightgbm.Dataset(x_test, reference=lgb_train)

    #sc = StandardScaler()
    #ds = sc.fit_transform(X=df, y=None)

    clf = lightgbm.train(params, lgb_train, 100, valid_sets=lgb_eval)
    print(clf)
    exit()
    return clf





#Aufrufen der Ausführung, bitte ganz unten
main()

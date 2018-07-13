# -*- coding: utf-8 -*-
'''
@author: TheUniverse
'''



#Module importieren
import pandas as pd
import scipy.stats as stats
import lightgbm
import os
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#os.environ["PATH"] += os.pathsep + 'H:/workspace coding/graphviz/bin'

'''
Hi, My name is Kevin. If ure reading this, youre probably reading it on your screen~
garphviz: 


https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
https://graphviz.gitlab.io/_pages/Download/Download_windows.html
windowstaste --> systemumgebungsvariablen bearbeiten
https://imgur.com/a/tZHiAlb

'''


class DataHandler:

    training_data = None
    testing_data = None

    category_mapping = {"ARSON": 0,
    "ASSAULT": 1,
    "BAD CHECKS": 2,
    "BRIBERY": 3,
    "BURGLARY": 4,
    "DISORDERLY CONDUCT": 5,
    "DRIVING UNDER THE INFLUENCE": 6,
    "DRUG/NARCOTIC": 7,
    "DRUNKENNESS": 8,
    "EMBEZZLEMENT": 9,
    "EXTORTION": 10,
    "FAMILY OFFENSES": 11,
    "FORGERY/COUNTERFEITING": 12,
    "FRAUD": 13,
    "GAMBLING": 14,
    "KIDNAPPING": 15,
    "LARCENY/THEFT": 16,
    "LIQUOR LAWS": 17,
    "LOITERING": 18,
    "MISSING PERSON": 19,
    "NON-CRIMINAL": 20,
    "OTHER OFFENSES": 21,
    "PORNOGRAPHY/OBSCENE MAT": 22,
    "PROSTITUTION": 23,
    "RECOVERED VEHICLE": 24,
    "ROBBERY": 25,
    "RUNAWAY": 26,
    "SECONDARY CODES": 27,
    "SEX OFFENSES FORCIBLE": 28,
    "SEX OFFENSES NON FORCIBLE": 29,
    "STOLEN PROPERTY": 30,
    "SUICIDE": 31,
    "SUSPICIOUS OCC": 32,
    "TREA": 33,
    "TRESPASS": 34,
    "VANDALISM": 35,
    "VEHICLE THEFT": 36,
    "WARRANTS": 37,
    "WEAPON LAWS": 38}

    def __init__(self):
        self.training_data = None
        self.testing_data = None


    def load_data(self, train, test):
        self.training_data = train
        self.testing_data = test

    #  <big_data> DataFrame erstellen um  feature preprocessing zu vereinfachen
    #  Ausgabe ist ein Dictionary wo train-Datensatz aufgeteilt wird.
    def transform_data(self, with_mask=1):
        features_columns = [ 'Year', 'Month', 'Hour', 'PdDistrict', 'X', 'Y']  #[ 'Year', 'Month', 'Day', 'Time', 'DayOfWeek', 'PdDistrict', 'X', 'Y']
        big_data = self.training_data[features_columns].append(self.testing_data[features_columns])

        categorical_features = ['Year', 'Month', 'Hour', 'PdDistrict']
        numerical_features = ['X', 'Y']


        print ('Encoding der Features')
        big_data = self.categorical_encoder(big_data, categorical_features)
        print ('Skalieren der Longitude und Latitude')
        big_data = self.features_preprocessing(big_data, numerical_features)
        train_X = big_data[0:self.training_data.shape[0]]
        train_Y = self.training_data['Category'].map(self.category_mapping)
        test_x = big_data[self.training_data.shape[0]::]

        if with_mask != 1:
            mask = np.random.rand(len(self.training_data)) < with_mask
            train_X = train_X[mask]
            train_Y = train_Y[mask]
        print ('Splitten des Datensatzes für Validierung bei Modelerstellung')
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)
        return {'x_train_split': x_train_split, 'y_train_split': y_train_split, 'x_test_split': x_test_split, 'y_test_split': y_test_split, 'test_x': test_x}


    def features_preprocessing(self, data, numerical_columns):
        for num_col in numerical_columns:
            data[num_col] = preprocessing.scale(data[num_col])
        return data

    def categorical_encoder(self, data, categorical_columns):
        le = LabelEncoder()
        for cat_col in categorical_columns:
            data[cat_col] = le.fit_transform(data[cat_col])
        return data



def main():

    df = readF("train.csv", True) # True wenn Index im File vorhanden, wie hier.
    test = readF('test.csv', False)
    dh = DataHandler()
    dh.load_data(train=df, test=test)
    data_sets = dh.transform_data()
    #  with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        #print(data_sets)
        #exit()

    resulttrain= lgbm(data_sets)
    print(resulttrain)
    exit()



def readF(path, index):
    print('Reading: ', path)
    if (index == True):
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, dtype={"AddressSuffix": str, 'X': float, 'Y': float}) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    else:
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, dtype={"AddressSuffix": str, 'X': float, 'Y': float}, index_col=0) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    print('Transforming', path)
    df['Year'] = df['Dates'].str[:4]
    df['Month'] = df['Dates'].str[5:7]
    df['Day'] = df['Dates'].str[8:10]
    df['Hour'] = df['Dates'].str[11:13]
    df['Season'] = df.apply(get_season, axis=1)

    df['DayOfWeek'] = df['DayOfWeek'].str.upper()
    #df['X'] = df['X'].apply(lambda x: 0 if float(x)>=-122.3649 or float(x)<=-122.5136 else x)#
    df['X'] = df['X'].apply(lambda x: 0 if float(x)>-122.3649 or float(x)<-122.513642064265 else x)
    df['Y'] = df['Y'].apply(lambda y: 0 if float(y)<37.70788 or float(y)>37.81998 else y) #verzieht die Auswertung über Chi-Square
    #alle datensätze mit ungültigen koords löschen
    df = df[df.X != 0]
    df = df[df.Y != 0]


    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        df = df.drop('Dates', 1)
        df = df.drop('Address', 1)
        if (path == 'train.csv'):
            df = df.drop('Descript', 1)
            df = df.drop('Resolution', 1)
    print('Success for ', path)

    return df

def get_season(row):
    if 3 <= int(row['Dates'][5:7]) <= 5:
        return "SPRING"
    elif 6 <= int(row['Dates'][5:7]) <= 8:
        return "SUMMER"
    elif 9 <= int(row['Dates'][5:7]) <= 11:
        return "AUTUMN"
    else: return "WINTER"


"""
Feature Extraction
Feature Extraction mit ChiSquare Test, welcher Wert nimmt am meisten Einfluß wenn Null Hypothese gilt
Chi-Square Erklärung 5-min YouTube: https://www.youtube.com/watch?v=VskmMgXmkMQ ;; Besser: https://www.youtube.com/watch?v=WXPBoFDqNVk (12 min)
Quelle: http://www.handsonmachinelearning.com/blog/2AeuRL/chi-square-feature-selection-in-python
"""
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        self.dfTabular = None
        self.dfExpected = None


    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha: # If P is low, Ho (null hypothesis) must go...  - alpha is der Wert der bestimmt ob Null Hypothese zutrifft oder nicht
            result="{} is IMPORTANT for Prediction. Value of chi2 {}".format(colX, self.chi2)
        else:
            result="{} is NOT an important predictor. Value of chi2 {}".format(colX, self.chi2)

        print(result)

    def TestIndependence(self,colX,colY, alpha=0.1):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y,X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)

#Feature Selection
def useChi(cT):
    testColumns = ['Year', 'Month', 'Hour', 'PdDistrict', 'X', 'Y'] #['Year', 'Month', 'Day', 'Time', 'DayOfWeek', 'PdDistrict', 'X', 'Y']
    for var in testColumns: #Für jede einzelne Column wird  Chi-Square ausgeführt
        cT.TestIndependence(colX=var,colY="Category") #Aufruf des Chi-Square Test mit Resolution als abhängiges Features

def lgbm(data_set):
    # Quelle -> https://github.com/Microsoft/LightGBM/blob/master/tests/python_package_test/test_sklearn.py
    print ('Aufteilen des Datensatzes nach Feature Preprocessing')
    x_train_split_t = data_set['x_train_split']
    y_train_split_t = data_set['y_train_split']
    x_test_split_t = data_set['x_test_split'] #Erstellung von Test df um auswerten zu könne -> vorher einer unseren großen Fehler
    y_test_split_t = data_set['y_test_split']
    test_x = data_set['test_x']

    print ('setup training and eval')
    lgb_train = lightgbm.Dataset(x_train_split_t, y_train_split_t)
    lgb_eval = lightgbm.Dataset(test_x, reference=lgb_train)

    print ('trying to perform GBDT')
    clf = lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=127, subsample_for_bin=200000, objective='multiclass', silent=False )
    clf.fit(x_train_split_t, y_train_split_t, eval_set=[(x_test_split_t, y_test_split_t)])
    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        clf.predict(test_x)

    #print(y_pred)
    #pred_string=np.array_str(y_pred)
    #print (pred_string)
    #with open('pred_file.txt','w') as f:
    #    f.write(pred_string)
        
    print ('trying to plot here?')

    #fuck graphviz
    #ax = lightgbm.plot_tree(clf, tree_index=83, show_info=['split_gain'])    
    #ax = lightgbm.plot_tree(clf, tree_index=83, figsize=(500, 80), show_info=['split_gain'])
    # plt.show(ax)

    print('Plot 84th tree with graphviz...')
    graph = lightgbm.create_tree_digraph(clf, tree_index=83, name='Tree84')
    graph.render(view=True)
    plt.show(graph)
#Multi_LogLoss bei 2.40556 ohne Day und DayOfWeek [StandardConfig]
#Multi_LogLoss bei 2.40635 mit Day und DayOfWeek [StandardConfig]
#Multi_LogLoss bei 2.40207 ohne Day und DayOfWeek - Iteration 127 - Danach Anstieg - [StandardConfig]

#Aufrufen der Ausführung, bitte ganz unten
main()

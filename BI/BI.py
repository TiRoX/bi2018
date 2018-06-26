# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''


#Module importieren
import pandas as pd
import scipy.stats as stats
import lightgbm



def main():
    #Hier werden alle verschiedenen Methoden aufgerufen, da es sonst wirklich ziemlich unübersichtlich wird
    #Einlesen des Files
    df = readF("trainrewritten.csv", False) # True wenn Index im File vorhanden, wie hier.
    test = readF('testrewritten.csv', False)
    print(df)
    resulttrain = train(df)
    print (resulttrain)
    #cT = ChiSquare(df) #
    #useChi(cT) #gibt aus, welche Columns "important" sind für "Category"; DESCRIPT is most important





#Data Understanding & Data Preparation von BI_martin.py, dort wird von train.csv die csv "rewritten.csv" erstellt, und hier wieder eingelesen zur Auswertung.
def readF(path, index): #index == True, wenn Index vorhanden
    if (index == True):
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, index_col = 0) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    else:
        df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
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


def train(df):
    params = {}
    params['task'] = 'train'
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'multiclass'
    params['metric'] = 'multi_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
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
    y_train = df[df.columns[0]].values
    x_train = df.drop([0]).values
    #print(x_train)
    #x_train = df.drop(0, axis=1).values


    lgb_train = lightgbm.Dataset(x_train, y_train)
    #lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    #sc = StandardScaler()
    #ds = sc.fit_transform(X=df, y=None)

    clf = lightgbm.train(params, lgb_train, num_boost_round=100)
    return clf





#Aufrufen der Ausführung, bitte ganz unten
main()

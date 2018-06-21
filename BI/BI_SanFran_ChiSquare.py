#Module importieren
import pandas as pd
import datetime as dt
import os
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency #für chi²

#Import of Train.csv
def readF(var):
    df = pd.read_csv(var, delimiter= ',', header = 0, error_bad_lines=False) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    #OUTDATED
    #Feld "Dates" zu "Date" und "Time" teilen:
    #output['Date'], output['Time'] = output['Dates'].str.split(' ', 1).str

    with pd.option_context('display.max_rows', 11, 'display.max_columns', 200):
        #print(output.ix[257059]) # --> Einige Zeilen sind abgeschnitten und ergeben nicht immer viel Sinn. So wie diese hier; Excel index + 2 = Python,,, index 257061 = 257059
        print(df)

        # Abfrage für bestimmten Wert "NONE" in Spalte "Resolution"
        #print(output.loc[output['Resolution'] == 'NONE'])

        #Entfernt alle Einträge "NONE" aus der Spalte "Resolution"
        #print(output.ix[~(output['Resolution'] != 'NONE')])


        #Will suchen nach 'OWNING' im Feld 'Descript'; um das zu tun müssen ggf. Descript Felder in Liste umgewandelt werden. oider einzelnd in CSV ausgelesen werden
        #print(output.loc[output['Descript'].isin('OWNING')])

        #Viele kompakte leicht zu verstehende Informationen auf Code Basis sind hier zu finden -v
        #further use: https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/

        #existieren duplicates?
        #print (output.duplicated(subset='Dates', keep=False)) #Keep=False markiert alle Duplikate als True, keep=first, nur den ersten nicht

        #Gebe den Dataframe zurück, da wir nun alle Daten in der CSV wie gewünscht bearbeitet haben
        return df
        
#Data Understanding
#
df = readF('train_prepared.csv')
df.shape
df.dtypes #Überprüfen ob alle Spalten nur aus Einträgen eines Typs bestehen

#Data Preparation
#

"""
Feature Extraction
Feature Extraction mit ChiSquare Test, welcher Wert nimmt am meisten Einfluß wenn Null Hypothese gilt
Chi-Square Erklärung 5-min YouTube: https://www.youtube.com/watch?v=VskmMgXmkMQ
Quelle: http://www.handsonmachinelearning.com/blog/2AeuRL/chi-square-feature-selection-in-python
"""
#
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
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str) #Konvertierung zu String der unabhängigen Features
        Y = self.df[colY].astype(str) #Konvertierung zu String des abhängigen Features
    
        self.dfObserved = pd.crosstab(Y,X) #Anzahl für Observed in Abhängigkeit von Resolution
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
    
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX, alpha)

#Initialisierung ChiSquare Test
cT = ChiSquare(df)

#Feature Selection
testColumns = ["Category","Descript","DayOfWeek","PdDistrict"]
for var in testColumns: #Für jede einzelne Column wird  Chi-Square ausgeführt
    cT.TestIndependence(colX=var,colY="Resolution") #Aufruf des Chi-Square Test mit Resolution als abhängiges Features

df #Dataframe als Tabelle

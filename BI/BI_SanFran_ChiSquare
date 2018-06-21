#Module importieren
import pandas as pd
import datetime as dt
import os
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency #für chi²

#Import of Train.csv
#
df = pd.read_csv("C:/Users/fmert/Python_CodeComa/train.csv", delimiter= ',', header = 0, error_bad_lines=False) # , dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str} columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
#print(df) #Dataframe als rohe Tabelle
        
#Data Understanding
#
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

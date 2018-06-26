# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''

import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import os


def readF(var):
    #speichert Dataframe in df
    #liest den Dateipfad var aus, teilt columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    #ALT:
    #df = read_csv(var, delimiter= ',', quotechar='"', header = 0, dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AddressSuffix": str, "X": str, "Y": str}) # type = pandas.core.frame.DataFrame
    df = read_csv(var,delimiter=',', quotechar='"', header=0, dtype={"Dates":str, "Category":str, "Descript":str, "DayOfWeek":str, "PdDistrict":str, "Resolution":str, "X":str, "Y":str})

    #Feld "Dates" zu "Date" und "Time" teilen:
    df['Date'], df['Time'] = df['Dates'].str.split(' ', 1).str
    df['Year'] = df['Dates'].str[:4]
    df['Month'] = df['Dates'].str[5:7]
    df['Day'] = df['Dates'].str[8:10]
    df['Hour'] = df['Dates'].str[11:13]
    df['Season'] = df.apply(get_season, axis=1)
    #Note the axis=1 specifier, that means that the application is done at a row, rather than a column level.
    df['AddressSuffix'] = df['Address'].str[-2:]
    df['DayOfWeek'] = df['DayOfWeek'].str.upper()
    df['Address'] = df['Address'].str.upper()

    #print("vorher:")
    #print(df.ix[660485])

    #X: longitude of the incident location. San Francisco city longitude ranges from -122.5136 to -122.3649. float conversion used to ensure correct comparism
    #Y: latitude of the incident location. San Francisco city latitude ranges from 37.70788 to 37.81998. float conversion used to ensure correct comparism
    df['X'] = df['X'].apply(lambda x: np.NaN if float(x)>=-122.3649 or float(x)<=-122.5136 else x)
    df['Y'] = df['Y'].apply(lambda y: np.NaN if float(y)<=37.70788 or float(y)>=37.81998 else y)

    #print(df.dtypes)
    #print("nachher:")
    #print(df.ix[660485])
    #print(df.ix[660486])

    df=df.reindex(columns=['Date', 'Time', 'Year', 'Month', 'Day', 'Hour', 'Season', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'AddressSuffix', 'X', 'Y'])

    with pd.option_context('display.max_rows', 19, 'display.max_columns', 200):
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

def get_season(row):
    if 3 <= int(row['Dates'][5:7]) <= 5:
        return "SPRING"
    elif 6 <= int(row['Dates'][5:7]) <= 8:
        return "SUMMER"
    elif 9 <= int(row['Dates'][5:7]) <= 11:
        return "AUTUMN"
    else: return "WINTER"

def write_csv(df, name):
    rdstr = ".csv"
    path = name + rdstr
    print(path)
    if(os.path.isfile(path) == False):
        df.to_csv(path_or_buf = path ,sep=',')
    else:
        print ('Writing didnt work, cuz File is already there, pls delete in before')


#Die Python-Datei muss im gleichen Ordner wie die CSV-Files sein.
df1 = readF('train.csv')
write_csv(df1, 'rewritten')
df2 = readF('test.csv')
write_csv(df2, 'testrewritten')

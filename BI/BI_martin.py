# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''

import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import os
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer



def readF(var):
    #speichert Dataframe in df
    #liest den Dateipfad var aus, teilt columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    #ALT:
    #df = read_csv(var, delimiter= ',', quotechar='"', header = 0, dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AddressSuffix": str, "X": str, "Y": str}) # type = pandas.core.frame.DataFrame

    if (var == 'test.csv'):
        df = read_csv(var,delimiter=',', quotechar='"', header=0, dtype={"Dates":str, "DayOfWeek":str, "PdDistrict":str, "X":str, "Y":str})
    elif (var == 'train.csv'):
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
    #df['X'] = df['X'].apply(lambda x: 0 if float(x)>-122.3649 or float(x)<-122.5136 else x)#
    df['X'] = df['X'].apply(lambda x: 0 if float(x)>-122.3649 or float(x)<-122.513642064265 else x)
    df['Y'] = df['Y'].apply(lambda y: 0 if float(y)<37.70788 or float(y)>37.81998 else y)
    # war vorher np.NaN, jetzt "0"

    #alle datensätze mit ungültigen koords löschen
    df = df[df.X != 0]
    df = df[df.Y != 0]



    #if (var == 'train.csv'):
    #    df['Descript1'], df['Descript2'] = df['Descript'].str.split(',', 1).str
    #    df['Descript2'] = df['Descript2'].apply(lambda y: '' if str(y) == "nan" else y)



    #print(df.dtypes)
    #print("nachher:")
    #print(df.ix[660485])
    #print(df.ix[660486])

    if (var == 'test.csv'):
        df=df.reindex(columns=['Date', 'Time', 'Year', 'Month', 'Day', 'Hour', 'Season', 'DayOfWeek', 'PdDistrict', 'Address', 'AddressSuffix', 'X', 'Y'])
    elif (var == 'train.csv'):
        #df=df.reindex(columns=['Date', 'Time', 'Year', 'Month', 'Day', 'Hour', 'Season', 'Category', 'Descript1', 'Descript2', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'AddressSuffix', 'X', 'Y'])
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
        df.to_csv(path_or_buf = path ,sep=',', index=False)
    else:
        print ('Writing didnt work, because File is already there, pls delete it before')

def readF2(path):
    df = pd.read_csv(path, delimiter= ',', quotechar='"', header = 0, error_bad_lines=False, dtype={"AddressSuffix": str}, index_col=0)
    with pd.option_context('display.max_rows', 19, 'display.max_columns', 200):
        return df


#Die Python-Datei muss im gleichen Ordner wie die CSV-Files sein.
df1 = readF('train.csv')
write_csv(df1, 'trainrewritten')
df2 = readF('test.csv')
write_csv(df2, 'testrewritten')

##df1 = readF2('trainrewritten.csv')
##df2 = readF2('testrewritten.csv')

##target_mapper = DataFrameMapper([
##    ("Category", LabelEncoder()),
##])

##y = target_mapper.fit_transform(df1.copy())

##print ("sample:", y[0])
##print ("shape:", y.shape)


##data_mapper = DataFrameMapper([
##    ("Date", LabelEncoder()),
##    ("Time", LabelEncoder()),
##    ("Year", StandardScaler()),
##    ("Month", StandardScaler()),
##    ("Day", StandardScaler()),
##    ("Hour", StandardScaler()),
##    ("Season", LabelEncoder()),
##    ("Descript1", LabelEncoder()),
##    ("Descript2", LabelEncoder()),
##    ("DayOfWeek", StandardScaler()),
##    ("PdDistrict", LabelBinarizer()),
##    ("Resolution", LabelEncoder()),
##    ("Address", [LabelEncoder(), StandardScaler()]),
##    ("AddressSuffix", LabelEncoder()),
##    ("X", StandardScaler()),
##    ("Y", StandardScaler()),
##])

#df1_new = np.array(df1).reshape(-1,1)
#df2_new = np.array(df2).reshape(-1,1)

## ab hier hagelts fehler
##data_mapper.fit(pd.concat([df1.copy(), df2.copy()]))
##X = data_mapper.fit_transform(df1.copy())
##X_test = data_mapper.fit_transform(df2.copy())

##print ("sample:", X[0])
##print ("shape:", X.shape)

##samples = np.random.permutation(np.arange(X.shape[0]))[:100000]
##X_sample = X[samples]
##y_sample = y[samples]

##y_sample = np.reshape(y_sample, -1)
##y = np.reshape(y, -1)

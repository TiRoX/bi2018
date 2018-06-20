# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''

import pandas as pd
from pandas.io.parsers import read_csv



def readF(var):
    #speichert Dataframe in output
    #liest den Dateipfad var aus, teilt columns mit (delimiter";"), die headzeile ist die 0., dtype bestimmt datentyp der Columns
    df = read_csv(var, delimiter= ',', quotechar='"', header = 0, dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AddressSuffix": str, "X": str, "Y": str}) # type = pandas.core.frame.DataFrame

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




#Die Python-Datei muss im gleichen Ordner wie die CSV-Files sein.
#readF('train.csv')
readF('train_prepared.csv')

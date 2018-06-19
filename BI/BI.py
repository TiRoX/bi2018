# -*- coding: utf-8 -*-
'''
Created on 18.06.2018

@author: Kevin
'''

import pandas as pa
from pandas.io.parsers import read_csv



def readF(var):

    output = read_csv(var, delimiter= ';', header = 0, dtype={"Date": str, "Time": str, "Year": int, "Month": int, "Day": int, "Hour": int, "Season": str,  "Descript": str, "DayOfWeek": str, "PdDistrict": str, "Resolution": str, "Address": str, "AdressSuffix": str, "X": str, "Y": str}) # type = pandas.core.frame.DataFrame

    #OUTDATED
    #split Dates
    #output['Date'], output['Time'] = output['Dates'].str.split(' ', 1).str

    with pa.option_context('display.max_rows', 11, 'display.max_columns', 200):
        #print(output.ix[257059]) # --> Einige Zeilen sind abgeschnitten und ergeben nicht immer viel Sinn. So wie diese hier; Excel index + 2 = Python,,, index 257061 = 257059
        print(output)
        
        # Abfrage für bestimmten Wert in Spalte
        #print(output.loc[output['Resolution'] == 'NONE'])
        
        #Will suchen nach 'OWNING' im Feld 'Descript'; um das zu tun müssen ggf. Descript Felder in Liste umgewandelt werden. :)
        #print(output.loc[output['Descript'].isin('OWNING')])
        
        #further use: https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
        
        #duplicates?
        #print (output.duplicated(subset='Dates', keep=False)) #Keep=False markiert alle Duplikate als True, keep=first, nur den ersten nicht
        
        #selecting
        #print(output.ix[output['Resolution'] != 'NONE'])
        #removing
        #print(output.ix[~(output['Resolution'] != 'NONE')])
        #return output

#Class has to be in same directory as the .csv
#readF('train.csv')
readF('train_prepared.csv')
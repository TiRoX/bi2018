'''
Created on 01.07.2018

@author: Kevin, Bereitstellung von Idrae, mit Änderungen entnommen aus: https://github.com/Idraen/kaggle_sf_crime/blob/master/scripts/data_handler.py#L97
'''

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

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
    dayofweek_mapping = {"MONDAY": 0, 
                         "TUESDAY": 1,
                         "WEDNESDAY": 2,
                         "THURSDAY": 3,
                         "FRIDAY": 4,
                         "SATURDAY": 5,
                         "SUNDAY": 6}

    def __init__(self):
        self.training_data = None
        self.testing_data = None

    # Runs a call to the get_data for each path, used to set <training_data> and <testing_data> attributes.
    def load_data(self, train, test):
        self.training_data = train
        self.testing_data = test

    # Creates a <big_data> DataFrame containing both training and testing set to mutualize the feature preprocessing.
    # Returns a dictionary containing a well-splitted version of the data.
    def transform_data(self, with_mask=1):
        features_columns = [ 'Year', 'Month', 'Day', 'Time', 'Season', 'DayOfWeek', 'PdDistrict', 'X', 'Y']#['DayOfWeek', 'PdDistrict', 'X', 'Y', 'Year', 'Month', 'Day', 'Hour', 'Minute']
        big_data = self.training_data[features_columns].append(self.testing_data[features_columns])

        categorical_features = ['Year', 'Month', 'Day','Time', 'Season', 'DayOfWeek', 'PdDistrict'] #'Resolution', ['DayOfWeek', 'PdDistrict', 'Year', 'Month', 'Day', 'Hour', 'Minute']
        numerical_features = ['X', 'Y']
        
        print ('Trying to transform data')
        big_data = self.categorical_encoder(big_data, categorical_features)
        big_data = self.features_preprocessing(big_data, numerical_features)
        print ('Success')
        train_X = big_data[0:self.training_data.shape[0]]
        train_Y = self.training_data['Category'].map(self.category_mapping)
        test_X = big_data[self.training_data.shape[0]::]

        if with_mask != 1:
            mask = np.random.rand(len(self.training_data)) < with_mask
            train_X = train_X[mask]
            train_Y = train_Y[mask]
        print ('Wrapping up Data Transformation')
        return {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X}

    # Method to scale the numerical columns of a DataFrame.
    def features_preprocessing(self, data, numerical_columns):
        # The preprocessing package from sklearn does a good job.
        for num_col in numerical_columns:
            data[num_col] = preprocessing.scale(data[num_col])
        return data

    # Method to encode the categorical columns of a DataFrame.
    def categorical_encoder(self, data, categorical_columns):
        # The LabelEncoder class from sklearn.preprocessing does a good job.
        le = LabelEncoder()
        for cat_col in categorical_columns:
            if cat_col == "DayOfWeek":
                data[cat_col] = data[cat_col].map(self.dayofweek_mapping)
            data[cat_col] = le.fit_transform(data[cat_col])
        return data
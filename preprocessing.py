#!/usr/bin/env python
# coding: utf-8
# author: Reza Saneei



import pandas as pd
from datetime import datetime
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# loads the sensors data
dataset_path = './datasets/'

file_name = 'Fredrikstad_cleaned_int.csv'
input_path = dataset_path + file_name

df = pd.read_csv(input_path, parse_dates=True, index_col='Date')

#loads the weather data
file_name = 'fredrikstad-weather-raw.csv'
input_path = dataset_path + file_name
weather_df = pd.read_csv(input_path)

# matchs the sensors and weather dataframe indecies
weather_df.rename(columns = {'dt_iso': 'Date'}, inplace= True)
weather_df['Date'].replace('\+0000 UTC', '',regex=True, inplace = True)
weather_df['Date'] = pd.to_datetime(weather_df['Date'], format= '%Y-%m-%d')
weather_df.set_index('Date', inplace= True);

# merges sensors and weather dataframes
big_df = pd.merge( weather_df, df, on= 'Date', how= 'outer')

# saves the new dataframe to a csv file
file_name = 'weather-traffic-data.csv'
output_path = dataset_path + file_name
big_df.to_csv(output_path)

df = big_df

# Specifies the features that will ream
weather_features = ['temp', 'wind_speed', 'rain_1h', 'clouds_all', 'weather_main']
sensor_features = ['GF-Ferjeleia', 'GF-Glasshytta', 'GF-Isegran',
       'GF-Kasernen', 'GF-Laboratoriegaten', 'GF-Magenta', 'GF-Mormors',
       'GF-Pumpehuset', 'Totalt']
input_features = weather_features + sensor_features
df = df[input_features]


# replace the NaN values with 0
df['rain_1h'].fillna(0, inplace=True);

# Converts the temperature unit from kelvin to celsius
df['temp'] = df['temp'].add(-273.15)
# rounds the temperature to 1 decimal place
df['temp'] = df['temp'].apply(lambda x: round(x, 1))

# converts the weather_main to one hot vector
# concatenates it with df
# drops the weather_main
dummy_df = pd.concat([pd.get_dummies(
    df['weather_main'], dummy_na=False), df], axis=1).drop(['weather_main'], axis=1)

#Trimming the parts with no sensor data
dummy_df = dummy_df['2018-10-01': '2019-07-08']


Norway_holidays_list = ['2018-12-25', '2018-12-26', '2019-01-01', '2019-01-01',
               '2019-04-18', '2019-04-19', '2019-04-21', '2019-04-22',
               '2019-05-01', '2019-05-17', '2019-05-30', '2019-06-09',
               '2019-06-10']
df = dummy_df

df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.dayofyear
df['Weekday'] = df.index.weekday
df['Hour'] = df.index.hour
df['Weekend'] = df.index.weekday//5
df['Norway Holiday'] = 0

# Set the Norway Holidays to 1
for holiday_date in Norway_holidays_list:
    df['Norway Holiday'][holiday_date] = 1


#dummy_df.columns
# Rearranges the order of columns
df = df[['Year', 'Month', 'Day', 'Weekday', 'Hour', 'Weekend', 'Norway Holiday',
       'Clear', 'Clouds', 'Drizzle', 'Fog', 'Mist', 'Rain',
       'Snow', 'Thunderstorm', 'temp', 'wind_speed', 'rain_1h', 'clouds_all',
       'GF-Ferjeleia', 'GF-Glasshytta', 'GF-Isegran', 'GF-Kasernen',
       'GF-Laboratoriegaten', 'GF-Magenta', 'GF-Mormors', 'GF-Pumpehuset',
       'Totalt']]

# Renaming the index
#df.rename(columns = {'Date': 'DateIndex'}, inplace= True)
df.index.names = ['DateIndex']
df = df.astype(np.int64)
#saves the dummy_dataframe to a new csv file
file_name = 'weather-traffic-dummy.csv'
output_path = dataset_path + file_name




df.to_csv(output_path)

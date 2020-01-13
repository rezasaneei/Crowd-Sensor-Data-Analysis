"""
Created on Mon Oct  7 20:58:57 2019

@author: Reza Saneei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import statsmodels.api as sm
import warnings
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# constants

# loads the sensors data
dataset_path = './datasets/'

data_path = dataset_path + 'weather-traffic-dummy.csv'
output_path = './csv_outputs/'
OUTPUT_NAME = 'Totalt'
SHIFT_HOURS = 24
SCALE_DATA = True

# loads data from root folder
df = pd.read_csv(data_path, index_col='DateIndex',
                 parse_dates=True)

# we shift the data to let the model to learn to predict (forecast)
# the future. In other words the model learns to map each data point
# of input signal (e.g. weather input data), to the ouput signal in
# the future (e.g. the sensor readings).
shift_hours = SHIFT_HOURS

# input and output signals names
input_names = df.columns[0:26].to_list()
output_name = OUTPUT_NAME

# two separate dataframes for input and output
input_df = df[input_names]
output_df = df[output_name].shift(-shift_hours)

# If output has only one signal and therefore is a Series, converts it to dataframe
if isinstance(output_df, pd.Series):
    output_df = output_df.to_frame()

# removing the NaN values after shiftting
input_df = input_df[0:-24]
output_df = output_df[0:-24]

# Normalizing the input values between 0 and 1
if SCALE_DATA:
    input_scaler = MinMaxScaler()
    input_df.loc[:, :] = input_scaler.fit_transform(input_df.loc[:, :])

    output_scaler = MinMaxScaler()
    output_df.loc[:, :] = output_scaler.fit_transform(output_df.loc[:, :])

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    input_df, output_df, test_size=0.1, shuffle=False, stratify=None)
print('Train X,Y shape: ', X_train.shape, y_train.shape)
print('Test X,Y shape: ', X_test.shape, y_test.shape)


#df = pd.read_csv('../main/weather-traffic-dummy.csv', index_col='DateIndex', parse_dates=True)

# reading csv file
pred = pd.read_csv(output_path + 'arima_one_step_ahead.csv', parse_dates=True, names = ['DateIndex', 'pred'], index_col='DateIndex')
pred_uc = pd.read_csv(output_path + 'arima_forecast.csv', parse_dates=True, names = ['DateIndex', 'pred'], index_col='DateIndex')
pred.head()

GRU_output = pd.read_csv(output_path + 'GRU_test_output.csv', parse_dates=True, index_col='DateIndex')

total_pred = pd.concat([pred, pred_uc])
#total_pred

ARIMA_pred = total_pred['2019-06-09 11:00:00':'2019-07-07 22:00:00'].copy()

ARIMA_pred.plot()

#ARIMA_df = ARIMA_pred.to_frame()
ARIMA_df = ARIMA_pred
ARIMA_df.loc[:, :] = output_scaler.inverse_transform(ARIMA_df.loc[:, :])


GRU_output = pd.read_csv(output_path + 'GRU_test_output.csv', parse_dates=True, index_col = 'DateIndex')
pred_df = GRU_output
pred_df['ARIMA'] = ARIMA_df['2019-06-09 11:00:00':'2019-07-07 22:00:00']
pred_df.rename(columns={"Totalt_pred": "GRU"}, inplace = True)
pred_df.plot(figsize=(14, 6))
pred_start = pred_df.index[-1]
pred_end = pred_df.index[-24]
# draw an orange square at right end of the plot to show the prediction area.
plt.axvspan(pred_start, pred_end, facecolor='orange',
            edgecolor='black', alpha=.5)
plt.title('ARIMA predictions compared to GRU and true sensor readings.')
plt.savefig('ARIMA_comparison.pdf', dpi=300)

"""## Reading and recovering the models_df from csv file"""

models_df = pd.read_csv(output_path + 'models_df.csv', index_col='model_name', dtype = object)
models_df.model_pred



model_names = ['RandomForestRegressor',
'BayesianRidge',
'LinearRegression',
'Lasso',
'Ridge',
'SVR']

for model_name in model_names:
  # Revovering lists from strings
  pred_str = models_df.model_pred.loc[model_name]
  print(pred_str)
  pred_str = pred_str.replace('    ', ' ')
  pred_str = pred_str.replace('   ', ' ')
  pred_str = pred_str.replace('  ', ' ')
  pred_str = pred_str.replace(' ', ',')

  pred_list = eval(pred_str)
#  pred_list = [item[0] for item in pred_list]
  pred_list
  pred_df[model_name] = pred_list[0:-1]
  pred_df.loc[:,[model_name]] = output_scaler.inverse_transform(pred_df.loc[:,[model_name]])


#plotting all_models

pred_df.plot(figsize=(14, 6))
pred_start = pred_df.index[-1]
pred_end = pred_df.index[-24]
# draw an orange square at right end of the plot to show the prediction area.
plt.axvspan(pred_start, pred_end, facecolor='orange',
            edgecolor='black', alpha=.5)
plt.title('Performance of all models on the test data.')
plt.savefig('all_models_comparison.pdf', dpi=300)

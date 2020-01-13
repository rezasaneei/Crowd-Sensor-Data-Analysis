# -*- coding: utf-8 -*-
# This code is later modified by Reza Saneei
# The oroginal code is authored by Thomas Vincent which is available at:
## https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
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
data_path = 'weather-traffic-dummy.csv'
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

SEASONALITY = 24
# Due to the nature of sensor data we have resampled from hours ('H') which 
# makes the algorithm runs slower. Might take about 20 minutes.
y = output_df['Totalt'][:'2019-07-06 23:00:00'].resample('H').mean()

y.plot(figsize=(15, 6))
plt.show()

rcParams['figure.figsize'] = 15, 6
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.savefig('SARIMAX-decomposition.pdf', dpi=300)
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], SEASONALITY)
                for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, SEASONALITY),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.savefig('SARIMAX-analysis.pdf', dpi=300)
plt.show()


pred = results.get_prediction(
    start=pd.to_datetime('2019-05-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y[:].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(
    pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Totalt')
plt.legend()
plt.savefig('SARIMAX-one-step.pdf', dpi=300)
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2019-05-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 5)))
print('The Root Mean Squared Error of our forecasts is {}'.format(
    round(np.sqrt(mse), 5)))

pred_uc = results.get_forecast(steps=1000)
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Totalt')

plt.legend()
plt.savefig('SARIMAX-pred.pdf', dpi=300)
plt.show()

pred_uc = results.get_forecast(steps=300)
pred_ci = pred_uc.conf_int()

ax = y['2019-06-09 11:00:00':].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Totalt')

plt.legend()
plt.savefig('SARIMA_long_pred.pdf', dpi=300)
plt.show()

pred = results.get_prediction(
    start=pd.to_datetime('2019-05-01'), dynamic=False)
pred_ci = pred.conf_int()
y_df = y.to_frame()
pred_df = pred.predicted_mean.to_frame()

y_df.loc[:, :] = output_scaler.inverse_transform(y_df.loc[:, :])
pred_df.loc[:, :] = output_scaler.inverse_transform(pred_df.loc[:, :])
#y_df = output_scaler.inverse_transform(y.to_frame().loc[:, :])

ax = y_df['2019-06-09 11:00:00':'2019-07-07 23:00:00'].plot(label='Totalt')
pred_df['2019-06-09 11:00:00':'2019-07-07 23:00:00'].plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(
    pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Totalt')
plt.legend()
plt.savefig('SARIMAX-one-step.pdf', dpi=300)
plt.show()

pred_uc.predicted_mean.to_csv('sarimax_forecast.csv')
pred.predicted_mean.to_csv('sarimax_one_step_ahead.csv')

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:35:49 2019

@authors: Reza Saneei
"""

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.preprocessing import MinMaxScaler
#from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import expon, uniform

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


# a list of lists, each item contains the model and the dictionary of hparams_ranges.
models_list = [
    RandomForestRegressor(),
    BayesianRidge(),
    LinearRegression(),
    Lasso(),
    Ridge(),
    SVR()]

# prepairing columns of the models_df
models_name = [type(model).__name__ for model in models_list]
models_hparams = [{} for model in models_list]
models_metrics = [None for model in models_list]
models_pred = [None for model in models_list]

# a dataframe for models and their hyperparameters
models_df = pd.DataFrame(
    {'model_name': models_name, 'model_instance': models_list,
     'model_hparams': models_hparams, 'model_metrics': models_metrics,
     'model_pred': models_pred}, dtype = object)
models_df.set_index('model_name', inplace=True)

# a refrence dictionatry of dictionaries. Each key correspond to a dictionary of hparams of a model.
# a refrence dictionary for hparams range. The optimizer chose the hparams by its key.
hparams_ref = {
    'RandomForestRegressor': {
        # 'max_depth': sp_randint(3, 50),
        'n_estimators': sp_randint(5, 200),
        'min_samples_split': sp_randint(2, 10),
        'max_features': sp_randint(1, 11),
        'min_samples_split': sp_randint(2, 11)
    },
    'GaussianNB': {

    },
    'LinearRegression': {

    },
    'Lasso': {
        'alpha': expon(scale=.01)
    },
    'SVR': {
        'epsilon':  expon(scale=.3),
        'C': expon(scale=5)

    },
    'Ridge': {
        'alpha': expon(scale=.01)
    },
    'BayesianRidge': {
        'alpha_1': expon(scale=.0001),
        'alpha_2': expon(scale=.0001),
        'lambda_1': expon(scale=.0001),
        'lambda_2': expon(scale=.0001)
    }
}


def set_hparams(model, hparams_dict):
    '''
    Set the hparams of a model object (its attributes) from a dictionary.
    Arg:
      model: the instance of the estimator class
      hparams_dict: dictionary of a few or all of hyperparameters    
    returns:
      dict of all hparams including the modified ones.
    '''
    print('Model\'s hparams are set to: ')
    for key, value in hparams_dict.items():
        print(key, value)
        setattr(model, key, value)


def report(results, n_top=3):
    '''
    reports the results of the sklearn optimizer.
    it is called by the optimize_model().
    Args:
      results: HPO method output results including best runs etc.
      n_top: top n best configuration that HPO has found.

    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def optimize_model(model, X_train, y_train, hparams_range, n_iter_search=20, cross_validation=2):
    '''
    Optimizes the hyperparameters of a given model using Random Search.
      Args:
        model: The instance of the sklearn model
        X_data: input data
        y_data: output data
        hparams_range: range of hyperparameters
        n_iter_search: number of iterations of random search
      returns: None
    '''
    # run randomized search
    random_search = RandomizedSearchCV(model, param_distributions=hparams_range,
                                       n_iter=n_iter_search, cv=cross_validation, iid=False,
                                       verbose=1)
    start = time()
    random_search.fit(X_train, y_train.values.ravel())
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    report(random_search.cv_results_)
    return random_search.best_params_, random_search.cv_results_


def optimize_all(models_df, X_train, y_train, hparams_ref, n_iter_search=2, cross_validation=2):
    '''
    Optimizes all the models and fills the models_df with the results.
    '''
    for index, row in models_df.iterrows():
        model = row['model_instance']
        model_name = row.name
        hparams_range = hparams_ref[model_name]
        # gets the best_params and results including the fitting time.
        best_hparams, results = optimize_model(
            model, X_train, y_train, hparams_range, n_iter_search, cross_validation)
        row['model_hparams'] = best_hparams


def train_model(model, hparams, X_train, y_train, X_test, y_test):
    '''
    function to train a given model, and optimize in with random 
    search if the hparams ranges are given.
    Args:
        model: an instance of the model class, e.g RandomForestRegressor
        data: dictionary holds 
        optimize: Optimizes the model if set True.
        verbose: prints the logs.
        logfile: If given, saves the logs in the file.
    '''
    # sets the hyperparameters of the model obj instance
    set_hparams(model, hparams)
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)
    #flattening the pred which is a np array
    pred = pred.ravel()
    model_mse = mse_score(y_test, pred)
    return pred, model_mse


def train_all(models_df, X_train, y_train, X_test, y_test):
    '''
    train all the models and saves the best metrics and predicted output in the models_df.
    Args:
        models_df  
    '''
    for index, row in models_df.iterrows():
        model = row['model_instance']
        model_name = row.name
        hparams = row['model_hparams']
        # trains each model and gets its predicted output and test MSE loss
        pred, model_mse = train_model(
            model, hparams, X_train, y_train, X_test, y_test)
        row['model_pred'], row['model_metrics'] = pred, model_mse


def plot_outputs(models_df, y_test, merge_other_results=False, output_name='all_models.pdf', plot_title='Predictions'):
    '''
    gets all the pridicted outputs from the models_df and plot them 
    on the same fig. Also reads the results of other outsider models from a csv
    and add them to the same models_df.
    '''
    # makes sure to take a copy of the y_test dataframe.
    pred_df = y_test.copy()
    # determines the red area on the plot to show the prediction.
    pred_start = pred_df.index[-1]
    pred_end = pred_df.index[-SHIFT_HOURS]
    #output_start = X_train.index[-240]

    for index, row in models_df.iterrows():
        model_name = row.name
        pred = row['model_pred']
        pred_df[model_name] = pred
    pred_df.loc[:, :] = output_scaler.inverse_transform(pred_df.loc[:, :])
    # attaches the other models results to the dataframe.
    if merge_other_results:
        GRU_path = output_path + 'GRU_test_output.csv'
        GRU_df = pd.read_csv(GRU_path, index_col='DateIndex', parse_dates=True)
        pred_df = pred_df.iloc[:-1, :]

        GRU_scaled = GRU_df.copy()
        GRU_scaled.loc[:, :] = output_scaler.fit_transform(
            GRU_scaled.loc[:, :])
        GRU_mse = mse_score(GRU_scaled['Totalt'], GRU_scaled['Totalt_pred'])
        # a row of GRU results to add to models_df
        GRU_row = {'model_instance': None, 'model_hparams': None,
                   'model_metrics': GRU_mse,  'model_pred': GRU_df['Totalt_pred']}
        models_df.loc['GRU'] = GRU_row
        pred_df['GRU'] = GRU_df['Totalt_pred']

    pred_df.plot(figsize=(14, 6))
    # draw an orange square at right end of the plot to show the prediction area.
    plt.axvspan(pred_start, pred_end, facecolor='orange',
                edgecolor='black', alpha=.5)
    plt.title(plot_title)
    plt.savefig(output_name, dpi=300)


def main(n_iter_search=2, cross_validation=5):
    optimize_all(models_df, X_train, y_train, hparams_ref,
                 n_iter_search, cross_validation)
    train_all(models_df, X_train, y_train, X_test, y_test)
    plot_outputs(models_df, y_test, merge_other_results=True,
                 plot_title='Performance of different models on multi-feature time-series regression.')
    best_model = models_df[models_df['model_metrics']
                           == models_df['model_metrics'].min()]
    plot_outputs(best_model, y_test,
                 merge_other_results=True, output_name='best_models.pdf', plot_title='Best models predictions.')

    models_mse = models_df['model_metrics']
    models_mse.to_csv(output_path + 'models_mse.csv')
    models_df.to_csv(output_path + 'models_df.csv')


# runs the whole pipeline. Modify the main function if necessary.
if __name__ == "__main__":
    main(n_iter_search=4, cross_validation=2)

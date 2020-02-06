# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:42:30 2019

@author: Reza
"""
import multiprocessing
import time
import os
import warnings
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
from colorama import Fore, Style
import colorama
colorama.init()


def calculate_SARIMAX(arg):
       print(Fore.GREEN + "Worker process id for {0}*{1} is {2}".format(arg[2][0], arg[2][1], os.getpid()), Style.RESET_ALL)
       mod = sm.tsa.statespace.SARIMAX(arg[0],
                                        exog = arg[1],
                                        order=arg[2][0],
                                        seasonal_order=arg[2][1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
              
       results = mod.fit()
       print(Fore.GREEN + 'SARIMAX{}x{}12 - AIC:{}'.format(arg[2][0], arg[2][1], results.aic), Style.RESET_ALL)
       return (arg[2][0], arg[2][1], results.aic)



if __name__ == '__main__':

       data = pd.read_excel('Desktop/SARIMAX/weather-traffic-dummy.xlsx')
       df = data

       df = df.replace('Yes',1)
       df = df.replace('No',0)
       df = df.drop(['Index', 'DateIndex'], axis=1)

       v1 = df.iloc[:,19:28].astype('float64')
       v2 = df.iloc[:,0:18].astype('float64')

       # v1 = v1.to_numpy()
       # v2 = v2.to_numpy()

       y = v1.iloc[:,8] #Totalt Sensor

       p = q = range(0, 6)
       d= range(0,2)

       pdq = list(itertools.product(p, d, q))
       seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

       print('Examples of parameter combinations for Seasonal ARIMA...')
       print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
       print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
       print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
       print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

       arguments = list(itertools.product(pdq,seasonal_pdq))

       #print("First few arguments are : ", arguments[0:4])
       
       start = time.time()

       p =  multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
       result = p.imap_unordered(calculate_SARIMAX, [(y, v2, x) for x in arguments[0:4]])
       
       end = time.time()

       #print(result)
       for r in result:
              print(Fore.RED + str(r) + Style.RESET_ALL)

       print(Fore.RED + "Time took : ",end - start,"s", Style.RESET_ALL)


       """         
       # fit best model SARIMAX
       model = sm.tsa.statespace.SARIMAX(y,
                                          exog = v2,
                                          order=pdq,
                                          seasonal_order=seasonal_pdq,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

       result = model.fit()

       print(result.summary().tables[1])
       """


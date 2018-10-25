import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss


def metrics(data):
    print('\nKpss test: ')
    # Stationary around a constant
    print(kpss(data, regression='c'))
    # Stationary around a trend
    print(kpss(data, regression='ct'))

    print('\nAdfuller test:')
    # Constant only
    print(adfuller(data, regression='c'))
    # Constant trend
    print(adfuller(data, regression='ct'))
    # Constant, and linear and quadratic trend
    print(adfuller(data, regression='ctt'))

def plot(dates, data, title='', ylabel=''):
    plt.plot(dates, data)
    plt.grid(True)
    plt.ylabel(title)
    plt.title(ylabel)
    plt.show()

def stationarize(timeserie, time_lag=10):
    """
    Transform the given time serie into stationary by normalizing, 
    using z-score, against the past 'time_lag' days
    """
    stationary_ts = [0] * timeserie.size
    for final_index in range(time_lag, timeserie.size, time_lag):
        start_index = final_index - time_lag
        stationary_ts[start_index:final_index] = stats.zscore(timeserie[start_index:final_index])

    if final_index != timeserie.size:
        stationary_ts[final_index:timeserie.size] = stats.zscore(timeserie[final_index:timeserie.size])
    
    return stationary_ts

def standardize_laggedly(timeseries, time_lag=10):
    standardized_ts = np.zeros(timeseries.size)
    standardized_ts[0:time_lag] = np.nan

    # Standardization against previous days
    for index in range(time_lag, timeseries.size):
        prev = index - time_lag
        mean = timeseries[prev:index].mean()
        std = timeseries[prev:index].std()
        standardized_ts[index] = np.divide(timeseries[index] - mean, std)

    # Transform infs into nans
    standardized_ts[abs(standardized_ts) == np.inf] = np.nan

    return standardized_ts

def standardize(timeseries, time_lag=10):
    standardized_ts = np.zeros(timeseries.size)
    standardized_ts[0:time_lag] = np.nan

    # Standardization against previous days
    for index in range(time_lag, timeseries.size):
        prev = index - time_lag
        # mean = timeseries[prev:index].mean()
        standardized_ts[index] = np.divide(timeseries[index], timeseries[prev])

    # Transform infs into nans
    standardized_ts[abs(standardized_ts) == np.inf] = np.nan

    return standardized_ts
    
def first_differentiate(timeseries, periods=1, use_log=False):
    stationary_ts = timeseries
    if use_log:
        # Calculate log
        stationary_ts = np.log(stationary_ts)

        # Log of negative numbers is nan
        # Log of 0 is -inf
        stationary_ts = stationary_ts.replace([np.inf, -np.inf, np.nan], 0)
    
    # Return first differentiated series
    return stationary_ts.diff(periods=periods)

def window_stack(array, window, step=1, spatial=False):
    """
    Method for creating timeseries windows.

    spatial is only applicable when array.ndim > 1
    """
    # array[initial_index:final_index:step]
    # each iteration of the for adds the ith element of every window
    # It is completed concurrently

    if array.ndim == 1:
        return np.vstack(
            array[i:(1 + i - window or None):step] 
            for i in range(0, window)
        )
    else:
        if spatial:
            return np.array([
                array[i:(window + i):step]
                for i in range(0, len(array) - window + 1)
            ])

        # TODO: Fix this
        return np.hstack(
            array[i:(1 + i - window or None):step] 
            for i in range(0, window)
        )
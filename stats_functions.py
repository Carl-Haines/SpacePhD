import numpy as np
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
import csv
import datetime
from scipy import stats
import matplotlib.pyplot as plt
import sys
#sys.path.insert('.useful_code')
import pandas_functions as pd_fn

def my_kde(data):
    #
    x = data
    #x_grid = np.linspace(int(min(data))-10, int(max(data) + 10), 1000)
    x_grid = np.linspace(0, int(max(data) + 10), 1000)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(2.5,4,1)},
                    cv=5, n_jobs=1)  # 20-fold cross-validation
    grid.fit(x[:, None])
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    #print(grid.best_params_)
    return x_grid, pdf


def subset_time_for_arrays(array, start_time, end_time):
    #array = np.asarray(array)
    new_array = array[(array >= start_time) & (array < end_time)]
    return new_array


def rolling_mean(data, number_of_values_in_window, center=True):
    # data: series or array: to be meaned
    # number_of_values_in_window: int: 8 for 24hours
    return data.rolling(number_of_values_in_window, center=center).mean()


def theta(df):
    theta = np.arctan2(-df['By'],df['Bz'])
    return theta


def geoeffectiveness(omni_data, alpha):
    np_term = omni_data['Proton density']**(2/3 -alpha)
    b_term = omni_data['HMF ave']**(2*alpha)
    v_term = omni_data['Solar wind speed']**(7/3 -alpha)
    theta_term = np.sin(theta(omni_data)/2)**4
    g = np_term * b_term * v_term * theta_term
    return g



def superposed_epoch(data, epochs, window_before, window_after, yscale='linear'):
    start_time = time.time()
    print('Starting your analysis')
    epoch_window_before = datetime.timedelta(days=window_before)
    epoch_window_after = datetime.timedelta(days=window_after)
    epochs = subset_time_for_arrays(epochs,
                                data['datetime'].iloc[0] + epoch_window_before,
                                data['datetime'].iloc[data.shape[0] - 1] - epoch_window_after)
    number_of_events = len(epochs)
    print('Analysing ' + str(number_of_events) + ' storms')
    epoch_matrix = np.full([(window_before + window_after) * 8 +1, number_of_events], np.NaN)
    for i in range(number_of_events):
        an_epoch = pd_fn.subset_time(data, 'datetime', epochs.iloc[i] - epoch_window_before,
                               epochs.iloc[i] + epoch_window_after + datetime.timedelta(seconds=1))
        epoch_matrix[:, i] = an_epoch['geomag'].values
    mean_array = np.mean(epoch_matrix, axis=1)
    standard_error_array = stats.sem(epoch_matrix, axis=1)
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))
    print('Analysis complete')

    fig, ax = plt.subplots()
    ax.errorbar(np.linspace(-window_before, window_after, (window_before + window_after) * 8+1), mean_array,
                yerr=standard_error_array, capsize=2, marker='x', markersize=1)
    ax.axvline(x=0, linestyle='--', color='k')
    ax.axhline(y=pd_fn.find_quantile(data,'geomag',0.9), linestyle='--', color='k')
    ax.set_xlabel('time (days)')
    ax.set_ylabel(r'$aa_H \ (nT)$')
    ax.set_xlim([-window_before, window_after])
    plt.show()
    return


def superposed_epoch_median(data, epochs, window_before, window_after, yscale='linear'):
    start_time = time.time()
    print('Starting your analysis')
    epoch_window_before = datetime.timedelta(days=window_before)
    epoch_window_after = datetime.timedelta(days=window_after)
    epochs = subset_time_for_arrays(epochs,
                                data['datetime'].iloc[0] + epoch_window_before,
                                data['datetime'].iloc[data.shape[0] - 1] - epoch_window_after)
    number_of_events = len(epochs)
    print('Analysing ' + str(number_of_events) + ' storms')
    epoch_matrix = np.full([(window_before + window_after) * 8 +1, number_of_events], np.NaN)
    for i in range(number_of_events):
        an_epoch = pd_fn.subset_time(data, 'datetime', epochs.iloc[i] - epoch_window_before,
                               epochs.iloc[i] + epoch_window_after + datetime.timedelta(seconds=1))
        epoch_matrix[:, i] = an_epoch['geomag'].values
    median_array = np.quantile(epoch_matrix, 0.5, axis=1)
    q1_array = np.quantile(epoch_matrix, 0.25, axis=1)
    q3_array = np.quantile(epoch_matrix,0.75, axis=1)
    percentile_10 = np.quantile(epoch_matrix, 0.1, axis=1)
    percentile_90 = np.quantile(epoch_matrix, 0.9, axis=1)
    np.quantile(epoch_matrix, 0.25, axis=1)
    #standard_error_array = stats.sem(epoch_matrix, axis=1)
    sigma_array = np.std(epoch_matrix, axis = 1)
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))
    print('Analysis complete')

    sigma_percentiles = [0.0015, 0.0225,0.16, 0.84, 0.9725, 0.9985]
    sigma = np.quantile(epoch_matrix, sigma_percentiles, axis=1)
    print(np.shape(sigma))
    fig, ax = plt.subplots()
    #ax.errorbar(np.linspace(-window_before, window_after, (window_before + window_after) * 8+1), mean_array,
    #            yerr=standard_error_array, capsize=2, marker='x', markersize=1)
    #for i in range(int(len(sigma_percentiles)/2)):
    #    print(i)
    #    ax.fill_between(np.linspace(-window_before, window_after, (window_before + window_after) * 8+1), sigma[i],sigma[5-i])
    ax.fill_between(np.linspace(-window_before, window_after, (window_before + window_after) * 8+1), percentile_10,
                    percentile_90, color='mediumpurple', alpha=0.5)
    ax.fill_between(np.linspace(-window_before, window_after, (window_before + window_after) * 8 + 1), q1_array,
                    q3_array, color='white', alpha=1)
    ax.fill_between(np.linspace(-window_before, window_after, (window_before + window_after) * 8+1), q1_array,
                    q3_array, color='orchid', alpha=0.5)

    ax.plot(np.linspace(-window_before, window_after, (window_before + window_after) * 8 + 1), median_array, color='white', marker = 'x', markersize=2)
    ax.axvline(x=0, linestyle='--', color='k')
    ax.axhline(y=pd_fn.find_quantile(data,'geomag',0.9), linestyle='--', color='k')
    ax.set_xlabel('time (days)', fontsize = 16)
    ax.set_ylabel(r'$aa_H \ (nT)$', fontsize = 16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim([-window_before, window_after])
    fig.tight_layout()
    ax.set_yscale(yscale)
    #plt.show()
    return


def cdf_sampler(data, size):
    rand = np.random.random(size)# genrate size x random numbers between 0 and 1
    values = np.percentile(data, rand*100)# find the values of these percentiles
    return values


def mean_and_se(data):
    mu = np.mean(data)
    se = stats.sem(data)
    return mu, se


def standard_error_of_standard_deviation(data):
    std = np.std(data)
    n = len(data)
    standard_error = (std/(np.sqrt(2*(n-1))))*(1+1/(4*n -5))
    #print(standard_error)
    return standard_error


def figFontSizes(small=12, medium=14, large=16):

    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=small)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title

def main():
    print(main)

if __name__ == '__main__':
    main()
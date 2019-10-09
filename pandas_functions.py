import pandas as pd

def find_quantile(df, column_name, quantile_level):
    # returns the value of specified quantile_level
    # input:
    #   df: dataframe
    #   column_name: str: name of column to find quantile of
    #   quantile_level: float in [0,1]: quantile to find
    return df[column_name].quantile(quantile_level)


def create_percentile_bins(df, col_name, number_of_bins):
    # returns an array of the edges of bins needed to split the data into even sized bins
    # input:
    #   df: dataframe
    #   col_name: str: name of column to find quantile of
    #   number_of_bins: int
    bins = []
    for i in range(number_of_bins):
        bins.append(find_quantile(df, col_name, i/3))
    bins.append(max(df[col_name])+1)
    return bins


def differential_bin(df, column_to_bin, min, max):
    # returns data in a single differential bin
    #input:
    #   df: dataframe
    #   column_to_bin: str: name of column to bin
    #   min: float: lower bin limit (inclusive)
    #   max: float: upper bin limit (exclusive)
    output_df = df[df[column_to_bin] >= min]
    output_df = output_df[output_df[column_to_bin] < max]
    return output_df


def subset_years(df, start_year, end_year, time_column = None):
    if time_column:
        mask = (df[time_column].dt.year >= start_year) & (df[time_column].dt.year < end_year)
    else:
        mask = (df['Peak time'].dt.year >= start_year) & (df['Peak time'].dt.year < end_year)
    return df.loc[mask]


def subset_time(df, time_column, start_time, end_time):
    mask = (df[time_column] >= start_time) & (df[time_column] < end_time)
    return df.loc[mask]


def remove_missing_data(df, col_name, value_for_missing_data):
    new_df = df[df[col_name] != value_for_missing_data]
    return new_df
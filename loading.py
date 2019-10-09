import pandas as pd


def load_sun_spot(path):
    sun_spots = pd.read_table(path, names=['data'])
    sun_spots = sun_spots.data.str.split(expand=True)
    sun_spots.columns = ['year', 'month', 'day', 'frac_year', 'spot_num', 'std', 'observation_num', 'provisional']
    sun_spots.index = pd.to_datetime(sun_spots[['year', 'month', 'day']])
    sun_spots.spot_num = sun_spots.spot_num.astype('float64')
    return sun_spots

def load_am(path):
    headers = ['DATE','TIME','DoY','am','Kpm','an', 'as','Am','An','As','|'] #column names
    dtypes = {'DATE': 'str',
              'TIME': 'str',
              'DoY': 'int',
              'am': 'float',
              'Kpm': 'str',
              'an': 'float',
              'as': 'float',
              'Am': 'float',
              'An': 'float',
              'As': 'float',
              '|': 'str'}
    parse_dates = ['DATE', 'TIME']
    am_data = pd.read_table(path,header = 31,delim_whitespace=True, names=headers, dtype=dtypes, parse_dates=parse_dates ) #import data as table, separate columns at whitespace
    am_data = am_data.drop(['Kpm','an','as','Am','An','As','|'],axis = 1)

    am_data['year'] = am_data['DATE'].dt.year.astype(int)
    am_data['month'] = am_data['DATE'].dt.month.astype(int)
    am_data['day'] = am_data['DATE'].dt.day.astype(int)
    am_data['hour'] = am_data['TIME'].dt.hour.astype(int)

    am_data['datetime'] = pd.to_datetime(am_data[['year','month','day','hour']])
    am_data = am_data[['datetime','DoY','am']]
    return am_data


def load_aaH(path):
    geomag_nums = pd.read_table(path, header=None,
                                delim_whitespace=True)  # import data as table, separate columns at whitespace
    geomag_nums.columns = ['year', 'month', 'day', 'hour', '5', 'geomag', '7', '8', '9', '10', '11']  # name columns
    geomag_nums['datetime'] = pd.to_datetime(geomag_nums[['year', 'month', 'day', 'hour']])  # make index timestamp
    geomag_nums = geomag_nums.drop(['year', 'month', 'day', 'hour', '5', '7', '8', '9', '10', '11'], axis=1)
    return geomag_nums


def load_new_aaH(path):
    headers = ['Year','Month','Day','Hour','Frac of year','aaH', 'aaHN','aaHS','aa/s','aaN/s','aaS/s']
    dtypes = {'Year': 'float',
              'Month': 'float',
              'Day': 'float',
              'Hour': 'float',
              'Frac of year': 'float',
              'aaH': 'float',
              'aaHN': 'float',
              'aaHS': 'float',
              'aa/s': 'float',
              'aaN/s': 'float',
              'aaS/s': 'float'}

    aaH_data = pd.read_table(path, header = 34, delim_whitespace=True, names=headers, dtype=dtypes) #import data as table, separate columns at whitespace
    aaH_data['datetime'] = pd.to_datetime(aaH_data[['Year','Month','Day','Hour']])
    aaH_data = aaH_data[['datetime', 'aaH']]
    return aaH_data


def load_storm_summary(path):
    headers = ['Start time', 'Peak time', 'End time', 'Duration', 'Peak value', 'Peak DoY', 'Lower threshold',
               'Threshold']
    dtypes = {'Start time': 'str', 'Peak time': 'str', 'End time': 'str', 'Duration': 'int', 'Peak value': 'float',
              'Peak DoY': 'int', 'Lower threshold': 'float', 'Threshold': 'float'}
    parse_dates = ['Start time', 'Peak time', 'End time']
    storm_summary = pd.read_table(path, header=None, sep=',', names=headers, dtype=dtypes,
                                  parse_dates=parse_dates)
    return storm_summary


def load_OMNI(path):
    headers = ['Year','DoY','hour','HMF ave','Bx','By','Bz','Proton density','Solar wind speed']
    dtypes = {'Year':'int',
              'DoY':'int',
              'hour':'int',
              'HMF ave':'float',
              'Bx':'float',
              'By':'float',
              'Bz':'float',
              'Proton density':'float',
              'Solar wind speed':'float'}

    OMNI_data = pd.read_table(path,delim_whitespace=True, names=headers, dtype=dtypes) #import data as table, separate columns at whitespace
    OMNI_data =OMNI_data[OMNI_data['HMF ave'] != 999.9]
    OMNI_data = OMNI_data[OMNI_data['Bx'] != 999.9]
    OMNI_data = OMNI_data[OMNI_data['By'] != 999.9]
    OMNI_data = OMNI_data[OMNI_data['Bz'] != 999.9]
    OMNI_data = OMNI_data[OMNI_data['Proton density'] != 999.9]
    OMNI_data = OMNI_data[OMNI_data['Solar wind speed'] != 9999.]
    return OMNI_data


def main():
    print('main')
    return


if __name__ == '__main__':
    main()

#
# Script to load data using Pandas
#

import numpy as np
import pandas as pd


MONTH_START_END_ROW_INDICES = {
    201501: [0, 625456],
    201502: [625457, 1252850],
    201503: [1252851, 1882059],
    201504: [1882060, 2512426],
    201505: [2512427, 3144383],
    201506: [3144384, 3776493],
    201507: [3776494, 4606310],
    201508: [4606311, 5449511],
    201509: [5449512, 6314951],
    201510: [6314952, 7207202],
    201511: [7207203, 8113311],
    201512: [8113312, 9025332],
    201601: [9025333, 9941601],
    201602: [9941602, 10862505],
    201603: [10862506, 11787581],
    201604: [11787582, 12715855],
    201605: [12715856, 13647308]
}


def load_data(filename, yearmonth_start, yearmonth_end):

    """
    Script to load data as pd.DataFrame
    """
    skiprows = MONTH_START_END_ROW_INDICES[yearmonth_start][0]
    nrows = MONTH_START_END_ROW_INDICES[yearmonth_end][1] - skiprows + 1
    df = pd.read_csv(filename, skiprows=range(1, skiprows + 1), nrows=nrows)
    return df


def clean_data_inplace(df):
    """
    Script to clean data in input DataFrame
    """


def process_data_inplace(df):
    """
    Script to process data in input DataFrame
    """

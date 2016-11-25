#
# Script to load data using Pandas
#
import logging
import pandas as pd
import numpy as np

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


def load_data(filename, yearmonths_list, nb_clients=-1):

    """
    Script to load data as pd.DataFrame
    """
    load_dtypes = {"sexo": str,
                   "ind_nuevo": str,
                   "ult_fec_cli_1t": str,
                   "indext": str,
                   "indrel_1mes": str,
                   "conyuemp": str}

    df = pd.DataFrame()
    if len(yearmonths_list) > 0:
        for yearmonth in yearmonths_list:
            skiprows = MONTH_START_END_ROW_INDICES[yearmonth][0]
            nrows = MONTH_START_END_ROW_INDICES[yearmonth][1] - skiprows + 1
            _df = pd.read_csv(filename, dtype=load_dtypes, skiprows=range(1, skiprows + 1), nrows=nrows)
            df = pd.concat([df, _df], axis=0, ignore_index=True)
    else:
        logging.info("-- Read all data from the file : %s" % filename)
        df = pd.read_csv(filename, dtype=load_dtypes)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["renta"] = pd.to_numeric(df["renta"], errors="coerce")
    if nb_clients > 0:
        logging.info("-- Select %s clients" % nb_clients)
        nb_months = len(yearmonths_list)
        clients = df['ncodpers'].value_counts()[df['ncodpers'].value_counts() == nb_months].index.values
        np.random.shuffle(clients)
        clients = clients[:nb_clients]
        df = df[df['ncodpers'].isin(clients)]
    return df


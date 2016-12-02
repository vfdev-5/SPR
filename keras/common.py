
#
# Script to load data using Pandas
#
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

TARGET_LABELS = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def load_data(filename, yearmonth_start, yearmonth_end, nb_clients=-1):

    """
    Script to load data as pd.DataFrame
    """
    load_dtypes = {"sexo": str,
                   "ind_nuevo": str,
                   "ult_fec_cli_1t": str,
                   "indext": str,
                   "indrel_1mes": str,
                   "conyuemp": str}

    skiprows = MONTH_START_END_ROW_INDICES[yearmonth_start][0]
    nrows = MONTH_START_END_ROW_INDICES[yearmonth_end][1] - skiprows + 1
    df = pd.read_csv(filename, dtype=load_dtypes, skiprows=range(1, skiprows + 1), nrows=nrows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if nb_clients > 0:
        nb_months = yearmonth_end - yearmonth_start + 1
        clients = df['ncodpers'].value_counts()[df['ncodpers'].value_counts() == nb_months].index.values
        clients = np.random.choice(clients, nb_clients)
        df = df[df['ncodpers'].isin(clients)]
    return df


def load_data2(filename, yearmonths_list, nb_clients=-1):

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


def to_yearmonth(yearmonthdate_str):
    """
    Convert '2016-01-23' -> 201601
    """
    # yearmonth = int(yearmonth_str[:7].replace('-', ''))
    yearmonth = int(yearmonthdate_str[:4] + yearmonthdate_str[5:7])
    return yearmonth


def _to_ym_dec(ym):
    """
    XXXXYY -> XXXX.ZZ
    ZZ = (YY - 1) * 100.0 / 12.0
    """
    XXXX = int(ym * 0.01)
    YY = int(100 * (ym * 0.01 - XXXX) + 0.5)
    ZZ = (YY - 1) * 100.0 / 12.0
    ym_dec = XXXX + 0.01 * ZZ
    return ym_dec


def _to_ym(ym_dec):
    """
    XXXX.ZZ -> XXXXYY
    """
    XXXX = int(ym_dec)
    ZZ = ym_dec - XXXX
    YY = int(ZZ * 12.0 + 0.5) + 1
    ym = XXXX * 100 + YY
    return ym


def _to_nb_months(ym_dec):
    """
    XXXX.ZZ -> number of months
    """
    nb_years = int(ym_dec)
    zz = ym_dec - nb_years
    return 12 * nb_years + int(zz * 12.0 + 0.5)


def minimal_clean_data_inplace(df):
    """
    Script to clean data in input DataFrame
    """
    # There are some 'unknown' users in train dataset only
    unknown_data_lines = df['sexo'].isnull() & df['age'].isnull() & df['ind_empleado'].isnull() & \
                    df['fecha_alta'].isnull() & df['pais_residencia'].isnull()

    logging.info("- Number of lines with unknown data : %s" % unknown_data_lines.sum())

    # Remove these users as clients
    _clients = df[unknown_data_lines]['ncodpers'].unique()
    bad_lines = df['ncodpers'].isin(_clients)
    df.drop(df[bad_lines].index, inplace=True)

    logging.info("- Number of columns with nan : %s" % df.isnull().any().sum())

    # Remove accent
    df.loc[df['nomprov'] == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA"

    unknown_cols = ['sexo',
                    'ind_empleado',
                    'pais_residencia',
                    'ult_fec_cli_1t',
                    'conyuemp',
                    'canal_entrada',
                    'nomprov',
                    'segmento',
                    'tiprel_1mes',
                    'indrel_1mes']
    # Start with cols -> replace nan with UNKNOWN
    for col in unknown_cols:
        df.loc[df[col].isnull(), col] = "UNKNOWN"

    # Set unknown renta to -99
    df.loc[df['renta'].isnull(), 'renta'] = -99

    # Next `fecha_alta` :
    assert df['fecha_alta'].isnull().sum() == 0, \
        "Need to replace nan in 'fecha_alta', count=%s" % df['fecha_alta'].isnull().sum()

    # **Remove 'tipodom' and 'cod_prov' columns**
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)

    if "ind_nomina_ult1" in df.columns and "ind_nom_pens_ult1" in df.columns:
        # Target labels : `ind_nomina_ult1`, `ind_nom_pens_ult1` : nan -> 0
        # I could try to fill in missing values for products by looking at previous months,
        # but since it's such a small number of values for now I'll take the cheap way out.
        df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
        df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

    # replace 'antiguedad' with the number of months between 'fecha_alta' and 'fecha_dato'
    def _compute_duration(row):
        ym1 = to_yearmonth(row['fecha_alta'])
        ym2 = to_yearmonth(row['fecha_dato'])
        ym_dec1 = _to_ym_dec(ym1)
        ym_dec2 = _to_ym_dec(ym2)
        return _to_nb_months(ym_dec2 - ym_dec1)
    df['antiguedad'] = df.apply(_compute_duration, axis=1)


def minimal_clean_test_data_inplace(df):
    """
    """
    pass


def preprocess_data_inplace(df):
    """
    Script to process data in input DataFrame
    """
    string_data = df.drop(['fecha_dato', 'fecha_alta'], axis=1).select_dtypes(include=["object"])
    for c in string_data.columns:
        le = LabelEncoder()
        le.fit(df[c])
        df[c] = le.transform(df[c])


def get_added_products(current_choice, last_choice):
    """
    current_choice is e.g. [0, 0, 1, 0, ..., 1], of length 24
    last_choice is e.g. [0, 0, 1, 0, ..., 1], of length 24
    """
    real = []
    for i, c in enumerate(current_choice):
        if c == 1:
            if last_choice[i] == 0:
                real.append(i)
    return real

def remove_last_choice(predictions, last_choice):
    """
    predictions is a list of product indices
    last_choice is e.g. [0, 0, 1, 0, ..., 1], of length 24
    """
    out = list(predictions)
    for i, c in enumerate(last_choice):
        if c == 1 and i in out:
            out.remove(i)
    return out
    

def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    
    return score / min(len(actual), k)



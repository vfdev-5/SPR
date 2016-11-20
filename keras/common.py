
#
# Script to load data using Pandas
#
from math import floor, ceil
import logging
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

TARGET_LABELS = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def load_data(filename, yearmonth_start, yearmonth_end):

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
    unknown_users = df['sexo'].isnull() & df['age'].isnull() & df['ind_empleado'].isnull() & \
                    df['fecha_alta'].isnull() & df['pais_residencia'].isnull()

    logging.info("- Number of unknown clients : %s" % unknown_users.sum())

    # **Remove these users** !
    df.drop(df[unknown_users].index, inplace=True)

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
                    'indrel_1mes',
                    'renta']
    # Start with cols -> replace nan with UNKNOWN
    for col in unknown_cols:
        df.loc[df[col].isnull(), col] = "UNKNOWN"

    # Next `fecha_alta` :
    assert df['fecha_alta'].isnull().sum() == 0, \
        "Need to replace nan in 'fecha_alta', count=%s" % df['fecha_alta'].isnull().sum()

    # **Remove 'tipodom' and 'cod_prov' columns**
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)

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


def preprocess_data_inplace(df):
    """
    Script to process data in input DataFrame
    """
    pass
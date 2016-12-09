
#
# Script to load data using Pandas
#
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from math import log

PREPROCESS_LABEL_ENCODERS = defaultdict(dict)

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

TARGET_LABELS = np.array(['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'])

TARGET_LABELS2 = np.array([
    "Saving Account", "Guarantees", "Current Accounts", "Derivada Account", "Payroll Account", "Junior Account",
    "Mas particular Account", "particular Account", "particular Plus Account", "Short-term deposits",
    "Medium-term deposits", "Long-term deposits", "e-account", "Funds", "Mortgage", "Pensions (plan fin)",
    "Loans", "Taxes", "Credit Card", "Securities", "Home Account", "Payroll", "Pensions", "Direct Debit" ])


def targets_str_to_indices(targets_str, **kwargs):
    out = []
    index_map = lambda x: x if 'index_map' not in kwargs else kwargs['index_map'][x]
    for s in targets_str:
        out.append([index_map(i) for i, c in enumerate(s) if int(c) == 1])
    return np.array(out)


def targets_dec_to_indices(targets_dec):
    out = []
    for v in targets_dec:
        print v, 
        s = decimal_to_dummies(v)
        print s, 
        ind = [i for i, c in enumerate(s) if int(c) == 1]
        out.append(ind)
        print ind 
    
    return np.array(out) 


def targets_to_labels(targets, tl):
    out = []
    tl = np.array(TARGET_LABELS2)
    for t in targets:
        out.append(tl[np.where(t > 0)])
    return out


def target_str_to_labels(targets_str, tl):
    indices = targets_str_to_indices(targets_str)
    return targets_indices_to_labels(indices)


def targets_indices_to_labels(targets_indices, tl):
    out = []
    tl = np.array(tl)
    for i in targets_indices:
        out.append(tl[i])
    return out


def dummies_to_decimal(row):
    output = ''
    for v in row.values:
        output += str(int(v))
    return log(int(output, 2)+1)


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
    if nb_clients > 0 or nb_clients == 'max':
        logging.info("-- Select %s clients" % nb_clients)
        nb_months = len(yearmonths_list)
        clients = df['ncodpers'].value_counts()[df['ncodpers'].value_counts() == nb_months].index.values
        np.random.shuffle(clients)
        if isinstance(nb_clients, int) and nb_clients < len(clients):
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
    
    # Convert 'ind_nuevo' to int
    df['ind_nuevo'] = df['ind_nuevo'].astype(int)
       
    # Remove floating point at string indrel_1mes
    df['indrel_1mes'] = df['indrel_1mes'].apply(lambda x: str(int(float(x))) if len(x) == 3 else x)

    if "ind_nomina_ult1" in df.columns and "ind_nom_pens_ult1" in df.columns:
        # Target labels : `ind_nomina_ult1`, `ind_nom_pens_ult1` : nan -> 0
        # I could try to fill in missing values for products by looking at previous months,
        # but since it's such a small number of values for now I'll take the cheap way out.
        df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
        df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

    # replace 'antiguedad' with the number of months between 'fecha_alta' and 'fecha_dato'
    func1 = lambda x: _to_ym_dec(to_yearmonth(x))
    func2 = lambda x: max(_to_nb_months(x), 0) 

    v1 = df['fecha_dato'].apply(func1)
    v2 = df['fecha_alta'].apply(func1)
    v3 = (v1 - v2).apply(func2)
    df.loc[:, 'antiguedad'] = v3
    
    # Replace 'ult_fec_cli_1t' by current nb of months from fecha_dato, if negative, set to zero
    mask = df['ult_fec_cli_1t'] == 'UNKNOWN'
    df.loc[mask, 'ult_fec_cli_1t'] = df[mask]['fecha_dato']
    v1 = df['fecha_dato'].apply(func1)
    v2 = df['ult_fec_cli_1t'].apply(func1)
    v3 = (v1 - v2).apply(func2)
    df.loc[:, 'ult_fec_cli_1t'] = v3
    
    
def encode(encoder, df, drop_cols=()):
    logging.debug("-- Call encode --")
    
    if len(drop_cols) > 0:
        string_data = df.drop(drop_cols, axis=1).select_dtypes(include=["object"])
    else:
        string_data = df.select_dtypes(include=["object"])
        
    for c in string_data.columns:
        unique_vals = df[c].unique()
        # initialize :
        if len(encoder[c]) == 0 :
            logging.debug("- Initialize : %s" % c)
            # fit :
            for i, v in enumerate(unique_vals):
                logging.debug("-1 Add : {} -> {}".format(v, i))
                encoder[c][v] = i
        
        # check :
        isin_mask = np.in1d(unique_vals, encoder[c].keys())
        logging.debug("- Check : %s" % c)
        logging.debug("-- isin_mask: {}".format(isin_mask.all()))
        if not isin_mask.all():
            logging.debug("- Check is failed : need to add more")
            next_val = np.max(encoder[c].values()) + 1
            logging.debug("-- next_val: %i" % next_val)
            # fit :
            for i, v in enumerate(unique_vals[~isin_mask]):                
                logging.debug("-2 Add : {} -> {}".format(v, next_val + i))                
                encoder[c][v] = next_val + i                
        
        # transform :
        logging.debug("- Transform : %s" % c)
        df.loc[:, c] = df[c].apply(lambda x: encoder[c][x])

                              
def preprocess_data_inplace(df):
    """
    Script to process data in input DataFrame
    """
    encode(PREPROCESS_LABEL_ENCODERS, df, ['fecha_dato', 'fecha_alta'])
                              
        
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


def map7_score(y, y_pred, clients_last_choice):
    """
    y is an ndarray of indicies: e.g. [[2, 13], [2], [24, 14, 5], ...]
    y_pred is an ndarray of indicies: e.g. [[2, 13], [2], [24, 14, 5], ...]
    clients_last_choice is an ndarray: e.g. [0, 0, 1, 0, ..., 1], of length 24
    """
    map7 = 0.0
    for last_choice, targets, products in zip(clients_last_choice, y, y_pred):
        added_products = remove_last_choice(targets, last_choice)
        predictions = remove_last_choice(products, last_choice)
        score = apk(added_products, predictions)    
        map7 += score            

    map7 /= len(y)
    logging.info('-- Predicted map7 score: {}'.format(map7))
    return map7



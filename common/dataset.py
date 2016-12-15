
import os
import numpy as np
import pandas as pd

import logging

from utils import _to_ym_dec, _to_ym, to_yearmonth, dummies_to_str, dummies_to_decimal
from utils import load_data2, minimal_clean_data_inplace, preprocess_data_inplace
from utils import TARGET_LABELS

TRAIN_FILE_PATH = os.path.join("..", "data", "train_ver2.csv")
TEST_FILE_PATH = os.path.join("..", "data", "test_ver2.csv")
LC_TARGET_LABELS = np.array(['lc_' + t for t in TARGET_LABELS])
TARGET_LABELS_FRQ = np.array([t + '_frq' for t in TARGET_LABELS])
TARGET_LABELS_DIFF = np.array([t + '_diff' for t in TARGET_LABELS])

LOGCOUNT_DICT = None


def load_train_yearmonth(yearmonth, n_clients='max'):
    """
    Method to load preprocessed data corresponding to `yearmonth` from the train dataset.
    :param yearmonth:
    :param n_clients: integer > 0 or 'max' corresponds to a number of clients to load with 'diff' fields containing  
    or -1 to load all client and do not provide 'diff' fields  
    :return: pd.DataFrame
    """

    filename = "train_%s__%s.csv" % (str(yearmonth), str(n_clients))
    filepath = '../data/generated/' + filename

    if os.path.exists(filepath) and os.path.isfile(filepath):
        train_df = pd.read_csv('../data/generated/' + filename)
        return train_df

    # else:
    logging.info("- Load training data : ")
    yearmonth_list = _ym_list_to_load(yearmonth)
    train_df = _load(yearmonth_list, n_clients)

    _months_ym_map = _check(train_df, yearmonth_list)
    # Add target_str and last client choice
    _process_add_target_str(train_df)
    _process_add_clc(train_df, yearmonth, _months_ym_map)
    _check_clc(train_df, yearmonth_list)
    logcount_dict = _get_logcount_dict(train_df)

    logging.info("-- Add logCount columns")
    _add_logcount(train_df, logcount_dict)

    logging.info("-- Add logDecimal columns")
    _add_logdecimal(train_df)

    logging.info("-- Transform age/renta/logdiff")
    _process2(train_df)

    logging.info("-- Add target values frequencies")
    _add_target_frq_values(train_df)

    logging.info("-- Add target diff")
    _add_target_diff_values(train_df)
    train_df.to_csv(filepath, index=False, index_label=False)

    return train_df


def load_trainval(train_yearmonths_list, val_yearmonths_list=(), train_nb_clients=-1, val_nb_clients='max'):
    """
    Method to load train/validation datasets
    :param train_yearmonths_list: should be sorted in ascending order without duplicates
    :param val_yearmonth:
    :param train_nb_clients:
    :return:
    cleaned, processed, with 'diff' and client last choice targets
    train_df, val_df (if val_yearmonths_list is not [])
    or
    train_df if val_yearmonths_list is not specified
    """

    global LOGCOUNT_DICT

    # ###################
    # Load training data
    # ###################
    logging.info("- Load training data : ")
    yearmonth_list = _ym_list_to_load(train_yearmonths_list)
    train_df = _load(yearmonth_list, train_nb_clients)

    _months_ym_map = _check(train_df, yearmonth_list)
    # Add target_str and last client choice
    _process_add_target_str(train_df)
    _process_add_clc(train_df, train_yearmonths_list, _months_ym_map)
    _check_clc(train_df, yearmonth_list)

    if len(val_yearmonths_list) > 0:
        # ###################
        # Load validation data
        # ###################
        logging.info("- Load validation data")
        yearmonth_list = _ym_list_to_load(val_yearmonths_list)
        val_df = _load(yearmonth_list, val_nb_clients)

        _months_ym_map = _check(val_df, yearmonth_list)
        # Add target_str and last client choice
        _process_add_target_str(val_df)
        _process_add_clc(val_df, val_yearmonths_list, _months_ym_map)
        _check_clc(val_df, yearmonth_list)

        # ###################
        # Insert logCount :
        # ###################
        LOGCOUNT_DICT = _compute_logcount_dict(train_df, val_df)
    else:
        val_df = None
        LOGCOUNT_DICT = _get_logcount_dict(train_df)


    logging.info("-- Add logCount columns")
    _add_logcount(train_df, LOGCOUNT_DICT)
    if val_df is not None:
        _add_logcount(val_df, LOGCOUNT_DICT)

    # ###################
    # Insert logDecimal :
    # ###################
    logging.info("-- Add logDecimal columns")
    _add_logdecimal(train_df)
    if val_df is not None:
        _add_logdecimal(val_df)

    # ###################
    # age/renta/logdiff :
    # ###################
    logging.info("-- Transform age/renta/logdiff")
    _process2(train_df)
    if val_df is not None:
        _process2(val_df)
    
    # ###################
    # Add target values frequencies
    # ###################
    logging.info("-- Add target values frequencies")
    _add_target_frq_values(train_df)
    if val_df is not None:
        _add_target_frq_values(val_df)

    # ###################
    # Add target diff
    # ###################
    logging.info("-- Add target diff")
    _add_target_diff_values(train_df)
    if val_df is not None:
        _add_target_diff_values(val_df)

    if val_df is not None:
        return train_df, val_df
    return train_df


def load_test():
    """
    Method to load whole test data
    :return: pd.DataFrame
    """


def load_train_test(train_yearmonths_list):
    """
    Method to load full train and test data
    :return: train_df, test_df with last client choice columns
    """
    # ###################
    # Load training file :

    logging.info("- Load training data : ")
    yearmonth_list = _ym_list_to_load(train_yearmonths_list)
    train_df = _load(yearmonth_list, 'max')

    _months_ym_map = _check(train_df, yearmonth_list)
    # Add target_str and last client choice
    _process_add_target_str(train_df)
    _process_add_clc(train_df, train_yearmonths_list, _months_ym_map)
    _check_clc(train_df, yearmonth_list)

    # Update LOGCOUNT_DICT :
    global LOGCOUNT_DICT
    if LOGCOUNT_DICT is None:
        LOGCOUNT_DICT = _get_logcount_dict(train_df)
    else:
        LOGCOUNT_DICT = _update_logcount_dict(LOGCOUNT_DICT, train_df)

    # Insert logCount :
    logging.info("-- Add logCount columns")
    _add_logcount(train_df, LOGCOUNT_DICT)

    # Insert logDecimal :
    logging.info("-- Add logDecimal columns")
    _add_logdecimal(train_df)

    # age/renta/logdiff :
    logging.info("-- Transform age/renta/logdiff")
    _process2(train_df)

    # Add target frequencies
    logging.info("-- Add target frequencies")
    _add_target_frq_values(train_df)

    # Add target diff
    logging.info("-- Add target diff")
    _add_target_diff_values(train_df)

    # ###################
    # Load test file :
    logging.info("- Load test data : ")
    last_month_train_df = _load([201605], -1)
    test_df = _load([], -1, filepath=TEST_FILE_PATH)
    # Select only clients from test file
    test_clients = test_df['ncodpers'].unique()
    last_month_train_df = last_month_train_df[last_month_train_df['ncodpers'].isin(test_clients)]
    test_df = pd.concat([last_month_train_df, test_df], axis=0, ignore_index=True)

    yearmonth_list = [201605, 201606]
    _months_ym_map = _check(test_df, yearmonth_list)
    # Add last client choice
    train_month_mask = test_df['fecha_dato'] == _months_ym_map[201605]
    _process_add_target_str(test_df, train_month_mask)
    _process_add_clc(test_df, [201606], _months_ym_map)
    _check_clc(test_df, yearmonth_list)
    # Add target frequencies
    logging.info("-- Add target frequencies")
    _add_target_frq_values(test_df, train_month_mask)

    # test_month_mask = test_df['fecha_dato'] == _months_ym_map[201606]
    # test_df = test_df[test_month_mask].drop(TARGET_LABELS, axis=1)

    return train_df, test_df


def get_age_group_index(age):
    if age < 10:
        return -3
    elif age < 15:
        return -2
    elif age < 18:
        return -1
    elif age < 23:
        return 0
    elif age < 25:
        return 1
    elif age < 27:
        return 2
    elif age < 28:
        return 3
    elif age < 32:
        return 4
    elif age < 37:
        return 5
    elif age < 42:
        return 6
    elif age < 47:
        return 7
    elif age < 52:
        return 8
    elif age < 57:
        return 9
    elif age < 60:
        return 10
    elif age < 65:
        return 11
    elif age < 70:
        return 12
    elif age < 75:
        return 13
    elif age < 80:
        return 14
    else:
        return 15


def get_income_group_index(income):
    if income < 0:
        return -1
    elif income < 45542.97:
        return 1
    elif income < 57629.67:
        return 2
    elif income < 68211.78:
        return 3
    elif income < 78852.39:
        return 4
    elif income < 90461.97:
        return 5
    elif income < 103855.23:
        return 6
    elif income < 120063.00:
        return 7
    elif income < 141347.49:
        return 8
    elif income < 173418.36:
        return 9
    elif income < 234687.12:
        return 10
    else:
        return 11

    
def _add_target_frq_values(df, mask=None):
    for t, nt in zip(TARGET_LABELS, TARGET_LABELS_FRQ):
        counts = df[t].value_counts()
        counts = counts/counts.sum()
        values = df.loc[:, t].unique()
        for v in values:
            m = df.loc[:, t] == v
            if mask is not None:
                df.loc[mask & m, nt] = counts[v]
            else:
                df.loc[m, nt] = counts[v]
            # if v > 0:  # Set frequency of choosing current product
                # df.loc[:, nt] = counts[v]


def _add_target_diff_values(df):
    mask = ~df['lc_targets_str'].isnull()
    for t, lct, nt in zip(TARGET_LABELS, LC_TARGET_LABELS, TARGET_LABELS_DIFF):
        diff = df[mask][t] - df[mask][lct]
        diff[diff < 0] = 0
        df.loc[mask, nt] = diff
        df.loc[~mask, nt] = -99999


def _get_prev_ym(ym):
    return _to_ym(_to_ym_dec(ym) - _to_ym_dec(2))

    
# def _add_diff_inplace(df, prev_ym_mask, ym_mask):
#     """
#     df should be imperatively sorted by clients in order to subtract and assign correctly
#     """
#     tmp_df = df[['fecha_dato', 'ncodpers']]
#     tmp_df.loc[:, 't'] = df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)
#     v1 = tmp_df[ym_mask][['ncodpers', 't']]
#     v2 = tmp_df[prev_ym_mask][['ncodpers', 't']]
#     assert len(v1) == len(v2), "Length of current month and previous month are not equal"
#     v2.index = v1.index
#     df.loc[ym_mask, 'diff'] = v1['t'] - v2['t']


def _add_clc_inplace(df, prev_ym_mask, ym_mask):
    """
    df should be imperatively sorted by clients in order to assign correctly
    """
    clients_last_choice = df[prev_ym_mask][['ncodpers'] + TARGET_LABELS]
    # Prepend "lc_" to column names
    clc_new_cols = clients_last_choice.columns.values
    clc_new_cols[1:] = "lc_" + clc_new_cols[1:]

    clients_last_choice.index = df[ym_mask].index
    for c in clc_new_cols[1:]:
        df.loc[ym_mask, c] = clients_last_choice[c]

    df.loc[ym_mask, 'lc_targets_str'] = df[ym_mask][LC_TARGET_LABELS].apply(dummies_to_str, axis=1)


def _check_clc(df, months):
    # Ensure that last client choice is correct
    for i, m in enumerate(months[:-1]):
        m2 = months[i + 1]
        tmask1 = df['fecha_dato'] == m
        tmask2 = df['fecha_dato'] == m2

        if df[tmask2]['lc_targets_str'].isnull().sum() == 0:
            assert (df[tmask1]['targets_str'].values == df[tmask2][
                'lc_targets_str'].values).all(), "Clients Last choice columns are not set correctly"


def _load(_yearmonths_list, nb_clients, filepath=TRAIN_FILE_PATH):
    logging.info("- Load data : {}".format(_yearmonths_list))
    df = load_data2(filepath, _yearmonths_list, nb_clients)
    minimal_clean_data_inplace(df)
    preprocess_data_inplace(df)
    return df


def _check(df, yearmonth_list):
    months = df['fecha_dato'].unique()
    clients = df['ncodpers'].unique()

    assert len(clients) == (df['ncodpers'].value_counts() == len(yearmonth_list)).sum()
    ll = len(clients)
    months_ym_map = {}
    for m in months:
        l = len(df[df['fecha_dato'] == m]['ncodpers'].unique())
        assert l == ll, "Number of clients should be identical for all monthes. (%s, %s, %s)" % (m, l, ll)
        months_ym_map[to_yearmonth(m)] = m
    return months_ym_map


def _process_add_target_str(df, mask=None):
    if mask is None:
        df.loc[:, 'targets_str'] = df[TARGET_LABELS].apply(dummies_to_str, axis=1)
    else:
        df.loc[mask, 'targets_str'] = df[mask][TARGET_LABELS].apply(dummies_to_str, axis=1)


def _process_add_clc(df, _yearmonths_list, _months_ym_map):
    # Imperatively sort by clients in order to subtract and assign correctly
    df.sort_values(['ncodpers', 'fecha_dato'], inplace=True)

    for ym in _yearmonths_list:
        logging.info("-- Process date : {}".format(ym))
        prev_ym = _get_prev_ym(ym)
        prev_ym_mask = df['fecha_dato'] == _months_ym_map[prev_ym]
        ym_mask = df['fecha_dato'] == _months_ym_map[ym]
        _add_clc_inplace(df, prev_ym_mask, ym_mask)


def _get_logcount_dict(df):
    logcount_dict = df['targets_str'].value_counts()#.apply(lambda x: np.log(x + 1))
    logcount_dict /= logcount_dict.sum()
    return logcount_dict
        
        
def _update_logcount_dict(logcount_dict1, df):
    logging.info("-- Compute logCount dictionary")
    logcount_dict2 = _get_logcount_dict(df)
    return _merge_logcount_dicts(logcount_dict1, logcount_dict2)


def _merge_logcount_dicts(logcount_dict1, logcount_dict2):
    targets_str1 = logcount_dict1.index.unique()
    targets_str2 = logcount_dict2.index.unique()
    targets_str21 = list(set(targets_str1) - set(targets_str2))
    targets_str12 = list(set(targets_str2) - set(targets_str1))
    logcount_dict1 = pd.concat(
        [logcount_dict1, pd.Series(np.zeros((len(targets_str12))), index=targets_str12)])
    logcount_dict2 = pd.concat(
        [logcount_dict2, pd.Series(np.zeros((len(targets_str21))), index=targets_str21)])
    logcount_dict = (logcount_dict1 + logcount_dict2).sort_values(ascending=False)
    logcount_dict /= logcount_dict.sum()
    return logcount_dict


def _compute_logcount_dict(_train_df, _val_df):
    logging.info("-- Compute logCount dictionary")
    #train_targets_str = _train_df['targets_str'].unique()
    #val_targets_str = _val_df['targets_str'].unique()
    train_logcount_dict = _get_logcount_dict(_train_df)
    val_logcount_dict = _get_logcount_dict(_val_df)
    return _merge_logcount_dicts(train_logcount_dict, val_logcount_dict)


def _add_logcount(df, logcount_dict):
    months = df['fecha_dato'].unique()
    for m in months:
        logging.info("-- Process month : %s" % m)
        tmask = df['fecha_dato'] == m
        current_logcount_dict = _get_logcount_dict(df[tmask])

        df.loc[tmask, 'targets_logcount1'] = df[tmask]['targets_str'].apply(lambda x: current_logcount_dict[x])
        df.loc[tmask, 'targets_logcount2'] = df[tmask]['targets_str'].apply(lambda x: logcount_dict[x])

        if df[tmask]['lc_targets_str'].isnull().sum() == 0:
            df.loc[tmask, 'lc_targets_logcount2'] = df[tmask]['lc_targets_str'].apply(lambda x: logcount_dict[x])
            df.loc[tmask, 'targets_logcount2_diff'] = df.loc[tmask, 'targets_logcount2'] - df.loc[
                tmask, 'lc_targets_logcount2']

    df.loc[df['targets_logcount2_diff'].isnull(), 'targets_logcount2_diff'] = -99999
    df.loc[df['lc_targets_logcount2'].isnull(), 'lc_targets_logcount2'] = -99999


def _add_logdecimal(df):
    df.loc[:, 'targets_logDec'] = df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)
    mask = ~df['lc_targets_str'].isnull()
    df.loc[mask, 'lc_targets_logDec'] = df[mask][LC_TARGET_LABELS].apply(dummies_to_decimal, axis=1)
    df.loc[mask, 'targets_diff'] = df.loc[mask, 'targets_logDec'] - df.loc[mask, 'lc_targets_logDec']
    df.loc[:, 'targets_logDec'] = df.loc[:, 'targets_logDec'].apply(lambda x: np.log(x + 1))
    df.loc[mask, 'lc_targets_logDec'] = df.loc[:, 'targets_logDec'].apply(lambda x: np.log(x + 1))
    mask = df['lc_targets_str'].isnull()
    df.loc[mask, 'targets_diff'] = -99999
    df.loc[mask, 'lc_targets_logDec'] = -99999


def _process2(df):
    mask = ~df['targets_diff'].isin([-99999])
    df.loc[:, 'age'] = df['age'].apply(get_age_group_index)
    df.loc[:, 'renta'] = df['renta'].apply(get_income_group_index)
    df.loc[mask, 'targets_logdiff'] = df[mask]['targets_diff'] \
        .apply(lambda x: np.sign(x) * np.log(np.abs(x) + 1))
    df.loc[~mask, 'targets_logdiff'] = -99999


def _ym_list_to_load(ym_list):
    """
    Append previous month to the each yearmonth
    """
    out = []
    for ym in ym_list:
        # get the previous month
        prev_ym = _get_prev_ym(ym)
        if len(out) and out[-1] == prev_ym:
            out += [ym]
        else:
            out += [prev_ym, ym]
    return out

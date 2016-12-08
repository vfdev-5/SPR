
import os
import numpy as np
import pandas as pd

import logging

from common import _to_ym_dec, _to_ym, to_yearmonth
from common import load_data2, minimal_clean_data_inplace, preprocess_data_inplace
from common import TARGET_LABELS, targets_str_to_indices, targets_dec_to_indices

TRAIN_FILE_PATH = os.path.join("..", "data", "train_ver2.csv")
TEST_FILE_PATH = os.path.join("..", "data", "test_ver2.csv")
LC_TARGET_LABELS = ['lc_' + t for t in TARGET_LABELS]


def _get_prev_ym(ym):
    return _to_ym(_to_ym_dec(ym) - _to_ym_dec(2))


def dummies_to_str(row):
    output = ''
    for v in row.values:
        output += str(int(v))
    return output


def dummies_to_decimal(row):
    output = dummies_to_str(row)
    return int(output, 2)


def decimal_to_dummies(value):
    return '{:024b}'.format(value)


def _add_diff_inplace(df, prev_ym_mask, ym_mask):
    """
    df should be imperatively sorted by clients in order to subtract and assign correctly
    """
    tmp_df = df[['fecha_dato', 'ncodpers']]
    tmp_df.loc[:, 't'] = df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)
    v1 = tmp_df[ym_mask][['ncodpers', 't']]
    v2 = tmp_df[prev_ym_mask][['ncodpers', 't']]
    assert len(v1) == len(v2), "Length of current month and previous month are not equal"
    v2.index = v1.index
    df.loc[ym_mask, 'diff'] = v1['t'] - v2['t']


def _add_clc_inplace(df, prev_ym_mask, ym_mask):
    """
    df should be imperatively sorted by clients in order to subtract and assign correctly
    """
    clients_last_choice = df[prev_ym_mask][['ncodpers'] + TARGET_LABELS]
    # Prepend "lc_" to column names
    clc_new_cols = clients_last_choice.columns.values
    clc_new_cols[1:] = "lc_" + clc_new_cols[1:]

    clients_last_choice.index = df[ym_mask].index
    for c in clc_new_cols[1:]:
        df.loc[ym_mask, c] = clients_last_choice[c]

    df.loc[ym_mask, 'lc_targets_str'] = df[ym_mask][LC_TARGET_LABELS].apply(dummies_to_str, axis=1)


def load_trainval(train_yearmonths_list, val_yearmonth, train_nb_clients=-1):
    """
    Method to load train/validation datasets
    :param train_yearmonths_list: should be sorted in ascending order without duplicates
    :param val_yearmonth:
    :param train_nb_clients:
    :return:
    cleaned, processed, with 'diff' and client last choice targets
    train_df, val_df
    """

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

    def _load(_yearmonths_list, nb_clients):
        yearmonth_list = _ym_list_to_load(_yearmonths_list)
        logging.info("- Load data : {}".format(yearmonth_list))
        df = load_data2(TRAIN_FILE_PATH, yearmonth_list, nb_clients)
        minimal_clean_data_inplace(df)
        preprocess_data_inplace(df)
        return yearmonth_list, df

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

    def _process1(df, _yearmonths_list):
        # Imperatively sort by clients in order to subtract and assign correctly
        df.sort_values(['ncodpers', 'fecha_dato'], inplace=True)
        df.loc[:, 'targets_str'] = df[TARGET_LABELS].apply(dummies_to_str, axis=1)
        for ym in _yearmonths_list:
            logging.info("-- Process date : {}".format(ym))
            prev_ym = _get_prev_ym(ym)
            prev_ym_mask = df['fecha_dato'] == months_ym_map[prev_ym]
            ym_mask = df['fecha_dato'] == months_ym_map[ym]
            _add_clc_inplace(df, prev_ym_mask, ym_mask)

    def _compute_logcount_dict(_train_df, _val_df):
        logging.info("-- Compute logCount dictionary")
        train_targets_str = _train_df['targets_str'].unique()
        val_targets_str = _val_df['targets_str'].unique()

        train_logcount_dict = _train_df['targets_str'].value_counts().apply(lambda x: np.log(x + 1))
        train_logcount_dict /= train_logcount_dict.sum()
        val_logcount_dict = _val_df['targets_str'].value_counts().apply(lambda x: np.log(x + 1))
        val_logcount_dict /= val_logcount_dict.sum()

        targets_str_to_val = list(set(train_targets_str) - set(val_targets_str))
        targets_str_to_train = list(set(val_targets_str) - set(train_targets_str))

        train_logcount_dict = pd.concat(
            [train_logcount_dict, pd.Series(np.zeros((len(targets_str_to_train))), index=targets_str_to_train)])
        val_logcount_dict = pd.concat(
            [val_logcount_dict, pd.Series(np.zeros((len(targets_str_to_val))), index=targets_str_to_val)])

        logcount_dict = (train_logcount_dict + val_logcount_dict).sort_values(ascending=False)
        logcount_dict /= logcount_dict.sum()
        return logcount_dict

    def _add_logcount(df, months, logcount_dict):
        for m in months:
            logging.info("-- Process month : %s" % m)
            tmask = df['fecha_dato'] == m
            current_logcount_dict = df[tmask]['targets_str'].value_counts().apply(lambda x: np.log(x + 1))
            current_logcount_dict /= current_logcount_dict.sum()

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
        df.loc[:, 'renta'] = train_df['renta'].apply(get_income_group_index)
        df.loc[mask, 'targets_logdiff'] = df[mask]['targets_diff']\
            .apply(lambda x: np.sign(x) * np.log(np.abs(x) + 1))
        df.loc[~mask, 'targets_logdiff'] = -99999

    # ###################
    # Load training data
    # ###################
    logging.info("- Load training data")
    yearmonth_list, train_df = _load(train_yearmonths_list, train_nb_clients)

    months_ym_map = _check(train_df, yearmonth_list)
    _process1(train_df, train_yearmonths_list)

    # ###################
    # Load validation data
    # ###################
    logging.info("- Load validation data")
    yearmonth_list, val_df = _load(val_yearmonth, 'max')

    months_ym_map = _check(val_df, yearmonth_list)
    _process1(val_df, val_yearmonth)

    # ###################
    # Insert logCount :
    # ###################
    months_ym_map = {}
    months = list(set(train_df['fecha_dato'].unique()) | set(val_df['fecha_dato'].unique()))
    for m in months:
        months_ym_map[to_yearmonth(m)] = m
    train_months = train_df['fecha_dato'].unique()
    val_months = val_df['fecha_dato'].unique()

    logcount_dict = _compute_logcount_dict(train_df, val_df)

    logging.info("-- Add logCount columns")
    _add_logcount(train_df, train_months, logcount_dict)
    _add_logcount(val_df, val_months, logcount_dict)

    # ###################
    # Insert logDecimal :
    # ###################
    logging.info("-- Add logDecimal columns")
    _add_logdecimal(train_df)
    _add_logdecimal(val_df)

    # ###################
    # age/renta/logdiff :
    # ###################
    logging.info("-- Transform age/renta/logdiff")
    _process2(train_df)
    _process2(val_df)

    return train_df, val_df


def load_test():
    """
    Method to load test data
    :return: test_df with
    """
    test_df = load_data2(TEST_FILE_PATH, [])
    minimal_clean_data_inplace(test_df)
    preprocess_data_inplace(test_df)
    test_df = test_df.sort_values(['ncodpers'])
    return test_df


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
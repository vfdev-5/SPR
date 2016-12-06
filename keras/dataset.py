
import os
import numpy as np

import logging

from common import _to_ym_dec, _to_ym, to_yearmonth
from common import load_data2, minimal_clean_data_inplace, preprocess_data_inplace
from common import TARGET_LABELS

TRAIN_FILE_PATH = os.path.join("..", "data", "train_ver2.csv")
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
    df.loc[ym_mask, 'lc_targets_dec'] = df[ym_mask][LC_TARGET_LABELS].apply(dummies_to_decimal, axis=1)


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

    # ###################
    # Load training data
    # ###################
    yearmonth_list = _ym_list_to_load(train_yearmonths_list)
    logging.info("- Load training data : {}".format(yearmonth_list))
    train_df = load_data2(TRAIN_FILE_PATH, yearmonth_list, train_nb_clients)
    minimal_clean_data_inplace(train_df)
    preprocess_data_inplace(train_df)

    months = train_df['fecha_dato'].unique()
    clients = train_df['ncodpers'].unique()
    assert len(clients) == (train_df['ncodpers'].value_counts() == len(yearmonth_list)).sum()
    ll = len(clients)
    months_ym_map = {}
    for m in months:
        l = len(train_df[train_df['fecha_dato'] == m]['ncodpers'].unique())
        assert l == ll, "Number of clients should be identical for all monthes. (%s, %s, %s)" % (m, l, ll)
        months_ym_map[to_yearmonth(m)] = m

    # Imperatively sort by clients in order to subtract and assign correctly
    train_df = train_df.sort_values(['ncodpers', 'fecha_dato'])

    for ym in train_yearmonths_list:
        logging.info("-- Process date : {}".format(ym))
        prev_ym = _get_prev_ym(ym)
        prev_ym_mask = train_df['fecha_dato'] == months_ym_map[prev_ym]
        ym_mask = train_df['fecha_dato'] == months_ym_map[ym]

        train_df.loc[:, 'targets_str'] = train_df[TARGET_LABELS].apply(dummies_to_str, axis=1)
        train_df.loc[:, 'targets_dec'] = train_df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)

        _add_diff_inplace(train_df, prev_ym_mask, ym_mask)
        _add_clc_inplace(train_df, prev_ym_mask, ym_mask)

    # ###################
    # Load validation data
    # ###################
    yearmonth_list = _ym_list_to_load(val_yearmonth)
    logging.info("- Load validation : {}".format(yearmonth_list))
    val_df = load_data2(TRAIN_FILE_PATH, yearmonth_list, 'max')
    minimal_clean_data_inplace(val_df)
    preprocess_data_inplace(val_df)

    months = val_df['fecha_dato'].unique()
    clients = val_df['ncodpers'].unique()
    assert len(clients) == (val_df['ncodpers'].value_counts() == len(yearmonth_list)).sum()
    ll = len(clients)
    months_ym_map = {}
    for m in months:
        l = len(val_df[val_df['fecha_dato'] == m]['ncodpers'].unique())
        assert l == ll, "Number of clients should be identical for all monthes. (%s, %s, %s)" % (m, l, ll)
        months_ym_map[to_yearmonth(m)] = m

    # Imperatively sort by clients in order to subtract and assign correctly
    val_df = val_df.sort_values(['ncodpers', 'fecha_dato'])
    for ym in val_yearmonth:
        logging.info("-- Process date : {}".format(ym))
        prev_ym = _get_prev_ym(ym)
        prev_ym_mask = val_df['fecha_dato'] == months_ym_map[prev_ym]
        ym_mask = val_df['fecha_dato'] == months_ym_map[ym]

        val_df.loc[:, 'targets_str'] = val_df[TARGET_LABELS].apply(dummies_to_str, axis=1)
        val_df.loc[:, 'targets_dec'] = val_df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)

        _add_diff_inplace(val_df, prev_ym_mask, ym_mask)
        _add_clc_inplace(val_df, prev_ym_mask, ym_mask)

    mask = (~val_df['diff'].isnull())
    assert (val_df[mask]['diff'] == val_df[mask]['targets_dec'] - val_df[mask]['lc_targets_dec']).all(), "Something is wrong"

    return train_df, val_df

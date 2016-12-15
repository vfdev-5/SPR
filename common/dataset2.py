import os
import numpy as np
import pandas as pd

import logging

from utils import _to_ym_dec, _to_ym, to_yearmonth, dummies_to_str, dummies_to_decimal
from utils import load_data2, minimal_clean_data_inplace, preprocess_data_inplace
from utils import TARGET_LABELS, FEATURES_NAMES

TRAIN_FILE_PATH = os.path.join("..", "data", "train_ver2.csv")
TEST_FILE_PATH = os.path.join("..", "data", "test_ver2.csv")

TARGET_GROUPS = [
    [2, 4, 12, 18, 21, 22, 23],
    [7, 8, 11, 13, 17, 19],
    [5, 6, 14, 15, 20],
    [0, 1, 3, 9, 16],
]

TARGET_GROUP_DEC_LABELS = ['targets_dec_g%i' % i for i in range(len(TARGET_GROUPS))]


def load_X(yearmonth, supp_yearmonths_list=(), n_clients=-1):
    """
    :param yearmonth:
    :param supp_yearmonths_list:
    :param n_clients: integer > 0 or '-1'
    :return: pd.DataFrame
    """

    filename = "dataset_X_%s_%s__%i.csv" % (
        str(yearmonth),
        '+'.join([str(ym) for ym in supp_yearmonths_list]),
        n_clients
    )
    filepath = '../data/generated/' + filename
    if os.path.exists(filepath) and os.path.isfile(filepath):
        logging.info("- Found already generated file, load it")
        X = pd.read_csv('../data/generated/' + filename)
        return X
    # else:
    fname = TEST_FILE_PATH if yearmonth == 201606 else TRAIN_FILE_PATH
    logging.info("- Load file : %s, yearmonth=%i, n_clients=%i" % (fname, yearmonth, n_clients))
    X = load_data2(fname, [yearmonth], n_clients)
    minimal_clean_data_inplace(X)
    preprocess_data_inplace(X)
    yearmonth_map = _get_yearmonth_map(X)

    yearmonth_str = yearmonth_map[yearmonth]
    logging.info("- Process targets for one yearmonth : %i" % yearmonth)
    process_targets(X, yearmonth_str)
    process_features(X, yearmonth_str)

    # ref_clients = X['ncodpers'].unique()
    # for ym in supp_yearmonths_list:
    #     fname = TEST_FILE_PATH if ym == 201606 else TRAIN_FILE_PATH
    #     X_ym = load_data2(fname, [ym])



    # processed_targets_labels =
    # output_columns = ['fecha_dato', 'ncodpers'] + \
    #     FEATURES_NAMES + TARGET_GROUP_LABELS
    # return X[output_columns]
    return X


def process_features(df, yearmonth_str):
    mask = df['fecha_dato'] == yearmonth_str
    replace_income(df, mask)
    replace_age(df, mask)


def replace_age(df, mask=None):
    if mask is None:
        df.loc[:, 'age'] = df['age'].apply(get_age_group_index)
    else:
        df.loc[mask, 'age'] = df[mask]['age'].apply(get_age_group_index)


def replace_income(df, mask=None):
    if mask is None:
        df.loc[:, 'renta'] = df['renta'].apply(get_income_group_index)
    else:
        df.loc[mask, 'renta'] = df[mask]['renta'].apply(get_income_group_index)


def process_targets(df, yearmonth_str, label_index=0):
    mask = df['fecha_dato'] == yearmonth_str
    targets_str_label = 'targets_str' if label_index == 0 else 'targets_str_%i' % label_index
    add_targets_str(df, targets_str_label, mask=mask)
    targets_logdecimal_label = 'targets_logdec' if label_index == 0 else 'targets_logdec_%i' % label_index
    add_targets_logdecimal(df, targets_logdecimal_label, mask=mask)
    for group, label in zip(TARGET_GROUPS, TARGET_GROUP_DEC_LABELS):
        if label_index > 0:
            label += '_%i' % label_index
        add_targets_group_decimal(df, group, label, mask)


def _get_yearmonth_map(df):
    months = df['fecha_dato'].unique()
    yearmonth_map = {}
    for m in months:
        yearmonth_map[to_yearmonth(m)] = m
    return yearmonth_map


def add_targets_str(df, field_name='targets_str', mask=None):
    if mask is None:
        df.loc[:, field_name] = df[TARGET_LABELS].apply(dummies_to_str, axis=1)
    else:
        df.loc[mask, field_name] = df[mask][TARGET_LABELS].apply(dummies_to_str, axis=1)


def add_targets_group_decimal(df, group, field_name, mask=None):
    nptl = np.array(TARGET_LABELS)
    if mask is None:
        df.loc[:, field_name] = df[nptl[group]].apply(dummies_to_decimal, axis=1)
    else:
        df.loc[mask, field_name] = df[mask][nptl[group]].apply(dummies_to_decimal, axis=1)


def add_targets_logdecimal(df, field_name='targets_logdec', mask=None):
    if mask is None:
        df.loc[:, field_name] = df[TARGET_LABELS].apply(dummies_to_decimal, axis=1)
        df.loc[:, field_name] = df.loc[:, field_name].apply(lambda x: np.log(x + 1))
    else:
        df.loc[mask, field_name] = df[mask][TARGET_LABELS].apply(dummies_to_decimal, axis=1)
        df.loc[mask, field_name] = df.loc[mask, field_name].apply(lambda x: np.log(x + 1))


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
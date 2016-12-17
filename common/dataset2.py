import os
import numpy as np
import pandas as pd

import logging
from time import time

from utils import _to_ym_dec, _to_nb_months, to_yearmonth, dummies_to_str, dummies_to_decimal
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

TARGET_LABELS_FRQ = np.array([t + '_frq' for t in TARGET_LABELS])

NP_FEATURES_NAMES = np.array(FEATURES_NAMES)
match_features_groups = [
    FEATURES_NAMES,
    NP_FEATURES_NAMES[[0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]],  # [u'ind_empleado', u'pais_residencia', u'sexo', u'age',  u'ind_nuevo', u'antiguedad', u'indrel', u'ult_fec_cli_1t', u'indrel_1mes', u'nomprov', u'ind_actividad_cliente', u'renta', u'segmento']
    NP_FEATURES_NAMES[[1, 2, 3, 4, 6, 7, 8, 15, 16, 17, 18]],  # [u'pais_residencia', u'sexo', u'age',  u'ind_nuevo', u'indrel', u'ult_fec_cli_1t', u'indrel_1mes', u'nomprov', u'ind_actividad_cliente', u'renta', u'segmento']
    NP_FEATURES_NAMES[[1, 2, 3, 4, 6, 7, 8, 15, 16, 18]],  # [u'pais_residencia', u'sexo', u'age', u'ind_nuevo', u'indrel', uult_fec_cli_1t', u'indrel_1mes', u'nomprov', u'ind_actividad_cliente', u'segmento']
    NP_FEATURES_NAMES[[1, 2, 3, 6, 7, 8, 15, 16, 18]],  # [u'pais_residencia', u'sexo', u'age', u'indrel', uult_fec_cli_1t', u'indrel_1mes', u'nomprov', u'ind_actividad_cliente', u'segmento']
    NP_FEATURES_NAMES[[1, 2, 3, 6, 7, 8, 15]],  # [u'pais_residencia', u'sexo', u'age', u'indrel', u'ult_fec_cli_1t', u'indrel_1mes', u'nomprov']
    ## Last resort
    NP_FEATURES_NAMES[[1, 2, 3, 6, 15]],  # [u'pais_residencia', u'sexo', u'age', u'indrel', u'nomprov', u'segmento']
    NP_FEATURES_NAMES[[1, 2, 3, 15]],  # [u'pais_residencia', u'sexo', u'age', u'nomprov', u'segmento']
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
    # yearmonth_map = _get_yearmonth_map(X)

    # yearmonth_str = yearmonth_map[yearmonth]
    logging.info("- Process targets for one yearmonth : %i" % yearmonth)
    process_targets(X)
    process_features(X)

    X = X.sort_values(['ncodpers'])
    ref_clients = X['ncodpers'].unique()

    for i, ym in enumerate(supp_yearmonths_list):
        logging.info("- Add a supplementary data : %i" % ym)
        fname = TEST_FILE_PATH if ym == 201606 else TRAIN_FILE_PATH
        X_ym = load_data2(fname, [ym])
        minimal_clean_data_inplace(X_ym)
        preprocess_data_inplace(X_ym)
        process_features(X_ym)

        clients = X_ym['ncodpers'].unique()
        missing_clients = np.array(list(set(ref_clients) - set(clients)))
        if missing_clients.shape[0] > 0:
            X_ym = add_missing_clients(X_ym, ym, X, yearmonth, ref_clients)

        X_ym = X_ym[X_ym['ncodpers'].isin(ref_clients)].sort_values(['ncodpers'])
        X_ym.index = X.index
        assert (X['ncodpers'] == X_ym['ncodpers']).all(), "Clients are not alignable"

        process_targets(X_ym, label_index=i+1)
        compute_diffs(X, X_ym, label_index=i+1)
        add_targets_columns(X, X_ym)

    return X


def features_to_str(features):
    return '_'.join([str(int(f)) for f in features])


def process_features(df, yearmonth_str=''):
    mask = df['fecha_dato'] == yearmonth_str if len(yearmonth_str) > 0 else None
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


def process_targets(df, yearmonth_str='', label_index=0):
    mask = df['fecha_dato'] == yearmonth_str if len(yearmonth_str) > 0 else None

    targets_str_label = 'targets_str' if label_index == 0 else 'targets_str_%i' % label_index
    add_targets_str(df, targets_str_label, mask=mask)

    targets_logdecimal_label = 'targets_logdec' if label_index == 0 else 'targets_logdec_%i' % label_index
    add_targets_logdecimal(df, targets_logdecimal_label, mask=mask)

    for group, label in zip(TARGET_GROUPS, TARGET_GROUP_DEC_LABELS):
        if label_index > 0:
            label += '_%i' % label_index
        add_targets_group_decimal(df, group, label, mask)

    add_target_frq_values(df, mask, label_index)


def _get_yearmonth_map(df):
    months = df['fecha_dato'].unique()
    yearmonth_map = {}
    for m in months:
        yearmonth_map[to_yearmonth(m)] = m
    return yearmonth_map


def compute_diffs(df1, df2, label_index):
    compute_targets_diff(df1, df2, label_index)
    compute_targets_group_diff(df1, df2, label_index)


def compute_targets_diff(df1, df2, label_index):
    field_name = 'targets_diff_%i' % label_index
    df1.loc[:, field_name] = df1[TARGET_LABELS].apply(dummies_to_decimal, axis=1) - df2[TARGET_LABELS].apply(
        dummies_to_decimal, axis=1)


def compute_targets_group_diff(df1, df2, label_index):
    for label1 in TARGET_GROUP_DEC_LABELS:
        label2 = label1 + '_%i' % label_index
        field_name = label1 + '_diff_%i' % label_index
        df1.loc[:, field_name] = df1[label1] - df2[label2]


def add_targets_columns(df1, df2):
    cols = list(set(df2.columns) - set(FEATURES_NAMES + ['fecha_alta', 'fecha_dato', 'ncodpers'] + TARGET_LABELS))
    for c in cols:
        df1.loc[:, c] = df2.loc[:, c]


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


def add_target_frq_values(df, mask=None, label_index=0):
    label_index = '' if label_index == 0 else '_%i' % label_index
    for t, nt in zip(TARGET_LABELS, TARGET_LABELS_FRQ):
        nt += label_index
        counts = df[t].value_counts()
        counts = counts/counts.sum()
        values = df.loc[:, t].unique()
        for v in values:
            m = df.loc[:, t] == v
            if mask is not None:
                df.loc[mask & m, nt] = counts[v]
            else:
                df.loc[m, nt] = counts[v]


def add_missing_clients(X_ym_, ym, X_, ref_ym, ref_clients):

    delta_months = _to_nb_months(_to_ym_dec(ref_ym) - _to_ym_dec(ym))

    def _update_seniority(row):
        row['antiguedad'] -= delta_months
        return row

    for fg in match_features_groups:
        clients = X_ym_['ncodpers'].unique()
        missing_clients = np.array(list(set(ref_clients) - set(clients)))
        logging.info("- Compute missing clients : {}/{}".format(missing_clients.shape[0], ref_clients.shape[0]))
        missing_clients_mask = X_['ncodpers'].isin(missing_clients)
        if TARGET_LABELS[5] in fg:
            x_features_str = X_[missing_clients_mask][fg].apply(_update_seniority, axis=1).apply(features_to_str, axis=1)
        else:
            x_features_str = X_[missing_clients_mask][fg].apply(features_to_str, axis=1)
        x_ym_features_str = X_ym_[fg].apply(features_to_str, axis=1)
        logging.info("- Match and add")
        X_ym_ = match_add_clients((X_[missing_clients_mask], x_features_str), (X_ym_, x_ym_features_str), delta_months)

        if X_ym_['ncodpers'].isin(missing_clients).sum() == X_['ncodpers'].isin(missing_clients).sum():
            break

    clients = X_ym_['ncodpers'].unique()
    missing_clients = np.array(list(set(ref_clients) - set(clients)))
    logging.info("- Compute missing clients : {}/{}".format(missing_clients.shape[0], ref_clients.shape[0]))

    # If remains missing clients, setup them with zero targets
    if missing_clients.shape[0] > 0:
        logging.warn("There are still missing clients ! Setup them with zero targets")
        print "There are still missing clients"
        missing_clients_mask = X_['ncodpers'].isin(missing_clients)
        supp_data_df = X_.loc[missing_clients_mask, ['ncodpers', 'fecha_alta', 'fecha_dato'] + FEATURES_NAMES].copy()
        supp_data_df.loc[:, 'fecha_dato'] = X_ym_['fecha_dato'].unique()[0]
        supp_data_df.loc[:, 'antiguedad'] -= delta_months

        for t in TARGET_LABELS:
            supp_data_df[t] = 0

        X_ym_ = pd.concat([X_ym_, supp_data_df], ignore_index=True)

    return X_ym_


def match_add_clients(ref_data, out_data, delta_months):
    _X, x_features_str = ref_data
    _X_ym_, x_ym_features_str = out_data

    tic = time()

    common_x_ym_features_str = x_ym_features_str[x_ym_features_str.isin(x_features_str)]
    common_x_ym_features_str = common_x_ym_features_str.sort_values()
    common_x_features_str = x_features_str[x_features_str.isin(common_x_ym_features_str)].sort_values()
    gb = common_x_features_str.groupby(common_x_features_str)
    gb_ym = common_x_ym_features_str.groupby(common_x_ym_features_str)

    common_x_ym_indices = gb_ym.apply(lambda x: x.index)
    feature_index_map = gb.transform(lambda x: common_x_ym_indices[x])

    supp_data_df = _X.loc[common_x_features_str.index, ['ncodpers', 'fecha_alta', 'fecha_dato'] + FEATURES_NAMES].copy()
    supp_data_df.loc[:, 'fecha_dato'] = _X_ym_['fecha_dato'].unique()[0]
    supp_data_df.loc[:, 'antiguedad'] -= delta_months

    for t in TARGET_LABELS:
        supp_data_df[t] = 0

    for index, value in feature_index_map.iteritems():
        targets = _X_ym_.loc[value, TARGET_LABELS].mean().apply(lambda x: int(np.ceil(x)))
        supp_data_df.loc[index, TARGET_LABELS] = targets

    _X_ym_ = pd.concat([_X_ym_, supp_data_df], ignore_index=True)
    logging.info("-- Match and add elapsed time: {}".format(time() - tic))

    assert not (_X_ym_['ncodpers'].value_counts() > 1).any(), "Something is wrong : {}".format(
        _X_ym_['ncodpers'].value_counts() > 1)
    return _X_ym_


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
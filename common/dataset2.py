import os
import numpy as np
import pandas as pd

import logging
from time import time

from utils import _to_ym_dec, _to_nb_months, to_yearmonth, _get_prev_ym, _get_year_january
from utils import dummies_to_str, dummies_to_decimal
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

FREQ_TARGET_LABELS = np.array(['freq_' + t for t in TARGET_LABELS])
LAST_TARGET_LABELS = np.array(['last_' + t for t in TARGET_LABELS])
ADDED_TARGET_LABELS = np.array(['added_' + t for t in TARGET_LABELS])

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


def TARGET_GROUPS_DEC(i):
    return [g + '_%i' % i for g in TARGET_GROUP_DEC_LABELS]


def PROCESSED_TARGETS(i):
    out = ['targets_str_%i' % i, 'targets_logdec_%i' % i]
    out += TARGET_GROUPS_DEC(i)
    out += [f + '_%i' % i for f in FREQ_TARGET_LABELS]
    return out


def DIFF_TARGETS(i, j):
    out = ['diff_targets_dec_%i%i' % (i, j)]
    out += DIFF_TARGET_GROUPS_DEC(i, j)
    return out


def DIFF_TARGET_GROUPS_DEC(i, j):
    return ['diff_' + g + '_%i%i' % (i, j) for g in TARGET_GROUP_DEC_LABELS]


def load_test():
    """

    :return: test X, Y (only LAST_TARGET_LABELS and last_targets_str) dataframe
    """
    def _get_XY(df):
        X = df[['ncodpers', 'fecha_dato', 'fecha_alta'] +
               FEATURES_NAMES +
               PROCESSED_TARGETS(1) +
               PROCESSED_TARGETS(2) +
               PROCESSED_TARGETS(3) +
               PROCESSED_TARGETS(4) +
               DIFF_TARGETS(1, 2) +
               DIFF_TARGETS(1, 3) +
               DIFF_TARGETS(1, 4)
               ]

        Y = df[['last_targets_str'] + LAST_TARGET_LABELS.tolist()]
        return X, Y

    filename = "test.csv"
    filepath = '../data/generated/' + filename
    if os.path.exists(filepath) and os.path.isfile(filepath):
        logging.info("- Found already generated file, load it")
        df = pd.read_csv('../data/generated/' + filename)
        X, Y = _get_XY(df)
        return X, Y
    # else:

    # load all test data:
    fname = TEST_FILE_PATH
    yearmonths_list = []
    logging.info("- Load file : %s" % (fname))
    df = load_data2(fname, [])
    minimal_clean_data_inplace(df)
    preprocess_data_inplace(df)
    ref_clients = df['ncodpers'].unique()

    # load data from train dataset
    fname = TRAIN_FILE_PATH
    yearmonth = 201606
    yearmonths_list = [_get_prev_ym(yearmonth)]
    logging.info("- Load file : %s, yearmonth=%i" % (fname, yearmonths_list[0]))
    df1 = load_data2(fname, yearmonths_list)
    minimal_clean_data_inplace(df1)
    preprocess_data_inplace(df1)
    df1 = df1[df1['ncodpers'].isin(ref_clients)]

    df = df.sort_values(['ncodpers', 'fecha_dato'])
    df1 = df1.sort_values(['ncodpers', 'fecha_dato'])
    df1.index = df.index
    assert (df['ncodpers'] == df1['ncodpers']).all(), "Clients are not alignable"

    # Transform main month:
    process_features(df)

    # Append products from the previous month:
    append_columns(df, df1[TARGET_LABELS], LAST_TARGET_LABELS)
    add_targets_str(df, 'last_targets_str', target_labels=LAST_TARGET_LABELS)

    # Process targets of yearmonth - 1
    process_targets(df1, label_index=1)
    append_columns(df, df1[PROCESSED_TARGETS(1)])

    assert not df.isnull().any().all(), "Some nan values appeared"

    # Load supplementary data
    supp_yearmonths_list = [_get_prev_ym(yearmonths_list[0]), _get_year_january(yearmonth), yearmonth - 100]
    ll = 'max'
    index_offset = 2
    for i, ym in enumerate(supp_yearmonths_list):
        logging.info("- Add a supplementary data : %i" % ym)
        df_ym = load_data2(fname, [ym], ll)
        minimal_clean_data_inplace(df_ym)
        preprocess_data_inplace(df_ym)
        #process_features(df_ym)

        df_ym = add_zero_missing_clients(df_ym, ym, df, yearmonth, ref_clients)

        df_ym = df_ym[df_ym['ncodpers'].isin(ref_clients)].sort_values(['ncodpers'])
        df_ym.index = df.index
        assert (df['ncodpers'] == df_ym['ncodpers']).all(), "Clients are not alignable"

        process_targets(df_ym, label_index=i+index_offset)
        append_columns(df, df_ym[PROCESSED_TARGETS(i+index_offset)])

        fn = 'diff_targets_dec_%i%i' % (1, i+index_offset)
        df.loc[:, fn] = compute_targets_diff(df1[TARGET_LABELS], df_ym[TARGET_LABELS])

        res = compute_targets_group_diff(df1[TARGET_GROUPS_DEC(1)],
                                         df_ym[TARGET_GROUPS_DEC(i+index_offset)])
        append_columns(df, res, DIFF_TARGET_GROUPS_DEC(1, i+index_offset))

    logging.info("Store computed data as file : %s" % filepath)
    df.to_csv(filepath, index=False, index_label=False)
    X, Y = _get_XY(df)
    return X, Y


def load_trainval(yearmonth, n_clients='max'):
    """

    Method to load train/validation datasets

    X = [Processed Features](yearmonth) +
        [Processed Targets](yearmonth - 1) +
        [Processed Targets](yearmonth - 1, Jan) +
        [Processed Targets](yearmonth - 1, yearmonth - 100)

    Y = [Targets](yearmonth) +
        [Targets](yearmonth - 1) +
        [Diff targets](yearmonth, yearmonth - 1)

    :param yearmonth: year-month data on which targets to train
    :param n_clients: integer > 0 or 'max'
    :return: X, Y dataframes
    """

    def _get_XY(df):
        X = df[['ncodpers', 'fecha_dato', 'fecha_alta'] +
               FEATURES_NAMES +
               PROCESSED_TARGETS(1) +
               PROCESSED_TARGETS(2) +
               PROCESSED_TARGETS(3) +
               PROCESSED_TARGETS(4) +
               DIFF_TARGETS(1, 2) +
               DIFF_TARGETS(1, 3) +
               DIFF_TARGETS(1, 4)
               ]

        Y = df[['targets_str', 'last_targets_str', 'added_targets_str', 'added_targets_dec'] +
               TARGET_LABELS + LAST_TARGET_LABELS.tolist() + ADDED_TARGET_LABELS.tolist()
        ]
        return X, Y

    filename = "trainval_%s__%s.csv" % (str(yearmonth), str(n_clients))
    filepath = '../data/generated/' + filename
    if os.path.exists(filepath) and os.path.isfile(filepath):
        logging.info("- Found already generated file, load it")
        df = pd.read_csv('../data/generated/' + filename)
        X, Y = _get_XY(df)
        return X, Y
    # else:

    assert yearmonth < 201606, "Yearmonth should be less 201606"

    fname = TRAIN_FILE_PATH

    # load main month and the previous one:
    yearmonths_list = [yearmonth, _get_prev_ym(yearmonth)]
    logging.info("- Load file : %s, yearmonth=%i, n_clients=%s" % (fname, yearmonth, str(n_clients)))
    df = load_data2(fname, yearmonths_list, n_clients)
    minimal_clean_data_inplace(df)
    preprocess_data_inplace(df)

    # Separate data into [main month] and [previous month]
    months_ym_map = _get_yearmonth_map(df)
    df = df.sort_values(['ncodpers', 'fecha_dato'])
    mask0 = df['fecha_dato'] == months_ym_map[yearmonths_list[0]]
    mask1 = df['fecha_dato'] == months_ym_map[yearmonths_list[1]]
    df1 = df[mask1]
    df = df[mask0]
    df1.index = df.index
    assert (df['ncodpers'] == df1['ncodpers']).all(), "Clients are not alignable"

    # Transform main month:
    process_features(df)
    add_targets_str(df)

    # Append products from the previous month:
    append_columns(df, df1[TARGET_LABELS], LAST_TARGET_LABELS)
    add_targets_str(df, 'last_targets_str', target_labels=LAST_TARGET_LABELS)

    # Compute added products from previous month
    compute_added_products(df)
    add_targets_str(df, 'added_targets_str', target_labels=ADDED_TARGET_LABELS)

    # Process targets of yearmonth - 1
    process_targets(df1, label_index=1)
    append_columns(df, df1[PROCESSED_TARGETS(1)])

    assert not df.isnull().any().all(), "Some nan values appeared"

    # Load supplementary data
    ref_clients = df['ncodpers'].unique()
    supp_yearmonths_list = [_get_prev_ym(yearmonths_list[1]), _get_year_january(yearmonth), yearmonth - 100]
    #ll = len(ref_clients)
    ll = 'max'
    index_offset = 2
    for i, ym in enumerate(supp_yearmonths_list):
        logging.info("- Add a supplementary data : %i" % ym)
        df_ym = load_data2(fname, [ym], ll)
        minimal_clean_data_inplace(df_ym)
        preprocess_data_inplace(df_ym)
        #process_features(df_ym)

        df_ym = add_zero_missing_clients(df_ym, ym, df, yearmonth, ref_clients)

        df_ym = df_ym[df_ym['ncodpers'].isin(ref_clients)].sort_values(['ncodpers'])
        df_ym.index = df.index
        assert (df['ncodpers'] == df_ym['ncodpers']).all(), "Clients are not alignable"

        process_targets(df_ym, label_index=i+index_offset)
        append_columns(df, df_ym[PROCESSED_TARGETS(i+index_offset)])

        fn = 'diff_targets_dec_%i%i' % (1, i+index_offset)
        df.loc[:, fn] = compute_targets_diff(df1[TARGET_LABELS], df_ym[TARGET_LABELS])

        res = compute_targets_group_diff(df1[TARGET_GROUPS_DEC(1)],
                                         df_ym[TARGET_GROUPS_DEC(i+index_offset)])
        append_columns(df, res, DIFF_TARGET_GROUPS_DEC(1, i+index_offset))

    logging.info("Store computed data as file : %s" % filepath)
    df.to_csv(filepath, index=False, index_label=False)
    X, Y = _get_XY(df)
    return X, Y


def compute_added_products(df):
    assert TARGET_LABELS[0] in df.columns and LAST_TARGET_LABELS[0] in df.columns, \
        "TARGET_LABELS and LAST_TARGET_LABELS should exist in df"
    for c1, c2, nc in zip(TARGET_LABELS, LAST_TARGET_LABELS, ADDED_TARGET_LABELS):
        diff = df[c1] - df[c2]
        diff[diff < 0] = 0
        df.loc[:, nc] = diff
    df.loc[:, 'added_targets_dec'] = df[ADDED_TARGET_LABELS].sum(axis=1)


def append_columns(df1, df2, new_columns=()):
    assert (df1.index == df2.index).all(), "Indices are not aligned"
    if len(new_columns) > 0:
        assert len(df2.columns) == len(new_columns), "Columns are not properly matched"
    else:
        new_columns = df2.columns
    for c, nc in zip(df2.columns, new_columns):
        df1.loc[:, nc] = df2[c]


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


# def compute_diffs(df1, df2, label_index):
#     compute_targets_diff(df1, df2, label_index)
#     compute_targets_group_diff(df1, df2, label_index)


# def compute_targets_diff(df1, df2, label_index):
#     field_name = 'targets_diff_%i' % label_index
#     df1.loc[:, field_name] = df1[TARGET_LABELS].apply(dummies_to_decimal, axis=1) -\
#                              df2[TARGET_LABELS].apply(dummies_to_decimal, axis=1)

def compute_targets_diff(df1, df2):
    return df1.apply(dummies_to_decimal, axis=1) - df2.apply(dummies_to_decimal, axis=1)


def compute_targets_group_diff(df1, df2):
    assert len(df1.columns) == len(df2.columns), "Columns length is not the same"
    res = pd.DataFrame()
    for c1, c2 in zip(df1.columns, df2.columns):
        res.loc[:, '%s-%s' % (c1, c2)] = df1[c1] - df2[c2]
    return res


# def add_targets_columns(df1, df2):
#     cols = list(set(df2.columns) - set(FEATURES_NAMES + ['fecha_alta', 'fecha_dato', 'ncodpers'] + TARGET_LABELS))
#     for c in cols:
#         df1.loc[:, c] = df2.loc[:, c]


def add_targets_str(df, field_name='targets_str', mask=None, target_labels=TARGET_LABELS):
    if mask is None:
        df.loc[:, field_name] = df[target_labels].apply(dummies_to_str, axis=1)
    else:
        df.loc[mask, field_name] = df[mask][target_labels].apply(dummies_to_str, axis=1)


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
    for t, nt in zip(TARGET_LABELS, FREQ_TARGET_LABELS):
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


def add_zero_missing_clients(X_ym_, ym, X_, ref_ym, ref_clients):
    delta_months = _to_nb_months(_to_ym_dec(ref_ym) - _to_ym_dec(ym))

    clients = X_ym_['ncodpers'].unique()
    missing_clients = np.array(list(set(ref_clients) - set(clients)))
    logging.info("-- Compute missing clients : {}/{}".format(missing_clients.shape[0], ref_clients.shape[0]))

    # If remains missing clients, setup them with zero targets
    if missing_clients.shape[0] > 0:
        logging.info("--- Setup them with zero targets")
        missing_clients_mask = X_['ncodpers'].isin(missing_clients)
        supp_data_df = X_.loc[missing_clients_mask, ['ncodpers', 'fecha_alta', 'fecha_dato'] + FEATURES_NAMES].copy()
        supp_data_df.loc[:, 'fecha_dato'] = X_ym_['fecha_dato'].unique()[0]
        supp_data_df.loc[:, 'antiguedad'] -= delta_months

        for t in TARGET_LABELS:
            supp_data_df[t] = 0

        X_ym_ = pd.concat([X_ym_, supp_data_df], ignore_index=True)

    return X_ym_


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

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import sys
sys.path.append("../common")
from utils import map7_score, TARGET_LABELS, targets_str_to_indices


def prepare_to_fit(X_train, Y_train):
    x_train = X_train.values
    x_train = StandardScaler().fit_transform(x_train)
    y_train = Y_train.values
    return x_train, y_train


def prepare_to_test(X_val, Y_val=None):
    x_val = X_val.values
    x_val = StandardScaler().fit_transform(x_val)
    y_val = Y_val.values if Y_val is not None else None
    return x_val, y_val


def train_all(X_train, Y_train,
              samples_masks_list,
              features_masks_dict,
              labels_masks_dict,
              models_dict,
              **kwargs):
    """
    Method to train a set of estimators from `models_dict`
    on the data obtained after applying all combinations of
    - samples mask from `samples_masks_list`,
    - features mask from `features_masks_dict` and
    - labels mask from `labels_masks_dict`

    :X_train: a pd.DataFrame of training dataset containing features, `(nb_samples, nb_features)`
    :Y_train: a pd.DataFrame of training dataset containing labels, `(nb_samples, nb_labels)`
    :samples_masks_list: a list, e.g. `[samples_mask_1, samples_mask_2, ...]` with samples_mask_i is a function to produce a boolean pd.DataFrame . Used only for training.
    If an empty list is providede, all samples are used for training

    :features_masks_list: a dictionary, e.g. `{fm1_name: features_mask_1, fm2_name: features_mask_2, ...]` with `features_mask_i` is a list of feature column names. They can oversect.
        Feature mask can be None to indicate all features.
    :labels_masks_list: a dictionary, e.g.`{lm1_name: labels_mask_1, lm2_name: labels_mask_2, ...}` with `labels_mask_i` is a list of labels column names. They can oversect.
        Label mask can be None to indicate all labels.
    :models_dict: a dictionary of functions to create a model, e.g. `{'rf': create_RF, 'nn': create_NN, 'gbt': create_GBT}`

    In `kwargs` it is possible to define :
        :verbose: True/False
        :models_pipelines: (optional) a dictionary, e.g. `{model_name: [(feature_mask_name, label_mask_name), ...]}`.
        It defines specific connection between a model and features/labels to train on. Useful, when a model can not train on
        all types of labels. It is possible to specify only one mask name `feature_mask_name` or `label_mask_name` with None, e.g. (None, label_mask_name).
        If models_pipelines is defined and a model is not added into models_pipelines. It will be used on all combinations of feature mask/label mask.

    :return: a list of trained estimators, e.g. `[([features_mask_name, labels_mask_name, model_name], estimator_object, fit_accuracy), ...]`
    """
    logging.debug("---------------")
    logging.info("-- Train all --")
    verbose = False if 'verbose' not in kwargs else kwargs['verbose']
    models_pipelines = None if 'models_pipelines' not in kwargs else kwargs['models_pipelines']

    if len(samples_masks_list) == 0:
        samples_masks_list.append(lambda df: df.index.isin(df.index[:]))

    estimators = []

    for i, samples_mask in enumerate(samples_masks_list):
        mask = samples_mask(X_train)
        X_train_ = X_train[mask]
        Y_train_ = Y_train[mask]

        for features_mask_name in features_masks_dict:
            features_mask = features_masks_dict[features_mask_name]
            X_train__ = X_train_[features_mask] if features_mask is not None else X_train_
            for labels_mask_name in labels_masks_dict:
                labels_mask = labels_masks_dict[labels_mask_name]
                Y_train__ = Y_train_[labels_mask] if labels_mask is not None else Y_train_
                logging.info("-- Process : sample_mask={}/{}, features_mask={}, labels_mask={}"
                             .format(len(X_train_), len(X_train), features_mask_name, labels_mask_name))
                x_train, y_train = prepare_to_fit(X_train__, Y_train__)
                logging.debug("--- Train data shapes : {}, {}".format(x_train.shape, y_train.shape))

                if y_train.shape[1] == 1:
                    # avoid DataConversionWarning
                    y_train = y_train.ravel()

                for model_name in models_dict:
                    logging.debug("-- Create the model : %s" % model_name)

                    can_fit = True
                    if models_pipelines is not None and model_name in models_pipelines:
                        can_fit = False
                        pipelines = models_pipelines[model_name]
                        # pipelines = [(feature_mask_name, label_mask_name), ...]
                        for _features_mask_name, _labels_mask_name in pipelines:
                            b1 = _features_mask_name is None
                            b2 = _labels_mask_name is None
                            assert not (b1 and b2), "Feature_mask_name and label_mask_name can not be both None"
                            if _features_mask_name is not None and _features_mask_name == features_mask_name:
                                b1 = True
                            if _labels_mask_name is not None and _labels_mask_name == labels_mask_name:
                                b2 = True
                            can_fit = b1 and b2
                            if can_fit:
                                break

                    if not can_fit:
                        continue

                    estimator = models_dict[model_name](input_shape=x_train.shape, output_shape=y_train.shape)
                    logging.debug("--- Fit the model")
                    estimator.fit(x_train, y_train)
                    acc = estimator.score(x_train, y_train)
                    logging.info("--- Score : model='%s', fit accuracy : %f" % (model_name, acc))
                    estimators.append(([features_mask_name, labels_mask_name, model_name], estimator, acc))

                    if verbose:
                        logging.info("\n\n\t -- Feature ranking : -- \n\n")
                        logging.info(
                            "--- Estimator : {}, {}, {}".format(features_mask_name, labels_mask_name, model_name))
                        importances = estimator.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        for f in range(len(features_mask)):
                            logging.info("%d. feature %d '%s' (%f)" % (
                            f + 1, indices[f], features_mask[indices[f]], importances[indices[f]]))
    return estimators


def probas_to_indices(Y_probas, **kwargs):
    mask = (~Y_probas.isnull()).any()
    all_columns = Y_probas.columns
    Y_probas = Y_probas[mask[mask].index]
    y_probas = Y_probas.as_matrix()

    threshold = 0.5 if 'threshold' not in kwargs else kwargs['threshold']
    n_highest = 7 if 'n_highest' not in kwargs else kwargs['n_highest']

    y_probas[y_probas < threshold] = 0.0
    predicted_added_products = np.argsort(y_probas, axis=1)
    predicted_added_products = predicted_added_products[:, ::-1][:, :n_highest]
    out = []
    index_map = np.where(all_columns.isin(mask[mask].index))[0]
    for i, t in enumerate(predicted_added_products):
        out.append([index_map[j] for j in t if y_probas[i, j] > 0.0])
    return np.array(out)


def merge_predictions(Y_probas, y_probas, labels_mask, mode='sum', **kwargs):
    if mode == 'max':
        Y_probas.loc[:, labels_mask] = np.maximum(Y_probas.loc[:, labels_mask], y_probas)
    elif mode == 'sum':
        Y_probas.loc[:, labels_mask] = Y_probas.loc[:, labels_mask] + y_probas
    else:
        raise Exception("Existing data merge is not yet implemented")

    return Y_probas


def predict_all(estimators, X_val, features_masks_dict, labels_masks_dict, labels, **kwargs):
    """
    Method to compute predictions using `estmators` from a test dataset `X_val`

    :estimators: a list of object of type ([features_mask_name, labels_mask_name, model_name], estimator_object, fit_accuracy)
    :X_val: a pd.DataFrame of shape `(nb_samples, nb_features)`
    :features_masks_dict: a dictionary of features masks (see train_all method)
    :labels_masks_dict: a dictionary of labels masks (see train_all method)
    :labels: a list of all available labels for the output

    In `kwargs` it is possible to define :
        :transform_proba_func: a function to transform computed probabilities into a custom form.
        Function signature should be `foo(Y_probas, **kwargs)`

        :verbose: True/False

    :return:
        if `transform_proba_func` is not defined, predicted label probabilites `Y_probas` (pd.DataFrame) are returned.
        Thus, output is an ndarray of shape (nb_samples, len(labels)).

        if `transform_proba_func` is defined, then output is an ndarray of shape `(nb_samples, ...)`, the output of `transform_proba_func`.

    """
    logging.debug("-----------------")
    logging.info("-- Predict all --")
    verbose = False if 'verbose' not in kwargs else kwargs['verbose']
    return_probas = False if 'return_probas' not in kwargs else kwargs['return_probas']
    transform_proba_func = None if 'transform_proba_func' not in kwargs else kwargs['transform_proba_func']

    Y_probas = pd.DataFrame(index=X_val.index, columns=labels)
    Y_probas = Y_probas.fillna(0.0)
    for estimator in estimators:
        # estimator is ([features_mask_name, labels_mask_name, model_name], estimator_object)
        features_mask_name, labels_mask_name, model_name = estimator[0]
        features_mask = features_masks_dict[features_mask_name]
        labels_mask = labels_masks_dict[labels_mask_name]
        logging.info("-- Process : model={}, features_mask={}, labels_mask={}".format(model_name, features_mask_name,
                                                                                      labels_mask_name))

        x_val, _ = prepare_to_test(X_val[features_mask])
        logging.debug("--- Test data shapes : {}".format(x_val.shape))

        y_probas = estimator[1].predict(x_val)
        logging.debug("--- Predicted data shape : {}".format(y_probas.shape))
        if y_probas.dtype == np.int:
            y_probas = y_probas.astype(np.float)
        if len(y_probas.shape) == 1:
            y_probas = y_probas.reshape((y_probas.shape[0], 1))
        # multiply by accuracy :
        y_probas *= estimator[2]
        Y_probas = merge_predictions(Y_probas, y_probas, labels_mask, **kwargs)

    if transform_proba_func is not None:
        if return_probas:
            return transform_proba_func(Y_probas, **kwargs), Y_probas
        else:
            return transform_proba_func(Y_probas, **kwargs)
    return Y_probas


def cross_val_score(data, nb_folds=5, **kwargs):
    logging.info("- Cross validation : ")
    x_df, y_df, clients_last_choice = data
    kf = KFold(n_splits=nb_folds)
    scores = []

    count = 0
    for train_index, test_index in kf.split(range(x_df.shape[0])):
        count += 1
        logging.info("\n\n\t\t-- Fold : %i / %i\n" % (count, nb_folds))

        X_train, X_val = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        Y_train, Y_val = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]
        clc_val = clients_last_choice[test_index, :]

        estimators = train_all(X_train, Y_train, **kwargs)
        if 'return_probas' in kwargs:
            y_preds, Y_probas = predict_all(estimators, X_val, **kwargs)
        else:
            y_preds = predict_all(estimators, X_val, **kwargs)

        y_val = targets_str_to_indices(Y_val[TARGET_LABELS].values)
        logging.info("- Compute map7 score")
        scores.append(map7_score(y_val, y_preds, clc_val))

    return np.array(scores)

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import sys
sys.path.append("../common")
from utils import map7_score, map7_score0, TARGET_LABELS, targets_str_to_indices


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
    :samples_masks_list: a list, e.g. `[samples_mask_1, samples_mask_2, ...]` with samples_mask_i is a function to
        produce a boolean pd.DataFrame . Used only for training.
    If an empty list is providede, all samples are used for training

    :features_masks_list: a dictionary, e.g. `{fm1_name: features_mask_1, fm2_name: features_mask_2, ...]` with
        `features_mask_i` is a list of feature column names. They can oversect.
        Feature mask can be None to indicate all features.
    :labels_masks_list: a dictionary, e.g.`{lm1_name: labels_mask_1, lm2_name: labels_mask_2, ...}` with `labels_mask_i`
        is a list of labels column names. They can oversect. Labels mask can be None to train over all labels.
    :models_dict: a dictionary of functions to create a model, e.g.
        `{'rf': create_RF, 'nn': create_NN, 'gbt': create_GBT}`

    In `kwargs` it is possible to define :
        :verbose: True/False
        :models_pipelines: (optional) a dictionary, e.g.
            `{model_name: [(samples_mask_code, features_mask_name, labels_mask_name), ...]}`.
            It defines specific connection between a model and samples/features/labels to train on.
            This is useful, when a model, for example, can not train on all types of labels.
            It is possible to specify only one mask name `samples_mask_name` or `features_mask_name` or
            `labels_mask_name` with None, e.g. (None, labels_mask_name). If models_pipelines is defined and a model is
             not added into models_pipelines, it will be used on all combinations of samples mask/features mask/labels
             mask. Parameter `samples_mask_code` can be either an integer index to correspond to a samples mask from
             `samples_masks_list`, either None or 'all' which means to train on all samples.

        :prepare_to_fit_func: (optional), a function to transform X_train, Y_train into acceptable ndarrays.
            Default, `prepare_to_fit` is used


    :return: a list of trained estimators, e.g.
        `[([features_mask_name, labels_mask_name, model_name], estimator_object, accuracy), ...]`
    """
    logging.debug("---------------")
    logging.info("-- Train all --")
    verbose = False if 'verbose' not in kwargs else kwargs['verbose']
    models_pipelines = None if 'models_pipelines' not in kwargs else kwargs['models_pipelines']
    prepare_to_fit_func = prepare_to_fit if 'prepare_to_fit_func' not in kwargs else kwargs['prepare_to_fit_func']

    # Add 'All' samples mask as the last one
    def _all_samples_mask(x, y):
        return x.index.isin(x.index[:])
    
    _samples_masks_list = list(samples_masks_list)
    _samples_masks_list.append(_all_samples_mask)

    estimators = []

    for samples_mask_index, samples_mask in enumerate(_samples_masks_list):
        if isinstance(samples_mask, str) and samples_mask == 'all':
            mask = _all_samples_mask(X_train, Y_train)
        else:
            mask = samples_mask(X_train, Y_train)
        X_train_ = X_train[mask]
        Y_train_ = Y_train[mask]

        for features_mask_name in features_masks_dict:
            features_mask = features_masks_dict[features_mask_name]
            X_train__ = X_train_[features_mask] if features_mask is not None else X_train_
            for labels_mask_name in labels_masks_dict:
                labels_mask = labels_masks_dict[labels_mask_name]
                Y_train__ = Y_train_[labels_mask] if labels_mask is not None else Y_train_
                x_train, y_train = prepare_to_fit_func(X_train__, Y_train__)

                if len(y_train.shape) > 1 and y_train.shape[1] == 1:
                    # avoid DataConversionWarning
                    y_train = y_train.ravel()

                for model_name in models_dict:
                    can_fit = True if samples_mask_index < len(_samples_masks_list)-1 else False
                    
                    if models_pipelines is not None and model_name in models_pipelines:
                        can_fit = False
                        pipelines = models_pipelines[model_name]
                        # pipelines = [(samples_mask_code, feature_mask_name, label_mask_name), ...]
                        for _samples_mask_code, _features_mask_name, _labels_mask_name in pipelines:
                            b0 = _samples_mask_code is None
                            b1 = _features_mask_name is None
                            b2 = _labels_mask_name is None

                            assert not (b0 and b1 and b2), \
                                "Samples mask name and features mask name and labels mask name can not be all None"

                            b0 = False
                            if (_samples_mask_code is not None and _samples_mask_code == 'all' and
                                        samples_mask_index == len(_samples_masks_list)-1) or \
                                    (len(_samples_masks_list) == 1) or \
                                    (_samples_mask_code is None and samples_mask_index < len(_samples_masks_list)-1) or \
                                    (_samples_mask_code is not None and _samples_mask_code == samples_mask_index):
                                b0 = True

                            if _features_mask_name is not None and _features_mask_name == features_mask_name:
                                b1 = True
                            if _labels_mask_name is not None and _labels_mask_name == labels_mask_name:
                                b2 = True
                            can_fit = b0 and b1 and b2
                            if can_fit:
                                break

                    if not can_fit:
                        continue

                    logging.info("-- Process : sample_mask={}/{}, features_mask={}, labels_mask={}"
                                 .format(len(X_train_), len(X_train), features_mask_name, labels_mask_name))
                    logging.debug("--- Train data shapes : {}, {}".format(x_train.shape, y_train.shape))
                    logging.debug("-- Create the model : %s" % model_name)

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


def merge_probas(Y_probas, y_probas, labels_mask, mode='sum', **kwargs):
    if mode == 'max':
        Y_probas.loc[:, labels_mask] = np.maximum(Y_probas.loc[:, labels_mask], y_probas)
    elif mode == 'sum':
        Y_probas.loc[:, labels_mask] = Y_probas.loc[:, labels_mask] + y_probas
    else:
        raise Exception("Existing data merge is not yet implemented")

    return Y_probas


def score_estimators(estimators, X_val, Y_val, features_masks_dict, labels_masks_dict, **kwargs):
    """
    Method to score estimators
    :estimators: a list of object of type ([features_mask_name, labels_mask_name, model_name], estimator_object, fit_accuracy)
    :X_val: a pd.DataFrame of shape `(nb_samples, nb_features)`
    :Y_val: (default `None`) a pd.DataFrame of shape `(nb_samples, nb_labels)`.

    In `kwargs` it is possible to define :
        :prepare_to_test_func: (optional), a function to transform X_val, Y_val(=None) into acceptable ndarrays.
            Default, `prepare_to_test` is used


    :return: list of [features_mask_name, labels_mask_name, model_name, score]
    """
    prepare_to_test_func = prepare_to_test if 'prepare_to_test_func' not in kwargs else kwargs['prepare_to_test_func']

    scores = []
    for estimator in estimators:
        # estimator is ([features_mask_name, labels_mask_name, model_name], estimator_object)
        features_mask_name, labels_mask_name, model_name = estimator[0]
        features_mask = features_masks_dict[features_mask_name]
        labels_mask = labels_masks_dict[labels_mask_name]
        x_val, y_val = prepare_to_test_func(X_val[features_mask], Y_val[labels_mask])
        logging.debug("--- Test data shapes : {}".format(x_val.shape))
        score = estimator[1].score(x_val, y_val)
        logging.info("-- Score : model={}, features_mask={}, labels_mask={} -> {}"
                     .format(model_name, features_mask_name, labels_mask_name, score))
        scores.append([features_mask_name, labels_mask_name, model_name, score])
    return scores
        

def predict_all(estimators, X_val, features_masks_dict, labels_masks_dict, labels, **kwargs):
    """
    Method to compute predictions using `estimators` from a test dataset `X_val`

    :estimators: a list of object of type ([features_mask_name, labels_mask_name, model_name], estimator_object, fit_accuracy)
    :X_val: a pd.DataFrame of shape `(nb_samples, nb_features)`
    :features_masks_dict: a dictionary of features masks (see train_all method)
    :labels_masks_dict: a dictionary of labels masks (see train_all method)
    :labels: a list of all available labels for the output
    
    In `kwargs` it is possible to define :
        :transform_proba_func: a function to transform computed probabilities into a custom form.
        Function signature should be `foo(Y_probas, **kwargs)`

        :verbose: True/False
        :prepare_to_test_func: (optional), a function to transform X_val, Y_val(=None) into acceptable ndarrays.
        Default, `prepare_to_test` is used. For example,
        ```
            def prepare_to_test(X_val, Y_val=None):
                x_val = X_val.values
                x_val = StandardScaler().fit_transform(x_val)
                y_val = Y_val.values if Y_val is not None else None
                return x_val, y_val
        ```
        :probas_to_labels_probas_func: (optional), a function to transform predicted class probabilities to output
        labels probabilities. Default, function is identity. Example of such function :
        ```
            # y_probas.shape = (n_samples, n_classes)
            # class_indices = np.array([c1, c2, ...]), len(class_indices) = n_classes,
            #   class_indices[np.argmax(y_probas, axis=1)], max probability classes
            # output_labels
            def probas_to_labels_probas(y_probas, class_indices, output_labels):
                pass

        ```

    :return:
        if `transform_proba_func` is not defined, predicted label probabilites `Y_probas` (pd.DataFrame) are returned.
        Thus, output is an ndarray of shape (nb_samples, len(labels)).

        if `transform_proba_func` is defined, then output is an ndarray of shape `(nb_samples, ...)`, the output of `transform_proba_func`.

    """
    logging.debug("-----------------")
    logging.info("-- Predict all --")
    verbose = False if 'verbose' not in kwargs else kwargs['verbose']
    prepare_to_test_func = prepare_to_test if 'prepare_to_test_func' not in kwargs else kwargs['prepare_to_test_func']

    return_probas = False if 'return_probas' not in kwargs else kwargs['return_probas']
    transform_proba_func = None if 'transform_proba_func' not in kwargs else kwargs['transform_proba_func']
    probas_to_labels_probas_func = None if 'probas_to_labels_probas_func' not in kwargs else kwargs['probas_to_labels_probas_func']

    Y_probas = pd.DataFrame(index=X_val.index, columns=labels)
    Y_probas = Y_probas.fillna(0.0)
    for estimator in estimators:
        # estimator is ([features_mask_name, labels_mask_name, model_name], estimator_object)
        features_mask_name, labels_mask_name, model_name = estimator[0]
        features_mask = features_masks_dict[features_mask_name]
        labels_mask = labels_masks_dict[labels_mask_name]
        logging.debug("-- Process : model={}, features_mask={}, labels_mask={}".format(model_name, features_mask_name,
                                                                                      labels_mask_name))
        x_val, _ = prepare_to_test_func(X_val[features_mask])
        logging.debug("--- Test data shapes : {}".format(x_val.shape))

        if hasattr(estimator[1], 'n_outputs_'):
            if estimator[1].n_outputs_ > 1:
                # Pass here when y_train is of shape [n_samples, ???]
                y_probas = estimator[1].predict(x_val)
            else:
                # Pass here when y_train is of shape [n_samples, n_classes] as dummies
                # -> y_probas looks like [[p1, p2, ... pn], ...]
                y_probas = estimator[1].predict_proba(x_val)
        else:
            y_probas = estimator[1].predict(x_val)

        logging.debug("--- Predicted data shape : {}".format(y_probas.shape))

        if probas_to_labels_probas_func is not None:
            y_probas = probas_to_labels_probas_func(y_probas, estimator[1].classes_, labels_mask)

        if y_probas.dtype == np.int:
            y_probas = y_probas.astype(np.float)
        if len(y_probas.shape) == 1:
            y_probas = y_probas.reshape((y_probas.shape[0], 1))

        # multiply by accuracy :
        y_probas *= estimator[2]
        Y_probas = merge_probas(Y_probas, y_probas, labels_mask, **kwargs)
    
    # Y_probas_max = Y_probas.max(axis=1)
    # mask = Y_probas_max > 0
    # Y_probas.loc[mask, :] = Y_probas[mask].div(Y_probas_max[mask], axis=0)
    
    if transform_proba_func is not None:
        y_preds = transform_proba_func(Y_probas, **kwargs)
        if return_probas:
            return y_preds, Y_probas
        else:
            return y_preds
      
    return Y_probas


def cross_val_score0(data, nb_folds=5, **kwargs):
    logging.info("- Cross validation : ")
    x_df, y_df = data
    kf = KFold(n_splits=nb_folds)
    scores = []

    count = 0
    for train_index, test_index in kf.split(range(x_df.shape[0])):
        count += 1
        logging.info("\n\n\t\t-- Fold : %i / %i\n" % (count, nb_folds))

        X_train, X_val = x_df.loc[x_df.index[train_index], :], x_df.loc[x_df.index[test_index], :]
        Y_train, Y_val = y_df.loc[y_df.index[train_index], :], y_df.loc[y_df.index[test_index], :]

        estimators = train_all(X_train, Y_train, **kwargs)
        if 'return_probas' in kwargs:
            y_preds, Y_probas = predict_all(estimators, X_val, **kwargs)
        else:
            y_preds = predict_all(estimators, X_val, **kwargs)

        y_val = targets_str_to_indices(Y_val[TARGET_LABELS].values)
        logging.info("- Compute map7 score")
        scores.append(map7_score0(y_val, y_preds))

    return np.array(scores)


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
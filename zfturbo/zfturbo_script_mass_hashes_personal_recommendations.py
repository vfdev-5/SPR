__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

"""
Fork of ZFTurbo 'Mass hashes' code : https://www.kaggle.com/zfturbo/santander-product-recommendation/mass-hashes/code

Added personal recommendations based on previous user's choices

"""

from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
import random
import math
random.seed(2016)

import numpy as np
import scipy

# Project
from common import parse_line, get_target_labels, get_choices, get_user, \
    clean_data, process_data, to_yearmonth, apk, get_real_values

#####################################################################

PERSONAL_RECOMMENDATIONS_WEIGHT = 0.7
COMMON_RECOMMENDATIONS_WEIGHT = 1.0 - PERSONAL_RECOMMENDATIONS_WEIGHT

####################################################################################


def compute_product_probas(user, personal_recommendations, profiles, common_recommendations, personal_recommendations_weight):

    common_predictions = compute_predictions_from_common(common_recommendations, profiles)
    personal_predictions = compute_predictions_from_personal(user, personal_recommendations)

    if common_predictions is not None and personal_predictions is not None:
        return (1.0 - personal_recommendations_weight) * common_predictions + personal_recommendations_weight * personal_predictions
    elif personal_predictions is not None:
        return personal_predictions
    elif common_predictions is not None:
        return common_predictions
    else:
        return []


def compute_predictions_from_personal(user, personal_recommendations):
    personal_predictions = None
    if user in personal_recommendations:
        personal_predictions = personal_recommendations[user]['recommendations']

    #print "\n\n Personal predictions: ", personal_predictions
    return personal_predictions


def compute_predictions_from_common(common_recommendations, profiles):
    """
    Compute suggestion using profiles
    
    if len(profile1) > len(profile2) -> profile1 is more important than profile2
    
    :return: list of target indices sorted by descreasing importance with proba
    
    """
    target_weights = None
    total_length = 0.0
    # compute a total length to of participating profiles to define profile weight
    for profile in profiles:
        if profile in common_recommendations:
            total_length += len(profile)

    if total_length > 0:
        target_weights = np.zeros(24)

    for profile in profiles:
        if profile in common_recommendations:
            profile_weight = len(profile) * 1.0 / total_length
            for target in common_recommendations[profile]:
                if isinstance(target, int):
                    target_weights[target] += common_recommendations[profile][target] * profile_weight

    #print "\n\n Common predictions : ", target_weights
    return target_weights


def compute_predictions(row, get_profiles_func,
                        personal_recommendations,
                        common_recommendations,
                        product_stats,
                        personal_recommendations_weight):
    predicted = []
    user = get_user(row)
    profiles = get_profiles_func(row)

    last_choice = None
    if user in personal_recommendations:
        last_choice = personal_recommendations[user]['last_choice']

    # 
    probas = compute_product_probas(user,
                                    personal_recommendations, profiles,
                                    common_recommendations,
                                    personal_recommendations_weight)

    # Remove the products from the last choice : 
    if last_choice is not None and len(probas) > 0:
        mask = np.abs(last_choice - 1)
        probas *= mask

    suggestions = np.argsort(probas)[::-1]

    # print "\n Suggestions after last choice : ", suggestions

    # add 7 to predicted:
    if len(predicted) < 7:
        l = min(7, len(suggestions))
        predicted = suggestions[:l].tolist()

    #print "\n- PREDICTED : ", predicted
    # add suggestions from product_stats:
    if len(predicted) < 7:
        for product in product_stats:
            # If user is not new
            if last_choice is not None and last_choice[product[0]] == 1:
                continue
            if product[0] not in predicted:
                predicted.append(product[0])
                if len(predicted) == 7:
                    break

    #print "FINAL PREDICTED : ", predicted 
    return predicted


def get_profiles(row):

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento) = row[:24]

    profiles = [
        ##(0, pais_residencia, nomprov, sexo, age, renta, segmento, ind_empleado),
        #(1, pais_residencia, nomprov, renta, ind_empleado),
        #(2, sexo, age, renta, segmento),
        ##(10, antiguedad, indrel_1mes, indrel, indresi, canal_entrada, ind_actividad_cliente, ind_nuevo),
        #(11, antiguedad, indrel_1mes, indrel, indresi),
        #(12, canal_entrada, ind_actividad_cliente, ind_nuevo),
        #(100, sexo, age, renta, antiguedad, indrel, ind_actividad_cliente),

        ## ZFTurbo
        (1, pais_residencia, sexo, age, ind_nuevo, segmento, ind_empleado, ind_actividad_cliente, indresi),
        (2, pais_residencia, sexo, age, segmento, nomprov),
        (3, pais_residencia, sexo, age, segmento, ncodpers),
        (4, pais_residencia, sexo, age, segmento, antiguedad),
        (5, pais_residencia, sexo, age, segmento, ind_nuevo),
        (6, pais_residencia, sexo, age, segmento, ind_actividad_cliente),
        (7, pais_residencia, sexo, age, segmento, canal_entrada),
        (8, pais_residencia, sexo, age, segmento, ind_nuevo,canal_entrada),
        (9, pais_residencia, sexo, age, segmento, ind_empleado),
        (10, pais_residencia, sexo, renta, age, segmento),
        (11, sexo, age, segmento)
    ]

    return profiles


def update_common_recommendations(common_recommendations, row, get_profiles_func):
    profiles = get_profiles_func(row)
    choices = get_choices(row)
    for profile in profiles:
        # Update common recommendations
        for i, t in enumerate(choices):
            if t > 0:
                common_recommendations[profile][i] += 1
        common_recommendations[profile]['total'] += 1


def update_personal_recommendations(personal_recommendations, row):
    user = get_user(row)
    choices = np.array(get_choices(row))
    # Init/Update personal recommendations
    if user not in personal_recommendations:
        personal_recommendations[user]['recommendations'] = choices
    else:
        updates = choices.copy()
        updates[updates == 0] = -1
        current_values = personal_recommendations[user]['recommendations']
        updates += current_values
        updates[updates < 0] = 0
        personal_recommendations[user]['recommendations'] = updates
    personal_recommendations[user]['last_choice'] = choices


def update_product_stats(product_stats, row):
    choices = get_choices(row)
    for i, t in enumerate(choices):
        # Update product statistics
        if t > 0:
            product_stats[i] += 1


def read_data(reader, yearmonth_begin, nb_months, get_profiles_func, return_raw_data=False,
              personal_recommendations=None, common_recommendations=None, product_stats=None):
    """
    :param reader:
    :param yearmonth_begin: e.g. 201501
    :param nb_months: e.g. 5 -> 201501, 201502, 201503, 201504, 201505
    :return:
    (personal_recommendations,
     common_recommendations,
     product_stats)

     or if return_raw_data == True:

     (personal_recommendations,
     common_recommendations,
     product_stats,
     raw_data)

     raw_data : list of read and parsed lines

    """

    def _to_yearmonth_str(yearmonth):
        year = int(math.floor(yearmonth * 0.01))
        month = yearmonth - year * 100
        return "%s-%s" % (str(year), str(month).zfill(2))

    year = int(math.floor(yearmonth_begin * 0.01)) * 100
    dates = [_to_yearmonth_str(yearmonth_begin), ]
    current_yearmonth = yearmonth_begin
    for i in range(1, nb_months):
        current_yearmonth += 1
        if current_yearmonth - year > 12:
            year += 100
            current_yearmonth = year + 1
        dates.append(_to_yearmonth_str(current_yearmonth))

    logging.info("- READ DATA : months to read {}".format(dates))

    raw_data = []

    if common_recommendations is None:
        common_recommendations = defaultdict(lambda: defaultdict(float))
    if personal_recommendations is None:
        personal_recommendations = defaultdict(lambda: defaultdict(float))
    if product_stats is None:
        product_stats = defaultdict(int)

    # Loop on lines in the file reader:
    removed_rows = 0
    total = 0
    while True:
        line = reader.readline()[:-1]
        total += 1

        if line == '':
            break
        row = parse_line(line)

        # data : ['2015-01-28',
        #  user id -> '1375586',
        #  profile -> 'N', 'ES', 'H', '35', '2015-01-12', '0', '6', '1', '', '1.0', 'A', 'S', 'N', '', 'KHL', 'N', '1', '29', 'MALAGA', '1', '87218.1', '02 - PARTICULARES',
        #  choices -> '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

        yearmonth_str = row[0][:7]
        if yearmonth_str not in dates:
            yearmonth = to_yearmonth(yearmonth_str)
            if yearmonth - yearmonth_begin < 0:
                total = 0
                continue
            else:
                break

        ### -------- !!! --------- ###
        #if total > 1000:
        #   continue
        ### -------- !!! -------- ###

        row = clean_data(row)

        if len(row) == 0:
            removed_rows += 1
            continue

        if return_raw_data:
            raw_data.append(row)

        processed_row = process_data(row)

        update_common_recommendations(common_recommendations, processed_row, get_profiles_func)
        update_personal_recommendations(personal_recommendations, processed_row)
        update_product_stats(product_stats, row)

        if total % 100000 == 0:
            logging.info('-- Processed {} lines . Current month : {}'.format(total, yearmonth_str))

    logging.debug("-- Removed rows : %s" % removed_rows)

    if return_raw_data:
        return personal_recommendations, common_recommendations, product_stats, raw_data
    return personal_recommendations, common_recommendations, product_stats


def personal_recommendations_to_proba(personal_recommendations, nb_months):
    """
        Transform personal recommendations to probabilities :
        proba = values / (2.0 * (nb_months)) + 0.5
            or 
        proba = values / nb_months
    """
    for key in personal_recommendations:
        values = personal_recommendations[key]['recommendations']
        proba = np.array(values) / (2.0 * nb_months) + 0.5
        #proba = np.array(values) / (1.0 * nb_months)
        personal_recommendations[key]['recommendations'] = proba


def common_recommendations_to_proba(common_recommendations):
    for profile in common_recommendations:
        total_count = common_recommendations[profile]['total']
        for choice_index in common_recommendations[profile]:
            if isinstance(choice_index, int):
                common_recommendations[profile][choice_index] /= total_count


def write_submission(writer, reader, target_labels,
                     get_profiles_func,
                     personal_recommendations,
                     common_recommendations,
                     product_stats,
                     personal_recommendations_weight):

    total = 0
    writer.write("ncodpers,added_products\n")

    while True:
        line = reader.readline()[:-1]
        total += 1

        if line == '':
            break

        row = parse_line(line)

        user = get_user(row)
        writer.write(user + ',')

        predicted = compute_predictions(row,
                                        get_profiles_func,
                                        personal_recommendations,
                                        common_recommendations,
                                        product_stats, personal_recommendations_weight)
        for p in predicted:
            writer.write(target_labels[p] + ' ')

        if total % 1000000 == 0:
            logging.info('Read {} lines'.format(total))

        writer.write("\n")


def predict_score(validation_data, get_profiles_func,
                  personal_recommendations,
                  common_recommendations,
                  product_stats,
                  personal_recommendations_weight):
    logging.debug("-- predict_score : personal_recommendations_weight=%s" % personal_recommendations_weight)
    map7 = 0.0
    for row in validation_data:
        predicted = compute_predictions(row, get_profiles_func,
                                        personal_recommendations,
                                        common_recommendations,
                                        product_stats,
                                        personal_recommendations_weight)

        real = get_real_values(row, personal_recommendations)

        score = apk(real, predicted)
        map7 += score

    if len(validation_data) > 0:
        map7 /= len(validation_data)

    logging.debug("--- predict_score : map7=%s" % map7)
    return map7


def run_solution(train_filename, test_filename):

    logging.info('--- Run solution ---')
    reader = open(train_filename, "r")
    target_labels = get_target_labels(reader.readline())

    nb_months_validation = 4

    (personal_recommendations_validation,
     common_recommendations_validation,
     product_stats_validation) = read_data(reader, 201601, nb_months_validation, get_profiles)

    logging.debug("-- common_recommendations_validation : %s " % len(common_recommendations_validation))
    logging.debug("-- personal_recommendations_validation : %s " % len(personal_recommendations_validation))
    logging.debug("-- product_stats_validation : %s " % len(product_stats_validation))

    personal_recommendations = deepcopy(personal_recommendations_validation)
    common_recommendations = deepcopy(common_recommendations_validation)
    product_stats = deepcopy(product_stats_validation)

    (personal_recommendations,
     common_recommendations,
     product_stats,
     validation_data) = read_data(reader, 201605, 1, get_profiles,
                                  return_raw_data=True,
                                  personal_recommendations=personal_recommendations,
                                  common_recommendations=common_recommendations,
                                  product_stats=product_stats)

    logging.debug("-- common_recommendations : %s " % len(common_recommendations))
    logging.debug("-- personal_recommendations : %s " % len(personal_recommendations))
    logging.debug("-- product_stats : %s " % len(product_stats))

    reader.close()

    personal_recommendations_to_proba(personal_recommendations, nb_months_validation)
    personal_recommendations_to_proba(personal_recommendations_validation, nb_months_validation+1)

    common_recommendations_to_proba(common_recommendations)
    common_recommendations_to_proba(common_recommendations_validation)

    # Sort product stats:
    product_stats_validation = sorted(product_stats_validation.items(), key=itemgetter(1), reverse=True)
    product_stats = sorted(product_stats.items(), key=itemgetter(1), reverse=True)

    logging.info("- Validation")
    map7 = predict_score(validation_data,
                         get_profiles,
                         personal_recommendations_validation,
                         common_recommendations_validation,
                         product_stats_validation,
                         PERSONAL_RECOMMENDATIONS_WEIGHT)

    logging.info("Predicted score: {}".format(map7))

    if test_filename is None:
        return

    logging.info('- Generate submission')
    submission_file = '../results/submission_' + \
                      str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + \
                      '.csv'
    writer = open(submission_file, "w")
    reader = open(test_filename, "r")

    # skip header:
    reader.readline()

    write_submission(writer, reader,
                     target_labels,
                     get_profiles,
                     personal_recommendations,
                     common_recommendations,
                     product_stats,
                     PERSONAL_RECOMMENDATIONS_WEIGHT)

    writer.close()
    reader.close()

#####################################################################

if __name__ == "__main__":

    # train_filename = "../data/train_ver2.csv"
    train_filename = "../data/train_ver2_201601-201605.csv"
    test_filename = "../data/test_ver2.csv"
    # test_filename = None
    run_solution(train_filename, test_filename)

#####################################################################

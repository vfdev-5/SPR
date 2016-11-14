__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

"""
Fork of ZFTurbo 'Mass hashes' code : https://www.kaggle.com/zfturbo/santander-product-recommendation/mass-hashes/code

Added personal recommendations based on previous user's choices

"""

from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

from collections import defaultdict
from operator import itemgetter
import random
# import itertools
import heapq
import math
random.seed(2016)

import numpy as np

# Project
from common import parse_line, get_target_labels, get_choices, get_user, \
    clean_data, get_age_group_index, get_income_group_index

#####################################################################

COMMON_RECOMMENDATIONS_WEIGHT = 0.4
PERSONAL_RECOMMENDATIONS_WEIGHT = 1.0 - COMMON_RECOMMENDATIONS_WEIGHT

#####################################################################


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


def get_next_best_prediction(best, hashes, predicted, cst):

    score = [0] * 24

    for h in hashes:
        if h in best:
            for i in range(len(best[h])):
                sc = 24-i + len(h)
                index = best[h][i][0]
                if cst is not None:
                    if cst[index] == '1':
                        continue
                if index not in predicted:
                    score[index] += sc

    final = []
    pred = heapq.nlargest(7, range(len(score)), score.__getitem__)
    for i in range(7):
        if score[pred[i]] > 0:
            final.append(pred[i])

    return final


####################################################################################

# def compute_predictions(user, profile, threshold=0.6, verbose=False):
#     common_predictions = None
#     if profile in common_recommendations:
#         common_predictions = []
#         for t in target_labels:
#             common_predictions.append(common_recommendations[profile][t])
#     if verbose: print "Common predictions: ", common_predictions
#
#     personal_predictions = None
#     if user_id in personal_recommendations:
#         personal_predictions = personal_recommendations[user_id]
#     if verbose: print "Personal predictions: ", personal_predictions
#
#     if common_predictions is not None and personal_predictions is not None:
#         predictions = np.array(common_predictions) * common_rc_weight + np.array(
#             personal_predictions) * personal_rc_weight
#     elif common_predictions is not None:
#         predictions = np.array(common_predictions)
#     elif personal_predictions is not None:
#         predictions = np.array(personal_predictions)
#     else:
#         raise Exception("Failed to compute predictions")
#
#     if verbose: print "Total predictions: ", predictions
#
#     predictions[predictions >= threshold] = 1
#     predictions[predictions < threshold] = 0
#     return predictions.astype(np.int)


# def compute_recommendations(user, personal_recommendations, profiles, common_recommendations):

def compute_predictions_from_common(common_recommendations, profiles, predicted, customer_data):
    # score = [0] * 24
    # for profile in profiles:
    #     if profile in common_recommendations:
    #         for i in range(len(common_recommendations[profile])):
    #             sc = 24-i + len(profile)
    #             index = common_recommendations[profile][i][0]
    #             if customer_data is not None:
    #                 if customer_data[index] == 1:
    #                     continue
    #             if index not in predicted:
    #                 score[index] += sc
    #
    # final = []
    # pred = heapq.nlargest(7, range(len(score)), score.__getitem__)
    # for i in range(7):
    #     if score[pred[i]] > 0:
    #         final.append(pred[i])
    #
    # return final
    return []


def compute_predictions(row, get_profiles_func,
                        personal_recommendations,
                        common_recommendations,
                        product_stats):
    predicted = []
    user = get_user(row)
    profiles = get_profiles_func(row)
    #

    customer_data = None
    if user in personal_recommendations:
        customer_data = personal_recommendations[user]['last_choice']

    predicted = compute_predictions_from_common(common_recommendations, profiles, predicted, customer_data)

    # add suggestions from product_stats:
    if len(predicted) < 7:
        for product in product_stats:
            # If user is not new
            if user in personal_recommendations:
                last_choice = personal_recommendations[user]['last_choice']
                if last_choice[product[0]] == 1:
                    continue

            if product[0] not in predicted:
                predicted.append(product[0])
                if len(predicted) == 7:
                    break
    return predicted


def get_real_values(row, personal_recommendations):
    real = []
    user = get_user(row)
    choices = get_choices(row)

    for i, c in enumerate(choices):
        if c == 1:
            if user in personal_recommendations:
                if personal_recommendations[user]['last_choice'][i] == 0:
                    real.append(i)
            else:
                real.append(i)
    return real


def _to_yearmonth(yearmonth_str):
    yearmonth = int(yearmonth_str[:7].replace('-', ''))
    return yearmonth


def process_data(row):
    """
    Method to process data rows (feature engineering)

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento)  # 21

    renta -> income group index
    age -> age group index
    antiguedad -> int(fecha_dato - fecha_alta)

    """
    row[22] = get_income_group_index(row[22])
    row[5] = get_age_group_index(row[5])
    res = _to_yearmonth(row[0])*0.01 - _to_yearmonth(row[6])*0.01
    row[8] = int(math.floor(res)) * 12 + int(math.ceil((res - int(res)) * 100))
    return row


def get_profiles(row):

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age_group,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento) = row[:24]

    profiles = [
        (0, pais_residencia, nomprov, sexo, age_group, renta, segmento),
        (1, pais_residencia, age_group, renta, segmento),
        (2, sexo, age_group, renta, segmento),

        (10, antiguedad, indrel_1mes, indrel, indresi),
        (11, canal_entrada, ind_actividad_cliente, ind_nuevo),

        (100, sexo, age_group, renta, antiguedad, indrel, ind_actividad_cliente),
    ]

    return profiles


def update_common_recommendations(common_recommendations, row, get_profiles_func):
    profiles = get_profiles_func(row)
    choices = get_choices(row)
    for i, t in enumerate(choices):
        # Update common recommendations
        for profile in profiles:
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
            yearmonth = _to_yearmonth(yearmonth_str)
            if yearmonth - yearmonth_begin < 0:
                total = 0
                continue
            else:
                break

        # if total > 1000:
        #     continue

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
            logging.info('-- Processed {} lines . Current month is : {}'.format(total, yearmonth_str))

    logging.debug("-- Removed rows : %s" % removed_rows)

    if return_raw_data:
        return personal_recommendations, common_recommendations, product_stats, raw_data
    return personal_recommendations, common_recommendations, product_stats


def personal_recommendations_to_proba(personal_recommendations, nb_months):
    """
        Transform personal recommendations to probabilities :
        proba = values / (2.0 * (nb_months)) + 0.5
    """
    for key in personal_recommendations:
        values = personal_recommendations[key]['recommendations']
        proba = np.array(values) / (2.0 * nb_months) + 0.5
        personal_recommendations[key]['recommendations'] = proba


def write_submission(writer, reader, get_profiles_func, personal_recommendations, common_recommendations, product_stats):

    target_labels = get_target_labels(reader.readline())
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
                                        product_stats)

        for p in predicted:
            writer.write(target_labels[p] + ' ')

        if total % 1000000 == 0:
            logging.info('Read {} lines'.format(total))

        writer.write("\n")


def run_solution():

    logging.info('--- Run solution ---')
    reader = open("../data/train_ver2.csv", "r")
    target_labels = get_target_labels(reader.readline())

    nb_months_validation = 2

    (personal_recommendations_validation,
     common_recommendations_validation,
     product_stats_validation) = read_data(reader, 201501, nb_months_validation, get_profiles)

    personal_recommendations = personal_recommendations_validation.copy()
    common_recommendations = common_recommendations_validation.copy()
    product_stats = product_stats_validation.copy()

    (personal_recommendations,
     common_recommendations,
     product_stats,
     validation_data) = read_data(reader, 201503, 1, get_profiles,
                                  return_raw_data=True,
                                  personal_recommendations=personal_recommendations,
                                  common_recommendations=common_recommendations,
                                  product_stats=product_stats)

    reader.close()

    personal_recommendations_to_proba(personal_recommendations, nb_months_validation)
    personal_recommendations_to_proba(personal_recommendations_validation, nb_months_validation+1)

    # Sort product stats:
    product_stats_validation = sorted(product_stats_validation.items(), key=itemgetter(1), reverse=True)
    product_stats = sorted(product_stats.items(), key=itemgetter(1), reverse=True)

    # Normal
    #best, overallbest = sort_main_arrays(best, overallbest)
    # Valid
    #best_valid, overallbest_valid = sort_main_arrays(best_valid, overallbest_valid)

    map7 = 0.0
    logging.info("- Validation")
    for row in validation_data:
        predicted = compute_predictions(row, get_profiles,
                                        personal_recommendations_validation,
                                        common_recommendations_validation,
                                        product_stats_validation)

        real = get_real_values(row, personal_recommendations_validation)

        score = apk(real, predicted)
        map7 += score

    if len(validation_data) > 0:
        map7 /= len(validation_data)

    logging.info("Predicted score: {}".format(map7))

    return

    logging.info('- Generate submission')
    submission_file = 'submission_' + \
                      str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + \
                      '.csv'
    writer = open(submission_file, "w")
    reader = open("../data/test_ver2.csv", "r")

    write_submission(writer, reader, get_profiles, personal_recommendations, common_recommendations, product_stats)

    writer.close()
    reader.close()

#####################################################################

if __name__ == "__main__":
    run_solution()

#####################################################################
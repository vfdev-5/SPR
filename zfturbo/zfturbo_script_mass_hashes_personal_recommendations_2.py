__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

"""
Fork of ZFTurbo 'Mass hashes' code : https://www.kaggle.com/zfturbo/santander-product-recommendation/mass-hashes/code

Added personal recommendations based on previous user's choices

"""

from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

from operator import itemgetter

from copy import deepcopy
import numpy as np

# Project
from zfturbo_script_mass_hashes_personal_recommendations import read_data, get_profiles, personal_recommendations_to_proba, common_recommendations_to_proba, get_target_labels, predict_score


# ### Define train/test files

# In[3]:

train_filename = "../data/train_ver2_201601-201605.csv"
# test_filename = "../data/test_ver2.csv"
test_filename = None


# ### Compute recommendations from training data

# In[4]:

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

common_recommendations_to_proba(common_recommendations)
common_recommendations_to_proba(common_recommendations_validation)

# Sort product stats:
product_stats_validation = sorted(product_stats_validation.items(), key=itemgetter(1), reverse=True)
product_stats = sorted(product_stats.items(), key=itemgetter(1), reverse=True)


# In[5]:

#personal_recommendations_to_proba(personal_recommendations, nb_months_validation)
personal_recommendations_to_proba(personal_recommendations_validation, nb_months_validation+1)


# In[6]:

print('Hashes num: ', len(common_recommendations))
print('Hashes valid num: ', len(common_recommendations_validation))
print('Valid part: ', len(validation_data))


# ### Declare methods :

PERSONAL_RECOMMENDATIONS_WEIGHT = 0.0
COMMON_RECOMMENDATIONS_WEIGHT = 1.0 - PERSONAL_RECOMMENDATIONS_WEIGHT

ZFTURBO_COMMON_WEIGHT = 1.0
MINE_COMMON_WEIGHT = 1.0 - ZFTURBO_COMMON_WEIGHT


_VERBOSE = False
__VERBOSE = False


def _compute_product_probas(user, personal_recommendations, profiles, common_recommendations, personal_recommendations_weight):

    common_predictions = _compute_predictions_from_common(common_recommendations, profiles)
    personal_predictions = _compute_predictions_from_personal(user, personal_recommendations)

    if common_predictions is not None and personal_predictions is not None:
        return (1.0 - personal_recommendations_weight) * common_predictions + personal_recommendations_weight * personal_predictions
    elif personal_predictions is not None:
        return personal_predictions
    elif common_predictions is not None:
        return common_predictions
    else:
        return []



def _compute_predictions_from_personal(user, personal_recommendations):
    personal_predictions = None
    if user in personal_recommendations:
        personal_predictions = personal_recommendations[user]['recommendations']

    if _VERBOSE:
        print "\n\n Personal predictions: ", personal_predictions
    return personal_predictions


def _compute_predictions_from_common(common_recommendations, profiles):
    """
    Compute suggestion using profiles
    
    if len(profile1) > len(profile2) -> profile1 is more important than profile2
    
    :return: list of target indices sorted by descreasing importance with proba
    
    """
    target_weights = None
    total_length = 0.0
    total_count = 0
    max_length = 0
    # compute a total length to of participating profiles to define profile weight
    for profile in profiles:
        if profile in common_recommendations:
            total_length += len(profile)
            max_length = max(len(profile), max_length)
            total_count += 1

    if total_length > 0:
        target_weights = np.zeros(24)

    for profile in profiles:
        if profile in common_recommendations:
            profile_weight = len(profile) * 1.0 / total_length
            # _common_recommendations[profile].items() -> [(target, proba)]
            target_probas = sorted(common_recommendations[profile].items(), key=itemgetter(1), reverse=True)
            #target_total_score = (25.0 + len(profile)) * total_count
            target_total_score = (25.0 + max_length) * total_count
            for i, target_proba in enumerate(target_probas):
                target_score = 25.0 - i + len(profile)  # 24 + 'total'
                if __VERBOSE:
                    print "-- i, target_score", i, target_score
                target = target_proba[0]
                proba = target_proba[1]
                if isinstance(target, int):
                    p1 = proba * profile_weight * MINE_COMMON_WEIGHT
                    p2 = target_score * 1.0 / target_total_score * ZFTURBO_COMMON_WEIGHT
                    if __VERBOSE:
                        print "--- i, target, p1, p2, p1 + p2 : ", i, target, p1, p2, p1 + p2
                    target_weights[target] += p1 + p2
            if _VERBOSE:
                print "--> scored labels : {}".format(np.argsort(target_weights)[::-1])
                print "---> target_weights : {}".format(target_weights)

    if _VERBOSE:
        print "\n\n Common predictions : ", target_weights
    return target_weights



def _compute_predictions(row, get_profiles_func,
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

    if _VERBOSE:
        print "Last choice : ", last_choice
    #
    probas = _compute_product_probas(user,
                                    personal_recommendations, profiles,
                                    common_recommendations,
                                    personal_recommendations_weight)

    # Remove the products from the last choice : 
    if last_choice is not None and len(probas) > 0:
        mask = np.abs(last_choice - 1)
        probas *= mask

    if _VERBOSE:
        print "\n Probas (-last_choice) : ", probas

    if len(probas) > 0:
        #suggestions = np.argsort(probas)[::-1]  # equal elements do not have the correct ordering, dunno why
        # probas -> int( probas  * 1000 ) before ordering
        probas_int = (probas * 1000).astype(np.int)
        suggestions = sorted(range(len(probas_int)), key=probas_int.__getitem__, reverse=True)
    else:
        suggestions = []

    if _VERBOSE:
        print "\n Suggestions  (-last_choice) - sorted : ", suggestions

    # add 7 to predicted:
    if len(predicted) < 7:
        l = min(7, len(suggestions))
        #predicted = suggestions[:l].tolist()
        predicted = suggestions[:l]

    if _VERBOSE:
        print "\n- PREDICTED : ", predicted
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

    
    if _VERBOSE:
        print "FINAL PREDICTED : ", predicted 
    return predicted


# In[13]:

from common import get_user, apk, get_real_values, get_choices
import heapq

def zfturbo_compute_predictions(row, get_profiles_func,
                        best,
                        personal_recommendations,
                        product_stats):
    predicted = []
    user = get_user(row)
    profiles = get_profiles_func(row)

    last_choice = None
    if user in personal_recommendations:
        last_choice = personal_recommendations[user]['last_choice']

    if _VERBOSE:
        print "Last choice : ", last_choice
        
    def _get_next_best_prediction(best, profiles, predicted, last_choice):
        score = [0] * 24
        for h in profiles:
            if h in best:
                if __VERBOSE:
                    print "-- profile : ", h
                for i in range(len(best[h])):
                    sc = 24 - i + len(h)
                    index = best[h][i][0]
                    if last_choice is not None:
                        if last_choice[index] == 1:
                            continue
                    if index not in predicted:
                        if __VERBOSE:
                            print "-- i, index, sc", i, index, sc
                        score[index] += sc
                if _VERBOSE:
                    print "--> scored labels: ", np.argsort(score)[::-1] #, np.sort(score)[::-1][:7]
                    print "---> target_weights : {}".format(score)
        
        if _VERBOSE:
            print "\n -- score : ", score, np.argsort(score)[::-1]
        
        final = []
        pred = heapq.nlargest(7, range(len(score)), score.__getitem__)
        if _VERBOSE:
            print "\n -- pred : ", pred
        for i in range(7):
            if score[pred[i]] > 0:
                final.append(pred[i])
                
        if _VERBOSE:
            print "\n -- final : ", final
        return final

    predicted = _get_next_best_prediction(best, profiles, predicted, last_choice)

    if _VERBOSE:
        print "\n- PREDICTED : ", predicted
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

    
    if _VERBOSE:
        print "FINAL PREDICTED : ", predicted
    return predicted


# In[14]:

def zfturbo_predict_score(validation_data, get_profiles_func,
                  common_recommendations,
                  personal_recommendations,
                  product_stats):
    
    logging.debug("-- zfturbo_predict_score")
    map7 = 0.0    
    count = 25
    for i, row in enumerate(validation_data):
        predicted = zfturbo_compute_predictions(row, get_profiles_func,
                                        common_recommendations,
                                        personal_recommendations,
                                        product_stats)
        real = get_real_values(row, personal_recommendations)
        score = apk(real, predicted)
        if count > 0:
            print "-- i : ", i, row[1], " score : ", score, " | predicted : ", predicted, ", real : ", real
        map7 += score
    
        count-=1
        if count == 0:
            break
        
    if len(validation_data) > 0:
        map7 /= len(validation_data)

    logging.debug("--- predict_score : map7=%s" % map7)
    return map7


# ## Prepare data for comparision

# In[15]:

import operator
def sort_common_recommendations(common_recommendations):
    out = dict()
    for b in common_recommendations:
        arr = common_recommendations[b]
        srtd = sorted(arr.items(), key=operator.itemgetter(1), reverse=True)
        # remove 'total'
        out[b] = [item for item in srtd if item[0] != 'total']
    return out
best_validation = sort_common_recommendations(common_recommendations_validation)


# ## Compare prediction methods on all data

# In[20]:

from zfturbo_script_mass_hashes_personal_recommendations import parse_line, process_data


# In[24]:

ZFTURBO_COMMON_WEIGHT = 1.0
MINE_COMMON_WEIGHT = 1.0 - ZFTURBO_COMMON_WEIGHT


# In[72]:

_VERBOSE=False
__VERBOSE=False
# def _predict_score(validation_data, get_profiles_func,
#                   personal_recommendations,
#                   common_recommendations,
#                   product_stats,
#                   personal_recommendations_weight):

get_profiles_func = get_profiles
_personal_recommendations = personal_recommendations_validation
_common_recommendations = common_recommendations_validation
_product_stats = product_stats_validation
personal_recommendations_weight = 0.0

logging.debug("-- predict_score : personal_recommendations_weight=%s" % personal_recommendations_weight)
map7_1 = 0.0   
map7_2 = 0.0   
count = 10000

for i, row in enumerate(validation_data):

    row = process_data(row)
    predicted1 = _compute_predictions(row, get_profiles_func,
                                    _personal_recommendations,
                                    _common_recommendations,
                                    _product_stats,
                                    personal_recommendations_weight)

    predicted2 = zfturbo_compute_predictions(row, get_profiles_func,
                                        best_validation,
                                        _personal_recommendations,
                                        _product_stats)
    
    real = get_real_values(row, _personal_recommendations)
    score1 = apk(real, predicted1)
    score2 = apk(real, predicted2)
    if count > 0 and score1 != score2:
        print "-- i : ", i, row[1]
        print "--- p1 : ", score1, predicted1, real
        print "--- p2 : ", score2, predicted2
    map7_1 += score1
    map7_2 += score2
    
    count -= 1
    if count == 0:
        break

if len(validation_data) > 0:
    map7_1 /= len(validation_data)
    map7_2 /= len(validation_data)

print map7_1, map7_2


# In[ ]:




# ## Compare predicitions on a single row

# In[64]:

_VERBOSE=True
__VERBOSE=False
row = validation_data[1061]


# In[68]:

zfturbo_compute_predictions(row, get_profiles, best_validation,
                        personal_recommendations_validation,
                        product_stats_validation)


# In[71]:

personal_recommendations_weight = 0.0
_compute_predictions(row, get_profiles,
                        personal_recommendations_validation,
                        common_recommendations_validation,
                        product_stats_validation,
                        personal_recommendations_weight)


# In[ ]:




# In[30]:

#test = [0, 0, 302, 0, 0, 0, 173, 218, 0, 0, 86, 204, 255, 157, 80, 145, 61, 246, 256, 187, 125, 292, 291, 0]
test = [ 0.  ,        0.   ,       0.80606061 , 0.  ,        0.72727273 , 0.80606061,
  0.46666667 , 0.    ,      0.    ,      0.     ,     0.     ,     0.6030303,
  0.77878788 , 0.46969697  ,0.  ,        0.       ,   0.26060606 , 0.54545455,
  0.54242424 , 0.46969697 , 0.  ,        0.58787879  ,0.61212121 , 0.72727273,]

print np.argsort(test)[::-1][:7]
print heapq.nlargest(7, range(len(test)), test.__getitem__)
print sorted(range(len(test)), key=test.__getitem__, reverse=True)[:7]


# In[108]:

print test[22], test[18], test[12]


# In[ ]:




# In[ ]:




# ## Compare on test data

# In[73]:

test_file = '../data/test_ver2.csv'


# ### My preprocessing

# In[74]:

test_data = []
reader = open(test_file, 'r')
reader.readline()
total = 0
limit = 1000
while 1:
    line = reader.readline()[:-1]
    total += 1

    if line == '':
        break

    row = parse_line(line)    
    test_data.append(row)

    if total > limit:
        break
    
    if total % 100000 == 0:
        logging.info('Read {} lines'.format(total))
    
reader.close()


# Check

# In[86]:

_VERBOSE=False
__VERBOSE=False
# def _predict_score(validation_data, get_profiles_func,
#                   personal_recommendations,
#                   common_recommendations,
#                   product_stats,
#                   personal_recommendations_weight):

get_profiles_func = get_profiles
_personal_recommendations = personal_recommendations_validation
_common_recommendations = common_recommendations_validation
_product_stats = product_stats_validation
personal_recommendations_weight = 0.0

logging.debug("-- predict_score : personal_recommendations_weight=%s" % personal_recommendations_weight)
count = 5

for i, row in enumerate(test_data):

    prow = process_data(row)
    predicted1 = _compute_predictions(prow, get_profiles_func,
                                    _personal_recommendations,
                                    _common_recommendations,
                                    _product_stats,
                                    personal_recommendations_weight)

    predicted2 = zfturbo_compute_predictions(row, get_profiles_func,
                                        best_validation,
                                        _personal_recommendations,
                                        _product_stats)
    
    #if count > 0 and predicted1 != predicted2:
    if True:
        print "-- i : ", i, row[1]
        print "--- p1 : ", predicted1, [target_labels[i] for i in predicted1]
        print "--- p2 : ", predicted2, [target_labels[i] for i in predicted2]
    
    count -= 1
    if count == 0:
        break

print "end"


# In[ ]:




# In[81]:

test_data[0]


# In[ ]:




__author__ = 'behzadbehzadan'

"""
In this version, a binary classification applies on 'worried' (positive) and 'not worried' (negative) training sets.
The test set, has data related to british gas tweets.
"""

import random
import math_extra
import my_util
import os
import funcs_worry
import copy
from operator import itemgetter
import worry_mixed_tr_ts_sets


#test_set_type = 'hand_picked'
test_set_type = 'MTurk'
training_set_type = 'worried'
#training_set_type = 'MTurk'



###################################################### read source data ################################################
worried_source_data = worry_mixed_tr_ts_sets.read_worried_data()
worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk = worry_mixed_tr_ts_sets.read_amazon_mech_turk_data()
########################################################################################################################

if test_set_type == 'MTurk':

    positives_test = worrieds_MTurk + tell_not_worry_MTurk
    negatives_test = not_worrieds_MTurk + nothing_MTurk

    positives_test = random.sample(positives_test, len(negatives_test))

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(worry_mixed_tr_ts_sets.home_dir + worry_mixed_tr_ts_sets.save_dir + 'ts_MTurk_neg', False, True, [[t] for t in negatives_test])
    my_util.write_csv_file(worry_mixed_tr_ts_sets.home_dir + worry_mixed_tr_ts_sets.save_dir + 'ts_MTurk_pos', False, True, [[t] for t in positives_test])

    pos_train_labs_test = []
    neg_train_labs_test = []

if test_set_type == 'hand_picked':

    tweets_test_worry, tweets_test_not_worry, tweets_test_others = read_hand_picked_data()

    positives_test = tweets_test_worry
    negatives_test = tweets_test_not_worry + tweets_test_others

    pos_train_labs_test = []
    neg_train_labs_test = []

if training_set_type == 'worried':

    positives, negatives, n_containing_tweets, neg_train_labs = create_pos_neg_sets_for_worried_data(worried_source_data)

    pos_train_labs = ['worried']

    # find all positive tweets that contain 'worry' phrases:
    positives_worry = [t for t in positives if 'worry' in t]
    positives_without_worry = [x for x in positives if x not in positives_worry]
    print "number of positive 'worried' tweets containing 'worry':", len(positives_worry)
    positives = positives_worry + random.sample(positives_without_worry, 10000-len(positives_worry))
    pos_train_labs.extend(['worry'])
    negatives = random.sample(negatives, 10000)

    #positives = random.sample(positives, 10000)
    #negatives = random.sample(negatives, 10000)
    #positives = random.sample(positives, len(negatives))

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(home_dir + save_dir + 'tr_worried_neg', False, True, [[t] for t in negatives])
    my_util.write_csv_file(home_dir + save_dir + 'tr_worried_pos', False, True, [[t] for t in positives])
    my_util.write_csv_file(home_dir + save_dir + 'n_neg_phrases_used', False, True, n_containing_tweets)

if training_set_type == 'Mturk':
    positives = worrieds_MTurk + tell_not_worry_MTurk
    negatives = not_worrieds_MTurk + nothing_MTurk

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(home_dir + save_dir + 'tr_MTurk_neg', False, True, [[t] for t in negatives])
    my_util.write_csv_file(home_dir + save_dir + 'tr_MTurk_pos', False, True, [[t] for t in positives])

    pos_train_labs = []
    neg_train_labs = []

positives = funcs_worry.remove_intersection_from_the_list(positives, positives_test + negatives_test)
negatives = funcs_worry.remove_intersection_from_the_list(negatives, positives_test + negatives_test)

feature_vects_pos, tweet_texts_pos, norm_factors_pos, feature_vects_neg, tweet_texts_neg, norm_factors_neg = \
create_feature_vectors_training(positives, negatives, pos_train_labs, neg_train_labs)

feature_vects_pos_test, tweet_texts_pos_test, norm_factors_pos_test, feature_vects_neg_test, tweet_texts_neg_test, norm_factors_neg_test = \
create_feature_vectors_test(positives_test, negatives_test, pos_train_labs_test, neg_train_labs_test)

n_distinct_features = len(features_dict)

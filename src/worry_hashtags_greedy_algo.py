# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

import os
import random
import funcs_worry
import my_util
import copy
import itertools
import operator
import math
import re
import worry_hashtags_new

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags_at_the_end/24_05_2013/'
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
########################################################################################################################
use_mech_turk_for_training = True
equal_sized_pos_neg_train_sets = True
########################################################################################################################
test_sets = ['hand_picked_data', 'mech_turk']
remove_retweets = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = False
random.seed(7)
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# other labels are associated to any other types of tweets.
class_labels = { 'pos' : +1, 'neg' : -1}#, 'oth' : 0}
#m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
min_ngram = 1
#length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
max_ngram = 3
###################################################### libsvm settings #################################################
# The nu_CSV does not take the C parameter (i.e. the cost function). Hence, there is no weight balancing option.
svm_type = 'C_SVC'#'nu_SVC'#
# Set the kernel. linear --> 0; polynomial --> 1; radial basis --> 2; sigmoid --> 3; precomputed kernel --> 4
kernel_type = 0
# Set the cost parameter for the C_CSV
cost = 10
# Set the nu parameter for the nu_SVC
# Note: if nu is not small enough, the following error message is shown: "specified nu is infeasible"
nu = 0.05
n_fold_cross_val = 10
# Assign different costs to balance unbalanced (different sized) training sets.
balance_sets = True
########################################################################################################################







def find_local_optimum_set(source_tweets_pos, source_tweets_combined_pos, source_tweets_neg,
                           pos_labels, combined_pos_labels, neg_labels, test_set_pos, test_set_neg):

    optimum_pos_labels = []
    optimum_neg_labels = []
    train_dict_pos = {}
    train_dict_combined_pos = {}
    train_dict_neg = {}

    neg_flag = True
    opt_pos_val = None
    for pos_label in pos_labels + combined_pos_labels:
        optimum_pos_labels.extend([pos_label])
        for opt_label_pos in optimum_pos_labels:
            if opt_label_pos in pos_labels:
                train_dict_pos[opt_label_pos] = source_tweets_pos[opt_label_pos]
            if opt_label_pos in combined_pos_labels:
                train_dict_combined_pos[opt_label_pos] = source_tweets_combined_pos[opt_label_pos]

        if neg_flag:
            opt_neg_val = None

            for neg_label in neg_labels:

                optimum_neg_labels.extend([neg_label])

                for opt_label in optimum_neg_labels:
                    train_dict_neg[opt_label] = source_tweets_neg[opt_label]

                ########################################### current dir to save stuff for each iteration# ########################
                tr_set_pos = optimum_pos_labels
                if use_mech_turk_for_training:
                    tr_set_pos = tr_set_pos + ['MechTurk']
                tr_set_neg = optimum_neg_labels
                current_dir = str(tr_set_pos) + '_' + str(tr_set_neg) + '_vs_' + ts_set + '/'
                if not os.path.exists(home_dir + save_dir + current_dir):
                    os.makedirs(home_dir + save_dir + current_dir)
                ##################################################################################################################

                worry_hashtags_new.remove_intersections(train_dict_pos, train_dict_combined_pos, train_dict_neg, current_dir)

                prediction_result, statistic, statistic_dict = \
                    worry_hashtags_new.train_and_test(train_dict_pos, train_dict_neg, train_dict_combined_pos,
                                                      mech_turk_train_set_pos, mech_turk_train_set_neg,
                                                      test_set_pos, test_set_neg,
                                                      current_dir)

                f1_mean = statistic_dict['f1_mean']
                if f1_mean > opt_neg_val:
                    opt_neg_val = f1_mean
                elif f1_mean < opt_neg_val:
                    optimum_neg_labels.remove(neg_label)

            neg_flag = False
            opt_pos_val = opt_neg_val

        else:

            for opt_label in optimum_neg_labels:
                train_dict_neg[opt_label] = source_tweets_neg[opt_label]

            worry_hashtags_new.remove_intersections(train_dict_pos, train_dict_combined_pos, train_dict_neg, current_dir)
            train_set_neg = get_equal_set_for_each_label(train_dict_neg)
            train_lab_neg = train_dict_neg.keys()
            if equal_sized_pos_neg_train_sets:
                min_pos_neg_size = min(len(train_set_pos), len(train_set_neg))
                train_set_pos = train_set_pos[: min_pos_neg_size]
                train_set_neg = train_set_neg[: min_pos_neg_size]
            prediction_result, statistic = train_and_test(train_set_pos, train_lab_pos, train_set_neg, train_lab_neg, train_dict_combined_pos, test_set_pos, test_set_neg)
            f1_mean = statistic['f1_mean']
            if f1_mean > opt_pos_val:
                opt_pos_val = f1_mean
            elif f1_mean < opt_pos_val:
                optimum_pos_labels.remove(pos_label)

    return optimum_neg_labels

hash_labels_pos = ['#worried', '#anxious']
combined_labels_pos = [('worry', 'help'), ('worry', 'eek'), ('anxious', 'help'), ('anxious', 'eek')]
hash_labels_neg = ['#easy', '#calm', '#relaxed']

final_statistics = []
final_header = ['hash_labels_at_end', 'tr_set_pos', 'tr_set_neg', 'ts_set',
    'min_ngram', 'max_ngram', 'n_features', 'svm_params',
    'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
    'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'f1_score_pos', 'f1_score_neg', 'f1_mean', 'f1_stdev']

mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg, \
hand_picked_pos, hand_picked_neg, \
hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg = \
    worry_hashtags_new.get_required_data(hash_labels_pos, combined_labels_pos, hash_labels_neg)

find_local_optimum_set(hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, hash_label_at_end_tweets_dict_neg,
                       hash_labels_pos, combined_labels_pos, hash_labels_neg, hand_picked_pos, hand_picked_neg)


# # sort in descending order based on f_mean.
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
# # sort in ascending order based on f1_stdev
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
# append the header to the beginning.
# statistics.insert(0, header)
# my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)
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
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags_at_the_end/greedy_algo/'
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
########################################################################################################################
hash_labels_pos = ['#worried', '#anxious']
combined_labels_pos = [('worry', 'help'), ('worry', 'eek'), ('anxious', 'help'), ('anxious', 'eek')]
hash_labels_neg = ['#easy', '#calm', '#relaxed']

use_mech_turk_for_training = True
equal_sized_pos_neg_train_sets = True
test_sets = ['hand_picked_data', 'mech_turk']


def find_local_optimum_set(source_tweets_pos, source_tweets_combined_pos, mech_turk_train_set_pos,
                           source_tweets_neg, mech_turk_train_set_neg,
                           pos_labels, combined_pos_labels, neg_labels,
                           test_set_pos, test_set_neg,
                           ts, use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir):

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
                current_dir = str(tr_set_pos) + '_' + str(tr_set_neg) + '_vs_' + ts + '/'
                if not os.path.exists(home_dir + save_dir + current_dir):
                    os.makedirs(home_dir + save_dir + current_dir)
                ##################################################################################################################

                worry_hashtags_new.remove_intersections(train_dict_pos, train_dict_combined_pos, train_dict_neg, save_dir, current_dir)

                prediction_result, statistic, statistic_dict = \
                    worry_hashtags_new.train_and_test(train_dict_pos, train_dict_neg, train_dict_combined_pos,
                                                      mech_turk_train_set_pos, mech_turk_train_set_neg,
                                                      test_set_pos, test_set_neg,
                                                      use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir, current_dir)

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

            worry_hashtags_new.remove_intersections(train_dict_pos, train_dict_combined_pos, train_dict_neg, save_dir, current_dir)

            prediction_result, statistic, statistic_dict = \
                worry_hashtags_new.train_and_test(train_dict_pos, train_dict_neg, train_dict_combined_pos,
                                                  mech_turk_train_set_pos, mech_turk_train_set_neg,
                                                  test_set_pos, test_set_neg,
                                                  use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir, current_dir)

            f1_mean = statistic_dict['f1_mean']
            if f1_mean > opt_pos_val:
                opt_pos_val = f1_mean
            elif f1_mean < opt_pos_val:
                optimum_pos_labels.remove(pos_label)

    return optimum_pos_labels, optimum_neg_labels

mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg, \
hand_picked_pos, hand_picked_neg, \
hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg = \
    worry_hashtags_new.get_required_data(hash_labels_pos, combined_labels_pos, hash_labels_neg)

for ts in test_sets:
    optimum_pos_labels, optimum_neg_labels = \
        find_local_optimum_set(hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, mech_turk_train_set_pos,
                               hash_label_at_end_tweets_dict_neg, mech_turk_train_set_neg,
                               hash_labels_pos, combined_labels_pos, hash_labels_neg, hand_picked_pos, hand_picked_neg,
                               ts, use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir)

    print 'optimum pos/neg labels for ', ts, ' :', optimum_pos_labels, 'and ', optimum_neg_labels


# # sort in descending order based on f_mean.
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
# # sort in ascending order based on f1_stdev
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
# append the header to the beginning.
# statistics.insert(0, header)
# my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)
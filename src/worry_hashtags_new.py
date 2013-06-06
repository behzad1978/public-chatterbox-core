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

home_dir = os.path.expanduser('~')
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
########################################################################################################################
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
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


def remove_intersection_from(the_keyword_tweet_dict, other_keyword_tweet_dict, file_name, save_dir, current_dir):

    header = ['the_keyword', 'the_size', 'other_keyword', 'other_size', 'intersection_size']
    intersection_file = []
    traversed_keywords = []
    comparison_nr = 0
    for keyword, tweets in the_keyword_tweet_dict.items():

        traversed_keywords.append(keyword)
        for other_keyword, other_tweets in other_keyword_tweet_dict.items():

            if other_keyword not in traversed_keywords:

                comparison_nr += 1
                print str(keyword) + ' and ' + str(other_keyword) + ' ...'
                no_intersect = funcs_worry.remove_intersection_from_the_list(tweets, other_tweets)
                the_keyword_tweet_dict[keyword] = no_intersect
                intersection_file.append([keyword, len(tweets), other_keyword, len(other_tweets), len(tweets) - len(no_intersect)])

    print 'number of comparisons to remove intersections:', comparison_nr
    if intersection_file:# check if intersection_file is not empty
        intersection_file.insert(0, header)
        my_util.write_csv_file(home_dir + save_dir + current_dir + 'intersecttion_size_' + file_name, False, True, intersection_file)


def remove_intersections(pos_set, combined_pos_set, neg_set, save_dir, current_dir):
    # remove intersections within the positive labels
    remove_intersection_from(pos_set, pos_set, 'pos_pos', save_dir, current_dir)
    # remove intersections within the combined_pos labels
    remove_intersection_from(combined_pos_set, combined_pos_set, 'combPos_combPos', save_dir, current_dir)
    # remove intersection of the tweets_with_combined_pos_labels from the positive set
    remove_intersection_from(combined_pos_set, pos_set, 'combPos_pos', save_dir, current_dir)
    # remove intersections within the negative set
    remove_intersection_from(neg_set, neg_set, 'neg_neg', save_dir, current_dir)
    # remove intersection of negative set from positive set
    remove_intersection_from(pos_set, neg_set, 'pos_neg', save_dir, current_dir)
    # remove intersection of negative set from combined_positive set
    remove_intersection_from(combined_pos_set, neg_set, 'combPos_neg', save_dir, current_dir)


def get_equal_set_for_each_label(label_tweet_dict):
    sizes = []
    for tweets in label_tweet_dict.values():
        sizes.extend([len(tweets)])
    print sizes

    returning_dict = {}
    for label, tweets in label_tweet_dict.items():
        # take a random equal size for each label
        returning_dict[label] = random.sample(tweets, min(sizes))

    return returning_dict

def aggregate_all_tweets(label_tweet_dict):
    all_tweets = []
    for tweets in label_tweet_dict.values():
        all_tweets.extend(tweets)

    # shuffle to mix all labels
    random.shuffle(all_tweets)

    return all_tweets

def get_raw_data():

    source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'

    hash_tweets_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_30_05_2013_noDup', False, True)
    hash_tweets_noDup = [t[0] for t in hash_tweets_noDup]

    worrieds_mech_turk_1, not_worrieds_mech_turk_1, tell_not_worry_mech_turk_1, nothing_MTurk_1 = \
        funcs_worry.read_amazon_mech_turk_data(home_dir + source_dir + 'AmazonMTurk_1', 2)

    worrieds_mech_turk_2, not_worrieds_mech_turk_2, tell_not_worry_mech_turk_2, nothing_MTurk_2 = \
        funcs_worry.read_amazon_mech_turk_data(home_dir + source_dir + 'AmazonMTurk_2', 2)

    mech_turk_pos_1 = worrieds_mech_turk_1 + tell_not_worry_mech_turk_1
    mech_turk_neg_1 = not_worrieds_mech_turk_1 + nothing_MTurk_1

    mech_turk_pos_2 = worrieds_mech_turk_2 + tell_not_worry_mech_turk_2
    mech_turk_neg_2 = not_worrieds_mech_turk_2 + nothing_MTurk_2

    worried_hand_picked, not_worried_hand_picked, nothing_hand_picked = funcs_worry.read_hand_picked_data(home_dir, source_dir)

    hand_picked_pos = worried_hand_picked
    hand_picked_neg = not_worried_hand_picked + nothing_hand_picked

    return hash_tweets_noDup, mech_turk_pos_1, mech_turk_neg_1, mech_turk_pos_2, mech_turk_neg_2, hand_picked_pos, hand_picked_neg

def filter_tweets_by_labels(hash_tweets, hash_labels_pos, combined_labels_pos, hash_labels_neg):

    # create a dictionary {'label': tweets_with_the_label}
    hash_label_tweets_dict_pos = {}
    hash_label_at_end_tweets_dict_pos = {}
    hash_label_tweets_dict_neg = {}
    hash_label_at_end_tweets_dict_neg = {}
    combined_labels_tweets_dict_pos = {}

    # happy_tweets = funcs_worry.find_happy_tweets(hash_tweets)
    # hash_tweets = [t for t in hash_tweets if t not in happy_tweets]

    for label in hash_labels_pos:
        tweets_with_hash_keyword = funcs_worry.find_tweets_with_hash_label(hash_tweets, label)
        tweets_with_hash_keyword_at_end = funcs_worry.find_tweets_with_hash_label_at_the_end(tweets_with_hash_keyword, label)

        hash_label_tweets_dict_pos[label] = tweets_with_hash_keyword
        hash_label_at_end_tweets_dict_pos[label] = tweets_with_hash_keyword_at_end

    for label in hash_labels_neg:
        tweets_with_hash_keyword = funcs_worry.find_tweets_with_hash_label(hash_tweets, label)
        tweets_with_hash_keyword_at_end = funcs_worry.find_tweets_with_hash_label_at_the_end(tweets_with_hash_keyword, label)

        hash_label_tweets_dict_neg[label] = tweets_with_hash_keyword
        hash_label_at_end_tweets_dict_neg[label] = tweets_with_hash_keyword_at_end

    for label in combined_labels_pos:
        tweets_with_combined_keyword_pos = funcs_worry.find_tweets_with_combined_labels(hash_tweets, label)
        combined_labels_tweets_dict_pos[label] = tweets_with_combined_keyword_pos

    return hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
           hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg


def get_required_data(hash_labels_pos, combined_labels_pos, hash_labels_neg):

    # read all raw data
    hash_tweets, mech_turk_pos_1, mech_turk_neg_1, mech_turk_pos_2, mech_turk_neg_2, hand_picked_pos, hand_picked_neg = get_raw_data()

    # this part is for quick debug
    #hash_tweets = random.sample(hash_tweets, 100)

    # check if there is any intersections:
    hash_tweets = funcs_worry.remove_intersection_from_the_list(hash_tweets, mech_turk_pos_1 + mech_turk_neg_1 + mech_turk_pos_2 + mech_turk_neg_2)
    hash_tweets = funcs_worry.remove_intersection_from_the_list(hash_tweets, hand_picked_pos + hand_picked_neg)

    # take some mech_turk for test set
    mech_turk_test_set_pos = random.sample(mech_turk_pos_1, 500) + random.sample(mech_turk_pos_2, 500)
    mech_turk_test_set_neg = random.sample(mech_turk_neg_1, 500) + random.sample(mech_turk_neg_2, 500)

    # keep the rest of mech_turk for training
    mech_turk_train_set_pos = [t for t in (mech_turk_pos_1 + mech_turk_pos_2) if t not in mech_turk_test_set_pos]
    mech_turk_train_set_neg = [t for t in (mech_turk_neg_1 + mech_turk_neg_2) if t not in mech_turk_test_set_neg]

    hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
    hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg = \
        filter_tweets_by_labels(hash_tweets, hash_labels_pos, combined_labels_pos, hash_labels_neg)

    return mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg, \
           hand_picked_pos, hand_picked_neg, \
           hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
           hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg


def train_and_test(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg, tweets_with_combined_keywords_pos,
                   mech_turk_train_set_pos, mech_turk_train_set_neg, test_set_pos, test_set_neg,
                   use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir, current_dir):

    statistic = []
    header = [
              'min_ngram', 'max_ngram', 'n_features', 'svm_params',
              'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
              'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'f1_score_pos', 'f1_score_neg', 'f1_mean', 'f1_stdev']

    tweets_with_hash_keywords_pos = get_equal_set_for_each_label(tweets_with_hash_keywords_pos)
    all_hash_tweets_pos = aggregate_all_tweets(tweets_with_hash_keywords_pos)

    tweets_with_hash_keywords_neg = get_equal_set_for_each_label(tweets_with_hash_keywords_neg)
    all_hash_tweets_neg = aggregate_all_tweets(tweets_with_hash_keywords_neg)

    tweets_with_combined_keywords_pos = get_equal_set_for_each_label(tweets_with_combined_keywords_pos)

    if equal_sized_pos_neg_train_sets:
        minimum = min(len(all_hash_tweets_pos), len(all_hash_tweets_neg))
        all_hash_tweets_pos = all_hash_tweets_pos[: minimum]
        all_hash_tweets_neg = all_hash_tweets_neg[: minimum]

        # tweets with combined labels are added to the positive set too.
        # Therefore, to make the positive and negative sets stay equal, we need to truncate the positive hash_tweets.
        cap_size = len(all_hash_tweets_pos) - len(aggregate_all_tweets(tweets_with_combined_keywords_pos))
        all_hash_tweets_pos = all_hash_tweets_pos[: cap_size]

        if use_mech_turk_for_training:
            minimum = min(len(mech_turk_train_set_pos), len(mech_turk_train_set_neg))
            mech_turk_train_set_pos = random.sample(mech_turk_train_set_pos, minimum)
            mech_turk_train_set_neg = random.sample(mech_turk_train_set_neg, minimum)

    #all_hash_tweets_pos = all_hash_tweets_pos[: int(0.65*len(all_hash_tweets_pos))]

    print 'creating feature vectors...'

    features_dict = {}
    features_count_dict = {}
    #the very first index is always 1.
    if new_normalisation_flag:
        max_index = 0
    else:
        max_index = 1

    ################################################ hash-tweets feature vects #######################################
    # since there is no intersection between tweets containing keywords,
    # we can send the aggregated tweets into the function below:
    hash_tweets_train_labs_pos = tweets_with_hash_keywords_pos.keys()
    hash_tweets_train_labs_neg = tweets_with_hash_keywords_neg.keys()

    hash_tweets_feature_vects_pos, hash_tweets_texts_pos, max_index, hash_tweets_norm_factors_pos = \
        funcs_worry.get_sparse_feature_vector_worry(
            all_hash_tweets_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_pos, random)

    hash_tweets_feature_vects_neg, hash_tweets_texts_neg, max_index, hash_tweets_norm_factors_neg = \
        funcs_worry.get_sparse_feature_vector_worry(
            all_hash_tweets_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_neg, random)
    ##################################################################################################################

    ################################################ combined tweets feature vects ###################################
    all_combined_tweets_feature_vects_pos = []
    all_combined_tweets_texts_pos = []
    all_combined_tweets_norm_factors_pos = []
    # since tweets with different combined_keywords have different training_labels,
    # we cannot put all of them in a single list and create feature vectors in one go.
    for combined_keywords, combined_tweets in tweets_with_combined_keywords_pos.items():

        combined_train_lab = [keyword for keyword in combined_keywords]

        combined_tweets_feature_vects_pos, combined_tweets_texts_pos, max_index, combined_tweets_norm_factors_pos = \
            funcs_worry.get_sparse_feature_vector_worry(
                combined_tweets, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                remove_stpwds_for_unigrams, combined_tweets, combined_train_lab, random)

        all_combined_tweets_feature_vects_pos.extend(combined_tweets_feature_vects_pos)
        all_combined_tweets_texts_pos.extend(combined_tweets_texts_pos)
        all_combined_tweets_norm_factors_pos.extend(combined_tweets_norm_factors_pos)
    ##################################################################################################################

    ###################################################### mech_turk feature vects ###################################
    mech_turk_feature_vects_pos, mech_turk_texts_pos, max_index, mech_turk_norm_factors_pos = \
        funcs_worry.get_sparse_feature_vector_worry(
            mech_turk_train_set_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

    mech_turk_feature_vects_neg, mech_turk_texts_neg, max_index, mech_turk_norm_factors_neg = \
        funcs_worry.get_sparse_feature_vector_worry(
            mech_turk_train_set_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)
    ##################################################################################################################

    ###################################################### training set ####################################
    train_set_vects_pos = hash_tweets_feature_vects_pos + all_combined_tweets_feature_vects_pos + mech_turk_feature_vects_pos
    train_set_vects_neg = hash_tweets_feature_vects_neg + mech_turk_feature_vects_neg
    ########################################################################################################

    funcs_worry.write_features_count_dict_to_csv(features_count_dict,
                                                 home_dir + save_dir + current_dir + 'features_count_dict_training')
    funcs_worry.write_features_and_freqs_to_csv(features_dict, features_count_dict,
                                                -1, home_dir + save_dir + current_dir + 'features_freq_training')

    ###################################################### test set feature vects ###################################
    test_set_feature_vects_pos, test_set_texts_pos, max_index, test_set_norm_factors_pos = \
        funcs_worry.get_sparse_feature_vector_worry(
            test_set_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

    test_set_feature_vects_neg, test_set_texts_neg, max_index, test_set_norm_factors_neg = \
        funcs_worry.get_sparse_feature_vector_worry(
            test_set_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)
    ########################################################################################################


    funcs_worry.write_features_dict_to_csv(features_dict, home_dir + save_dir + current_dir + 'features_dict')
    n_features = len(features_dict)

    tr_size_pos, tr_size_neg = len(train_set_vects_pos), len(train_set_vects_neg)
    ts_size_pos, ts_size_neg = len(test_set_feature_vects_pos), len(test_set_feature_vects_neg)
    training_sizes = {'pos': tr_size_pos, 'neg': tr_size_neg}

    x_train = train_set_vects_pos + train_set_vects_neg
    y_train = [class_labels['pos']] * len(train_set_vects_pos) + [class_labels['neg']] * len(train_set_vects_neg)

    x_test = test_set_feature_vects_pos + test_set_feature_vects_neg
    y_test = [class_labels['pos']] * len(test_set_feature_vects_pos) + [class_labels['neg']] * len(test_set_feature_vects_neg)
    test_set_texts = test_set_texts_pos + test_set_texts_neg

    svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, class_labels, training_sizes)

    p_label, p_acc, p_val = funcs_worry.train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)

    print 'calculating validation statistics ...'
    prediction_result, accuracy, precisions, recalls, f1_scores = \
        funcs_worry.calc_prediction_stats_2(y_test, test_set_texts, p_label, p_val, class_labels)

    precision_pos, precision_neg = precisions['pos'], precisions['neg']
    recall_pos, recall_neg = recalls['pos'], recalls['neg']
    f1_score_pos, f1_score_neg = f1_scores['pos'], f1_scores['neg']
    f1_mean = round((f1_score_pos + f1_score_neg) / 2, 2)
    f1_stdev = round(math.sqrt((f1_score_pos - f1_mean) ** 2 + (f1_score_neg - f1_mean) ** 2), 2)# note we divide by 2-1.
    my_util.write_csv_file(home_dir + save_dir + current_dir + 'prediction_result', False, True, prediction_result)

    statistic.append(header)
    statistic.append([eval(h) for h in header])
    statistic_dict = dict(zip(header, statistic))

    return prediction_result, statistic, statistic_dict


def search_through_labels(hash_labels_pos, combined_labels_pos, hash_labels_neg,
                          mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg,
                          hand_picked_pos, hand_picked_neg,
                          hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos,
                          hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg,
                          test_sets, hash_labels_at_the_end_flags, use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir):

    header = []
    statistics = []
    basic_header = ['hash_labels_at_end', 'tr_set_pos', 'tr_set_neg', 'ts']
    for hash_labels_at_end in hash_labels_at_the_end_flags:

        # loop through all test sets
        for ts in test_sets:

            # i is the subset length of keywords_source_pos
            for i in range(1, len(hash_labels_pos)+1):
                # j is the subset length of keywords_source_neg
                for j in range(3, len(hash_labels_neg)+1):
                    # k is the subset length of keywords_combined_source_pos
                    for k in range(0, len(combined_labels_pos) + 1):

                        if i+k <> 0 and j <> 0:

                            # loop through all subsets of keywords_source_pos
                            for keywords_pos in itertools.combinations(hash_labels_pos, i):
                                keywords_pos = list(keywords_pos)

                                # loop through all subsets of keywords_source_neg
                                for keywords_neg in itertools.combinations(hash_labels_neg, j):
                                    keywords_neg = list(keywords_neg)

                                    # loop through all subsets of keyword_combined_pos
                                    for keyword_combined_pos in itertools.combinations(combined_labels_pos, k):
                                        keyword_combined_pos = list(keyword_combined_pos)

                                        tr_set_pos = keywords_pos + keyword_combined_pos
                                        tr_set_neg = keywords_neg
                                        if use_mech_turk_for_training:
                                            tr_set_pos = tr_set_pos + ['MechTurk']
                                            tr_set_neg = tr_set_neg + ['MechTurk']

                                        ########################################### current dir to save stuff for each iteration# ########################
                                        current_dir = str(tr_set_pos) + '_' + str(tr_set_neg) + '_vs_' + ts + '/'
                                        if hash_labels_at_end:
                                            current_dir = 'endHash_' + current_dir

                                        if not os.path.exists(home_dir +  save_dir + current_dir):
                                            os.makedirs(home_dir +  save_dir + current_dir)
                                        ##################################################################################################################

                                        tweets_with_hash_keywords_pos = {}
                                        tweets_with_hash_keywords_neg = {}
                                        tweets_with_combined_keywords_pos = {}

                                        for keyword in keywords_pos:

                                            if not hash_labels_at_end:
                                                tweets_with_hash_keywords_pos[keyword] = hash_label_tweets_dict_pos[keyword]
                                                print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_pos[keyword])
                                            else:
                                                tweets_with_hash_keywords_pos[keyword] = hash_label_at_end_tweets_dict_pos[keyword]
                                                print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_pos[keyword])

                                        for keyword in keywords_neg:

                                            if not hash_labels_at_end:
                                                tweets_with_hash_keywords_neg[keyword] = hash_label_tweets_dict_neg[keyword]
                                                print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_neg[keyword])
                                            else:
                                                tweets_with_hash_keywords_neg[keyword] = hash_label_at_end_tweets_dict_neg[keyword]
                                                print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_neg[keyword])

                                        for keyword in keyword_combined_pos:
                                            tweets_with_combined_keywords_pos[keyword] = combined_labels_tweets_dict_pos[keyword]
                                            print 'number of tweets containing ' + str(keyword) + ' :', len(tweets_with_combined_keywords_pos[keyword])

                                        remove_intersections(tweets_with_hash_keywords_pos, tweets_with_combined_keywords_pos,
                                                             tweets_with_hash_keywords_neg, save_dir, current_dir)


                                        if ts == 'hand_picked_data':
                                            test_set_pos = hand_picked_pos
                                            test_set_neg = hand_picked_neg
                                        elif ts == 'mech_turk':
                                            test_set_pos = mech_turk_test_set_pos
                                            test_set_neg = mech_turk_test_set_neg

                                        prediction_result, statistic, statistic_dict = \
                                        train_and_test(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg, tweets_with_combined_keywords_pos,
                                                   mech_turk_train_set_pos, mech_turk_train_set_neg, test_set_pos, test_set_neg,
                                                   use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir, current_dir)

                                        basic_stat = [eval(h) for h in basic_header]
                                        header = basic_header + statistic[0]
                                        statistic = basic_stat + statistic[1]
                                        my_util.write_csv_file(home_dir + save_dir + current_dir + 'statistic', False, True, [header, statistic])
                                        statistics.append(statistic)


    # # sort in descending order based on f_mean.
    # statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
    # # sort in ascending order based on f1_stdev
    # statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
    # append the header to the beginning.
    statistics.insert(0, header)
    my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)

    return statistics


def run_the_codes():

    save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags_at_end_plus_MT/final_exps/new_norm_factor/'

    hash_labels_pos = ['#anxious']#[,'#worried']
    combined_labels_pos = []#[('worry', 'help')]#, ('worry', 'eek'), ('anxious', 'help'), ('anxious', 'eek')]
    hash_labels_neg = ['#easy', '#calm', '#relaxed']

    test_sets = ['hand_picked_data', 'mech_turk']
    hash_labels_at_the_end_flags = [True]#, False]
    use_mech_turk_for_training = True
    equal_sized_pos_neg_train_sets = True

    mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg, \
    hand_picked_pos, hand_picked_neg, \
    hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos, \
    hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg = \
        get_required_data(hash_labels_pos, combined_labels_pos, hash_labels_neg)

    search_through_labels(hash_labels_pos, combined_labels_pos, hash_labels_neg,
                          mech_turk_test_set_pos, mech_turk_test_set_neg, mech_turk_train_set_pos, mech_turk_train_set_neg,
                          hand_picked_pos, hand_picked_neg,
                          hash_label_tweets_dict_pos, hash_label_at_end_tweets_dict_pos, combined_labels_tweets_dict_pos,
                          hash_label_tweets_dict_neg, hash_label_at_end_tweets_dict_neg,
                          test_sets, hash_labels_at_the_end_flags, use_mech_turk_for_training, equal_sized_pos_neg_train_sets, save_dir)

run_the_codes()
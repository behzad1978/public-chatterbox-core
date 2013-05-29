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
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags_at_the_end/20_05_2013/'
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
########################################################################################################################
equal_sized_pos_neg_train_sets = True
hash_labels_at_the_end_flags = [True, False]
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

def read_hash_tweets_source_data():
    if remove_retweets:
        data_source = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_20_05_2013', False, True)
        header = data_source[0]
        tweet_hashtags = [row[header.index('text')] for row in data_source[1:]]
        tweet_hashtags = [t.lower() for t in tweet_hashtags]
        tweet_hashtags = [' '.join(t.split()) for t in tweet_hashtags]
        tweet_hashtags_noDup = funcs_worry.remove_duplicate_tweets(tweet_hashtags, False, None)
        my_util.write_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True,
                               [[t] for t in tweet_hashtags_noDup])
    else:
        tweet_hashtags_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_20_05_2013_noDup', False, True)
        tweet_hashtags_noDup = [t[0] for t in tweet_hashtags_noDup]
        #tweet_hashtags_noDup = tweet_hashtags_noDup[:100]

    return tweet_hashtags_noDup

def find_tweets_with_hash_keyword(tweets, keyword):
    tweets_with_keyword = []
    for tweet_text in tweets:
        if keyword in tweet_text:
            tweets_with_keyword.append(tweet_text)

    return tweets_with_keyword

def find_tweets_with_hash_keyword_at_the_end(tweets, keyword):
    tweets_with_the_keyword_at_the_end = []
    for tweet_text in tweets:

        tweet_text_no_url = funcs_worry.remove_url(tweet_text)

        #find any pattern like: #keyword (emoticons spaces emoticons #another_keyword#and_another_keyword) (spaces emoticons xxxx)
        # the x's at the end (xxxx) could be a sign of sympathy and, hence, a sign of worry!
        pattern = '(?u)' + keyword + r'(\W*\s*W*(#\w+)*)*(\s\W*x*)*$'
        if re.search(pattern, tweet_text_no_url) <> None:
            tweets_with_the_keyword_at_the_end.append(tweet_text)

    return tweets_with_the_keyword_at_the_end

def find_tweets_with_combined_keywords(tweets, combined_keyword):
    tweets_with_the_keyword = []
    for tweet in tweets:
        # some keywords like 'help' might be followed by punctuation chars, like: 'help!'
        # put a space between non-punc-chars and punc-chars
        tweet_separated_puncs = re.sub(r"([^'\".,;:/?\!@#£$%^&*()_\-=+`~])(['\".,;:/?\!@#£$%^&*()_\-=+`~])", r"\1 \2", tweet)
        # put a space between punc-chars (EXCEPT #) and non-punc-chars --> note: #label would stay as it is!
        tweet_separated_puncs = re.sub(r"(['\".,;:/?\!£$%^&*()_\-=+`~])([^'\".,;:/?\!@#£$%^&*()_\-=+`~#@])", r"\1 \2", tweet_separated_puncs)
        # better to split the text --> the keyword 'eek' also exists in 'week'.
        if all(k in tweet_separated_puncs.split() for k in combined_keyword):
            #print tweet
            tweets_with_the_keyword.append(tweet)

    my_util.write_csv_file(home_dir + save_dir + current_dir + str(combined_keyword), False, True, [[t] for t in tweets_with_the_keyword])
    return tweets_with_the_keyword

def remove_intersection_from(the_keyword_tweet_dict, other_keyword_tweet_dict, file_name):

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

###################################################### read source data ################################################
hash_tweets = read_hash_tweets_source_data()

worrieds_mech_turk, not_worrieds_mech_turk, tell_not_worry_mech_turk, nothing_MTurk = funcs_worry.read_amazon_mech_turk_data(home_dir, source_dir, 2)
mech_turk_pos = worrieds_mech_turk + tell_not_worry_mech_turk
mech_turk_neg = not_worrieds_mech_turk + nothing_MTurk
# the negative set size is smaller than the positive one. Hence, select an equal size for the positive set.
mech_turk_pos = random.sample(mech_turk_pos, len(mech_turk_neg))

worried_hand_picked, not_worried_hand_picked, nothing_hand_picked = funcs_worry.read_hand_picked_data(home_dir, source_dir)
hand_picked_pos = worried_hand_picked
hand_picked_neg = not_worried_hand_picked + nothing_hand_picked
########################################################################################################################

keywords_source_pos = ['#worried', '#anxious']
keywords_combined_source_pos = [('worry', 'help'), ('worry', 'eek'), ('anxious', 'help'), ('anxious', 'eek')]
keywords_source_neg = ['#easy', '#calm', '#relaxed']

###################################################### read source labels ################################################
source_tweets_with_hash_keywords_pos = {}
source_tweets_with_hash_keywords_at_end_pos = {}
source_tweets_with_combined_keywords_pos = {}
for keyword in keywords_source_pos:

    tweets_with_hash_keyword = find_tweets_with_hash_keyword(hash_tweets, keyword)
    tweets_with_hash_keyword_at_end = find_tweets_with_hash_keyword_at_the_end(tweets_with_hash_keyword, keyword)

    min_size = min(len(tweets_with_hash_keyword), len(tweets_with_hash_keyword_at_end))

    source_tweets_with_hash_keywords_pos[keyword] = tweets_with_hash_keyword[: min_size]
    source_tweets_with_hash_keywords_at_end_pos[keyword] = tweets_with_hash_keyword_at_end[: min_size]

source_tweets_with_hash_keywords_neg = {}
source_tweets_with_hash_keywords_at_end_neg = {}
for keyword in keywords_source_neg:

    tweets_with_hash_keyword = find_tweets_with_hash_keyword(hash_tweets, keyword)
    tweets_with_hash_keyword_at_end = find_tweets_with_hash_keyword_at_the_end(tweets_with_hash_keyword, keyword)

    min_size = min(len(tweets_with_hash_keyword), len(tweets_with_hash_keyword_at_end))

    source_tweets_with_hash_keywords_neg[keyword] = tweets_with_hash_keyword[: min_size]
    source_tweets_with_hash_keywords_at_end_neg[keyword] = tweets_with_hash_keyword_at_end[: min_size]

for keyword in keywords_combined_source_pos:
    tweets_with_combined_keyword_pos = find_tweets_with_combined_keywords(hash_tweets, keyword)
    source_tweets_with_combined_keywords_pos[keyword] = tweets_with_combined_keyword_pos
########################################################################################################################

statistics = []
header = ['hash_labels_at_end', 'tr_set_pos', 'tr_set_neg', 'ts_set',
    'min_ngram', 'max_ngram', 'n_features', 'svm_params',
    'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
    'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'f1_score_pos', 'f1_score_neg', 'f1_mean', 'f1_stdev']

best_combined_label = {}
best_neg_label = {}
best_pos_label = {}

for k in range(0, len(keywords_combined_source_pos) + 1):
    best_combined_label[k] = None
for j in range(0, len(keywords_source_neg) + 1):
    best_neg_label[j] = None
for i in range(0, len(keywords_source_pos)+1):
    best_pos_label[i] = None

for hash_labels_at_end in hash_labels_at_the_end_flags:

    # loop through all test sets
    for ts_set in test_sets:

        # i is the subset length of keywords_source_pos
        for i in range(0, len(keywords_source_pos)+1):
            # j is the subset length of keywords_source_neg
            for j in range(0, len(keywords_source_neg)+1):
                # k is the subset length of keywords_combined_source_pos
                for k in range(0, len(keywords_combined_source_pos) + 1):

                    if i+k <> 0 and j <> 0:

                        if best_pos_label[i] == None:
                            # loop through all subsets of keywords_source_pos
                            pos_labels_iteration_list = itertools.combinations(keywords_source_pos, i)
                            best_val = None
                        else:
                            pos_labels_iteration_list = [best_pos_label[i]]

                        # loop through all subsets of keywords_source_pos
                        for keywords_pos in pos_labels_iteration_list:
                            keywords_pos = list(keywords_pos)



                            if best_neg_label[j] == None:
                                # loop through all subsets of keywords_source_neg
                                neg_labels_iteration_list = itertools.combinations(keywords_source_neg, j)
                                best_val = None
                            else:
                                neg_labels_iteration_list = [best_neg_label[j]]

                            for keywords_neg in neg_labels_iteration_list:
                                keywords_neg = list(keywords_neg)



                                if best_combined_label[k] == None:
                                    # loop through all subsets of keyword_combined_pos
                                    combined_labels_iteration_list = itertools.combinations(keywords_combined_source_pos, k)
                                    best_val = None
                                else:
                                    combined_labels_iteration_list = [best_combined_label[k]]

                                for keyword_combined_pos in combined_labels_iteration_list:
                                    keyword_combined_pos = list(keyword_combined_pos)

                                    tr_set_pos = keywords_pos + keyword_combined_pos
                                    tr_set_neg = keywords_neg

                                    ########################################### current dir to save stuff for each iteration# ########################
                                    current_dir = str(tr_set_pos) + '_' + str(tr_set_neg) + '_vs_' + ts_set + '/'
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
                                            tweets_with_hash_keywords_pos[keyword] = source_tweets_with_hash_keywords_pos[keyword]
                                            print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_pos[keyword])
                                        else:
                                            tweets_with_hash_keywords_pos[keyword] = source_tweets_with_hash_keywords_at_end_pos[keyword]
                                            print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_pos[keyword])

                                    for keyword in keywords_neg:

                                        if not hash_labels_at_end:
                                            tweets_with_hash_keywords_neg[keyword] = source_tweets_with_hash_keywords_neg[keyword]
                                            print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_neg[keyword])
                                        else:
                                            tweets_with_hash_keywords_neg[keyword] = source_tweets_with_hash_keywords_at_end_neg[keyword]
                                            print 'number of tweets containing ' + keyword + ' :', len(tweets_with_hash_keywords_neg[keyword])

                                    for keyword in keyword_combined_pos:
                                        tweets_with_combined_keywords_pos[keyword] = source_tweets_with_combined_keywords_pos[keyword]
                                        print 'number of tweets containing ' + keyword + ' :', len(tweets_with_combined_keywords_pos[keyword])

                                    ############################################# remove intersections ###############################################
                                    # remove intersections within the positive labels
                                    remove_intersection_from(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_pos, 'pos_pos')
                                    # remove intersections within the combined_pos labels
                                    remove_intersection_from(tweets_with_combined_keywords_pos, tweets_with_combined_keywords_pos, 'combPos_combPos')
                                    # remove intersection of the tweets_with_combined_keywords_pos from the positive set
                                    remove_intersection_from(tweets_with_combined_keywords_pos, tweets_with_hash_keywords_pos, 'combPos_pos')
                                    # remove intersections within the negative set
                                    remove_intersection_from(tweets_with_hash_keywords_neg, tweets_with_hash_keywords_neg, 'neg_neg')
                                    # remove intersection of negative set from positive set
                                    remove_intersection_from(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg, 'pos_neg')
                                    # remove intersection of negative set from combined_positive set
                                    remove_intersection_from(tweets_with_combined_keywords_pos, tweets_with_hash_keywords_neg, 'combPos_neg')
                                    ##################################################################################################################

                                    sizes_pos =[]
                                    for tweets in tweets_with_hash_keywords_pos.values():
                                        sizes_pos.extend([len(tweets)])
                                    print sizes_pos

                                    sizes_neg = []
                                    for tweets in tweets_with_hash_keywords_neg.values():
                                        sizes_neg.extend([len(tweets)])
                                    print sizes_neg

                                    all_hash_tweets_pos = []
                                    for tweets in tweets_with_hash_keywords_pos.values():
                                        random.shuffle(tweets)
                                        # take an equal size of each label
                                        all_hash_tweets_pos.extend(tweets[: min(sizes_pos)])
                                        #all_hash_tweets_pos.extend(tweets)

                                    all_hash_tweets_neg = []
                                    for tweets in tweets_with_hash_keywords_neg.values():
                                        random.shuffle(tweets)
                                        # take an equal size of each label
                                        all_hash_tweets_neg.extend(tweets[: min(sizes_neg)])
                                        #all_hash_tweets_neg.extend(tweets)

                                    hash_tweets_train_labs_pos = tweets_with_hash_keywords_pos.keys()
                                    hash_tweets_train_labs_neg = tweets_with_hash_keywords_neg.keys()

                                    #################################################### training set size curve #####################################
                                    # shuffle to mix all labels --> this part is also needed
                                    random.shuffle(all_hash_tweets_pos)
                                    random.shuffle(all_hash_tweets_neg)

                                    if equal_sized_pos_neg_train_sets:
                                        min_pos_neg_size = min(len(all_hash_tweets_pos), len(all_hash_tweets_neg))
                                        training_set_pos = all_hash_tweets_pos[: min_pos_neg_size]
                                        training_set_neg = all_hash_tweets_neg[: min_pos_neg_size]

                                    my_util.write_csv_file(home_dir + save_dir+current_dir + 'training_set_pos', False, True, [[t] for t in training_set_pos])
                                    my_util.write_csv_file(home_dir + save_dir+current_dir + 'training_set_neg', False, True, [[t] for t in training_set_neg])

                                    print 'creating feature vectors...'

                                    features_dict = {}
                                    features_count_dict = {}
                                    #the very first index is always 1.
                                    if new_normalisation_flag:
                                        max_index = 0
                                    else:
                                        max_index = 1

                                    # since there is no intersection between tweets containing keywords,
                                    # we can send the aggregated tweets into the function below:
                                    hash_tweets_feature_vects_pos, hash_tweets_texts_pos, max_index, hash_tweets_norm_factors_pos = \
                                        funcs_worry.get_sparse_feature_vector_worry(
                                            training_set_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                        remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_pos, random)

                                    hash_tweets_feature_vects_neg, hash_tweets_texts_neg, max_index, hash_tweets_norm_factors_neg = \
                                        funcs_worry.get_sparse_feature_vector_worry(
                                            training_set_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                        remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_neg, random)

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

                                    ###################################################### training set ####################################
                                    train_set_vects_pos = hash_tweets_feature_vects_pos + all_combined_tweets_feature_vects_pos
                                    train_set_vects_neg = hash_tweets_feature_vects_neg
                                    ########################################################################################################

                                    funcs_worry.write_features_count_dict_to_csv(features_count_dict,
                                                                                 home_dir + save_dir + current_dir + 'features_count_dict_training')
                                    funcs_worry.write_features_and_freqs_to_csv(features_dict, features_count_dict,
                                                                                -1, home_dir + save_dir + current_dir + 'features_freq_training')

                                    if ts_set == 'hand_picked_data':

                                        hand_picked_feature_vects_pos, hand_picked_texts_pos, max_index, hand_picked_norm_factors_pos = \
                                            funcs_worry.get_sparse_feature_vector_worry(
                                                hand_picked_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                                        hand_picked_feature_vects_neg, hand_picked_texts_neg, max_index, hand_picked_norm_factors_neg = \
                                            funcs_worry.get_sparse_feature_vector_worry(
                                                hand_picked_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                                        ###################################################### test set ##############################################
                                        test_set_vects_pos = hand_picked_feature_vects_pos
                                        test_set_vects_neg = hand_picked_feature_vects_neg

                                        test_set_texts_pos = hand_picked_texts_pos
                                        test_set_texts_neg = hand_picked_texts_neg
                                        ##############################################################################################################

                                    if ts_set == 'mech_turk':

                                        mech_turk_feature_vects_pos, mech_turk_texts_pos, max_index, mech_turk_norm_factors_pos = \
                                            funcs_worry.get_sparse_feature_vector_worry(
                                                mech_turk_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                                        mech_turk_feature_vects_neg, mech_turk_texts_neg, max_index, mech_turk_norm_factors_neg = \
                                            funcs_worry.get_sparse_feature_vector_worry(
                                                mech_turk_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                                        ###################################################### test set ##############################################
                                        test_set_vects_pos = mech_turk_feature_vects_pos
                                        test_set_vects_neg = mech_turk_feature_vects_neg

                                        test_set_texts_pos = mech_turk_texts_pos
                                        test_set_texts_neg = mech_turk_texts_neg
                                        ##############################################################################################################

                                    funcs_worry.write_features_dict_to_csv(features_dict, home_dir + save_dir + current_dir + 'features_dict')
                                    n_features = len(features_dict)

                                    tr_size_pos, tr_size_neg = len(train_set_vects_pos), len(train_set_vects_neg)
                                    ts_size_pos, ts_size_neg = len(test_set_vects_pos), len(test_set_vects_neg)
                                    training_sizes = {'pos':tr_size_pos, 'neg':tr_size_neg}

                                    x_train = train_set_vects_pos + train_set_vects_neg
                                    y_train = [class_labels['pos']] * len(train_set_vects_pos) + [class_labels['neg']] * len(train_set_vects_neg)

                                    x_test = test_set_vects_pos + test_set_vects_neg
                                    y_test = [class_labels['pos']] * len(test_set_vects_pos) + [class_labels['neg']] * len(test_set_vects_neg)
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
                                    f1_stdev = round(math.sqrt((f1_score_pos-f1_mean) ** 2 + (f1_score_neg-f1_mean) ** 2), 2)# note we divide by 2-1.
                                    my_util.write_csv_file(home_dir + save_dir + current_dir + 'prediction_result', False, True, prediction_result)

                                    statistic = [eval(h) for h in header]
                                    my_util.write_csv_file(home_dir + save_dir + current_dir + 'statistic', False, True, [header, statistic])
                                    statistics.append(statistic)

                                    if best_combined_label[k] == None:
                                        if f1_mean > best_val:
                                            best_val = f1_mean
                                            best_label = keyword_combined_pos


                                if best_combined_label[k] == None:
                                    best_combined_label[k] = best_label
                                    best_val = None

                                if best_neg_label[j] == None:
                                    if f1_mean > best_val:
                                        best_val = f1_mean
                                        best_label = keywords_neg

                            if best_neg_label[j] == None:
                                best_neg_label[j] = best_label
                                best_val = None


# # sort in descending order based on f_mean.
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
# # sort in ascending order based on f1_stdev
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
# append the header to the beginning.
statistics.insert(0, header)
my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)
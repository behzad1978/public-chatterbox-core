__author__ = 'behzadbehzadan'

import os
import random
import funcs_worry
import my_util
import copy
import itertools
import operator
import math

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags/'
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
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
        data_source = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags', False, True)
        header = data_source[0]
        tweet_hashtags = [row[header.index('text')] for row in data_source]
        tweet_hashtags = [t.lower() for t in tweet_hashtags]
        tweet_hashtags = [' '.join(t.split()) for t in tweet_hashtags]
        tweet_hashtags_noDup = funcs_worry.remove_duplicate_tweets(tweet_hashtags, False, None)
        my_util.write_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True,
                               [[t] for t in tweet_hashtags_noDup])
    else:
        tweet_hashtags_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True)
        tweet_hashtags_noDup = [t[0] for t in tweet_hashtags_noDup]
        #tweet_hashtags_noDup = tweet_hashtags_noDup[:100]

    return tweet_hashtags_noDup

def find_tweets_with_keyword(tweets, keyword):
    tweets_with_the_keyword = []
    for tweet in tweets:
        if keyword in tweet:
            tweets_with_the_keyword.append(tweet)

    return tweets_with_the_keyword

def find_tweets_with_combined_keywords(tweets, combined_keywords):
    tweets_with_the_keyword = []
    for tweet in tweets:
        if all(k in tweet for k in combined_keywords):
            tweets_with_the_keyword.append(tweet)

    return tweets_with_the_keyword

def remove_intersections(tweets_with_keywords_pos, tweets_with_keywords_neg):

    intersection_file = []
    comparison_nr = 0
    tweets_with_keywords = copy.deepcopy(tweets_with_keywords_pos)
    tweets_with_keywords.update(tweets_with_keywords_neg)

    traversed_keywords = []
    for keyword, tweets in tweets_with_keywords.items():

        traversed_keywords.append(keyword)
        no_intersect = tweets[:]
        for other_keyword, other_tweets in tweets_with_keywords.items():

            if other_keyword not in traversed_keywords:

                comparison_nr+=1
                print keyword + ' and ' + other_keyword + ' ...'
                other_tweets = tweets_with_keywords[other_keyword]
                no_intersect = funcs_worry.remove_intersection_from_the_list(no_intersect, other_tweets)
                # # this part is to visualise what is the original intersection between two lists
                # real_no_intersect = funcs_worry.remove_intersection_from_the_list(tweets, other_tweets)
                # intersection_file.append([keyword, other_keyword, len(tweets)-len(real_no_intersect)])

        if keyword in tweets_with_keywords_pos:
            tweets_with_keywords_pos[keyword] = no_intersect
        if keyword in tweets_with_keywords_neg:
            tweets_with_keywords_neg[keyword] = no_intersect
    print 'number of comparisons to remove intersections:', comparison_nr
    my_util.write_csv_file(home_dir + source_dir + 'hash_tweets_intersect_size', False, True, intersection_file)

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
keywords_source_neg = ['#easy', '#relaxed', '#calm']

statistics = []
header = ['tr_set_pos', 'tr_set_neg', 'ts_set',
    'min_ngram', 'max_ngram', 'n_features', 'svm_params',
    'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
    'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'f1_score_pos', 'f1_score_neg', 'f1_mean', 'f1_stdev']

# loop through all test sets
for ts_set in test_sets:

    # i is the subset length of keywords_source_pos
    for i in range(1, len(keywords_source_pos)+1):
        # j is the subset length of keywords_source_neg
        for j in range(1, len(keywords_source_neg)+1):

            # loop through all subsets of keywords_source_pos
            for keywords_pos in itertools.combinations(keywords_source_pos, i):
                keywords_pos = list(keywords_pos)
                # loop through all subsets of keywords_source_neg
                for keywords_neg in itertools.combinations(keywords_source_neg, j):
                    keywords_neg = list(keywords_neg)

                    tr_set_pos, tr_set_neg = keywords_pos, keywords_neg

                    ############################################### current dir to save stuff for each iteration# ###############################
                    current_dir = ''
                    for s in tr_set_pos + tr_set_neg:
                        current_dir = current_dir + s + '_'
                    current_dir = current_dir + 'vs_' + ts_set
                    current_dir = current_dir + '/'

                    if not os.path.exists(home_dir +  save_dir + current_dir):
                        os.makedirs(home_dir +  save_dir + current_dir)
                    #############################################################################################################################

                    tweets_with_hash_keywords_pos = {}
                    tweets_with_hash_keywords_neg = {}

                    for keyword in keywords_pos:
                        tweets_with_keyword = find_tweets_with_keyword(hash_tweets, keyword)
                        tweets_with_hash_keywords_pos[keyword] = tweets_with_keyword
                        print 'number of tweets containing '+ keyword + ' :', len(tweets_with_keyword)

                    for keyword in keywords_neg:
                        tweets_with_keyword = find_tweets_with_keyword(hash_tweets, keyword)
                        tweets_with_hash_keywords_neg[keyword] = tweets_with_keyword
                        print 'number of tweets containing ' + keyword + ' :', len(tweets_with_keyword)

                    remove_intersections(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg)

                    all_hash_tweets_pos = []
                    for tweets in tweets_with_hash_keywords_pos.values():
                        all_hash_tweets_pos.extend(tweets)

                    all_hash_tweets_neg = []
                    for tweets in tweets_with_hash_keywords_neg.values():
                        all_hash_tweets_neg.extend(tweets)

                    hash_tweets_train_labs_pos = tweets_with_hash_keywords_pos.keys()
                    hash_tweets_train_labs_neg = tweets_with_hash_keywords_neg.keys()

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
                        all_hash_tweets_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                        remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_pos, random)

                    hash_tweets_feature_vects_neg, hash_tweets_texts_neg, max_index, hash_tweets_norm_factors_neg = \
                        funcs_worry.get_sparse_feature_vector_worry(
                        all_hash_tweets_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                        remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_neg, random)

                    ###################################################### training set ####################################
                    train_set_vects_pos = hash_tweets_feature_vects_pos
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

                        ###################################################### test set ####################################
                        test_set_vects_pos = hand_picked_feature_vects_pos
                        test_set_vects_neg = hand_picked_feature_vects_neg

                        test_set_texts_pos = hand_picked_texts_pos
                        test_set_texts_neg = hand_picked_texts_neg
                        ####################################################################################################

                    if ts_set == 'mech_turk':

                        mech_turk_feature_vects_pos, mech_turk_texts_pos, max_index, mech_turk_norm_factors_pos = \
                            funcs_worry.get_sparse_feature_vector_worry(
                                mech_turk_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                        mech_turk_feature_vects_neg, mech_turk_texts_neg, max_index, mech_turk_norm_factors_neg = \
                            funcs_worry.get_sparse_feature_vector_worry(
                                mech_turk_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                        ###################################################### test set ####################################
                        test_set_vects_pos = mech_turk_feature_vects_pos
                        test_set_vects_neg = mech_turk_feature_vects_neg

                        test_set_texts_pos = mech_turk_texts_pos
                        test_set_texts_neg = mech_turk_texts_neg
                        ####################################################################################################

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

                    statistics.append([eval(h) for h in header])

# # sort in descending order based on f_mean.
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
# # sort in ascending order based on f1_stdev
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
# append the header to the beginning.
statistics.insert(0, header)
my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)
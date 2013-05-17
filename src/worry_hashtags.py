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
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags_diff_train_size/16_10_2013/'
# source_dir = '/worry_hashtags/source/'
# save_dir = '/worry_hashtags/'
########################################################################################################################
n_slices = 10
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
        data_source = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_16_05_2013', False, True)
        #data_source = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_10_05_2013', False, True)
        header = data_source[0]
        tweet_hashtags = [row[header.index('text')] for row in data_source[1:]]
        tweet_hashtags = [t.lower() for t in tweet_hashtags]
        tweet_hashtags = [' '.join(t.split()) for t in tweet_hashtags]
        tweet_hashtags_noDup = funcs_worry.remove_duplicate_tweets(tweet_hashtags, False, None)
        my_util.write_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True,
                               [[t] for t in tweet_hashtags_noDup])
    else:
        tweet_hashtags_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_16_05_2013_noDup', False, True)
        #tweet_hashtags_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_10_05_2013_noDup', False, True)
        tweet_hashtags_noDup = [t[0] for t in tweet_hashtags_noDup]
        #tweet_hashtags_noDup = tweet_hashtags_noDup[:100]

    return tweet_hashtags_noDup

def find_tweets_with_keyword(tweets, keyword):
    tweets_with_the_keyword = []
    for tweet in tweets:
        if keyword in tweet:
            tweets_with_the_keyword.append(tweet)

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

keywords_source_pos = ['#worried']#, '#anxious']
#keywords_combined_source_pos = [('worry', 'help'), ('worry', 'eek'), ('anxious', 'help'), ('anxious', 'eek')]
keywords_source_neg = ['#easy']#, '#relaxed', '#calm']

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
                    current_dir = str(tr_set_pos) + '_' + str(tr_set_neg) + '_vs_' + ts_set + '/'

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

                    # remove intersections within the positive set
                    remove_intersection_from(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_pos, 'pos_pos')
                    # remove intersections within the negative set
                    remove_intersection_from(tweets_with_hash_keywords_neg, tweets_with_hash_keywords_neg, 'neg_neg')
                    # remove intersection of negative set from positive set
                    remove_intersection_from(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg, 'pos_neg')

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

                    #################################################### training set size curve #################################################
                    # shuffle to mix all labels --> this part is also needed
                    min_size = min(len(all_hash_tweets_pos), len(all_hash_tweets_neg))
                    random.shuffle(all_hash_tweets_pos)
                    random.shuffle(all_hash_tweets_neg)
                    all_hash_tweets_pos = all_hash_tweets_pos[: min_size]
                    all_hash_tweets_neg = all_hash_tweets_neg[: min_size]

                    slice_size_pos = int(math.ceil(len(all_hash_tweets_pos)/n_slices))
                    slice_size_neg = int(math.ceil(len(all_hash_tweets_neg)/n_slices))
                    for n in range(1, n_slices+1):

                        training_set_pos = all_hash_tweets_pos[0 : n*slice_size_pos]
                        training_set_neg = all_hash_tweets_neg[0 : n*slice_size_neg]


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

                        ###################################################### training set ####################################
                        train_set_vects_pos = hash_tweets_feature_vects_pos
                        train_set_vects_neg = hash_tweets_feature_vects_neg
                        ########################################################################################################

                        funcs_worry.write_features_count_dict_to_csv(features_count_dict,
                                                                     home_dir + save_dir + current_dir + 'features_count_dict_training_'+str(n))
                        funcs_worry.write_features_and_freqs_to_csv(features_dict, features_count_dict,
                                                                    -1, home_dir + save_dir + current_dir + 'features_freq_training_'+str(n))

                        if ts_set == 'hand_picked_data':

                            hand_picked_feature_vects_pos, hand_picked_texts_pos, max_index, hand_picked_norm_factors_pos = \
                                funcs_worry.get_sparse_feature_vector_worry(
                                    hand_picked_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                    remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                            hand_picked_feature_vects_neg, hand_picked_texts_neg, max_index, hand_picked_norm_factors_neg = \
                                funcs_worry.get_sparse_feature_vector_worry(
                                    hand_picked_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
                                    remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

                            #funcs_worry.write_features_dict_to_csv(features_dict, home_dir + save_dir + current_dir + 'features_dict_train' + str(n))

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

                        funcs_worry.write_features_dict_to_csv(features_dict, home_dir + save_dir + current_dir + 'features_dict_' + str(n))
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
                        my_util.write_csv_file(home_dir + save_dir + current_dir + 'prediction_result_' + str(n), False, True, prediction_result)

                        statistic = [eval(h) for h in header]
                        my_util.write_csv_file(home_dir + save_dir + current_dir + 'statistic_' + str(n), False, True, [header, statistic])
                        statistics.append(statistic)

# # sort in descending order based on f_mean.
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_mean')), reverse = True)
# # sort in ascending order based on f1_stdev
# statistics = sorted(statistics, key = operator.itemgetter(header.index('f1_stdev')), reverse = False)
# append the header to the beginning.
statistics.insert(0, header)
my_util.write_csv_file(home_dir + save_dir + 'statistics', False, True, statistics)
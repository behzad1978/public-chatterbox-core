__author__ = 'behzadbehzadan'

"""
In this version, a binary classification applies on 'worried' (positive) and 'not worried' (negative) training sets.
The test set, however, contains 'concerned' (positive) and 'not concerned' (negative) tweets. This is to examine how a
system trained on 'worried' collection can be applied on similar collections like 'concerned'.
"""

import random
import math_extra
import my_util
import os
import funcs_worry
import copy
from operator import itemgetter

home_dir = os.path.expanduser('~')
# source_dir = '/Worried_12/Sources/'
# save_dir = '/Worried_12/worried_vs_no_hot_water/'
source_dir = '/Chatterbox_UCL_Advance/Worry/Sources/'
save_dir = '/Chatterbox_UCL_Advance/Worry/worried_vs_no_hot_water/'
collection_name_train = 'worried'
collection_name_test = 'no_hot_water'
source_file = 'source' + '_' + collection_name_train
source_file_test = 'source' + '_' + collection_name_test
source_file_noDup = source_file + '_noDup'
source_file_noDup_test = source_file_test + '_noDup'
labels_features_file_name = 'labels_features'
tweet_texts_file_name = 'all_tweet_texts'
norm_factor_file_name = 'norm_factor'
result_file_name = 'result'
features_dict_file_name = 'features_dict'
features_count_dict_file_name = 'features_count_dict'
table_file_name = 'table'
# the train_lab is the list of key-phrases that are used to detect negative tweets from positive tweets.
# These phrases must not exist - and, hence, excluded - from the feature space.
neg_train_labs = funcs_worry.get_negative_phrases(collection_name_train)
pos_train_labs = [collection_name_train]# + ['worry'] + ['worries'] + ['worrie'] + ['worr']

########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
read_data_from_file = False
strip_thresholds = [0]
random.seed(7)
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# other labels are associated to any other types of tweets.
labels = { 'pos' : +1, 'neg' : -1}#, 'oth' : 0}
#m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
m = 1
#length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
n = 3
###################################################### libsvm settings #################################################
# The nu_CSV does not take the C parameter (i.e. the cost function). Hence, there is no weight balancing option.
svm_type = 'C_SVC'#'nu_SVC'#
# Set the kernel. linear --> 0; polynomial --> 1; radial basis --> 2; sigmoid --> 3; precomputed kernel --> 4
kernel_type = 0
# Set the cost parameter for the C_CSV
cost = 1
# Set the nu parameter for the nu_SVC
# Note: if nu is not small enough, the following error message is shown: "specified nu is infeasible"
nu = 0.05
# Assign different costs to balance unbalanced (different sized) training sets.
balance_sets = True
use_even_test_sets = False
########################################################################################################################

labels_pos = []
labels_neg = []
labels_oth = []
all_labels = []
feature_vects_pos = []
feature_vects_neg = []
feature_vects_oth = []
all_feature_vects = []
tweet_texts_pos = []
tweet_texts_neg = []
tweet_texts_oth = []
all_texts = []
norm_factors_pos = []
norm_factors_neg = []
norm_factors_oth = []

# {'feature' : feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
# when using svm, an address represents a dimension on the input-space. So it is important to keep the address
# consistent for positive and negative sets and also for training and test sets.
features_dict = dict()
# {feature_address (dimension no) : 'feature'} --> used for debugging to visualise features
features_dict_reverse = dict()
#{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
features_count_dict = dict()
# whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

if remove_retweets:
    try:
        tweets_noDup = my_util.read_csv_file(home_dir + source_dir + source_file_noDup, False, True)
        tweets = [t[0] for t in tweets_noDup[:100]]
        tweets = [t.lower() for t in tweets]
        # remove extra spaces that may exist between words. Is good for when finding not worried tweets, as we look
        # for certain strings like 'aint worried' (don't care about one or double space between 'aint' & 'worried')
        tweets = [' '.join(t.split()) for t in tweets]
    except IOError:
        #read the source file --> [[text1], [text2], [test3], ...]
        tweets = my_util.read_csv_file(home_dir + source_dir + source_file, False, True)
        #create list of texts --> [text1, text2, text3, ...]
        tweets = [t[0] for t in tweets]
        tweets = funcs_worry.remove_retweets(tweets, use_qr_to_remove_dups)
        my_util.write_csv_file(home_dir + source_dir + source_file_noDup, False, True, [[t] for t in tweets])
    try:
        tweets_noDup_test = my_util.read_csv_file(home_dir + source_dir + source_file_noDup_test, False, True)
        tweets_test = [t[0] for t in tweets_noDup_test[:100]]
        tweets_test = [t.lower() for t in tweets_test]
        # remove extra spaces that may exist between words. Is good for when finding not worried tweets, as we look
        # for certain strings like 'aint worried' (don't care about one or double space between 'aint' & 'worried')
        tweets_test = [' '.join(t.split()) for t in tweets_test]
    except IOError:
        #read the source file --> [[text1], [text2], [test3], ...]
        tweets_test = my_util.read_csv_file(home_dir + source_dir + source_file_test, False, True)
        # exclude header:
        tweets_test = tweets_test[1:]
        # the second column are the texts
        tweets_test = [t[1] for t in tweets_test]
        #create list of texts --> [text1, text2, text3, ...]
        tweets_test = funcs_worry.remove_retweets(tweets_test, use_qr_to_remove_dups)
        my_util.write_csv_file(home_dir + source_dir + source_file_noDup_test, False, True, [[t] for t in tweets_test])

    #create pos/neg sets for training set.
    positives, negatives = funcs_worry.find_pos_neg_tweets(collection_name_train, tweets)

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(home_dir + source_dir + 'not_' + collection_name_train, False, True, [[t] for t in negatives])
    my_util.write_csv_file(home_dir + source_dir + collection_name_train, False, True, [[t] for t in positives])

    print 'creating feature vectors...'

    #the very first index is always 1.
    if new_normalisation_flag:
        max_index = 0
    else:
        max_index = 1

    feature_vects_pos, tweet_texts_pos, max_index, norm_factors_pos = funcs_worry.get_sparse_feature_vector_worry(
        positives, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, pos_train_labs, random)

    feature_vects_neg, tweet_texts_neg, max_index, norm_factors_neg = funcs_worry.get_sparse_feature_vector_worry(
        negatives, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, neg_train_labs, random)

    feature_vects_test, tweet_texts_test, max_index, norm_factors_test = funcs_worry.get_sparse_feature_vector_worry(
        tweets_test, features_dict, features_count_dict, max_index, m, n,  remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

    features_dict_reverse = funcs_worry.get_features_dict_reverse(features_dict)

    print 'feature vectors created!', 'No of distinct features:', len(features_dict)

    #create a list from feature_dict in the form of [ ['feature', address], ...] to save in a csv file (tab deliminated)
    feature_list = [list(z) for z in zip(features_dict.keys(), features_dict.values())]
    my_util.write_csv_file(home_dir + save_dir + features_dict_file_name, True, True, feature_list)
    #create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
    feature_count_list = [list(z) for z in zip(features_count_dict.keys(), features_count_dict.values())]
    my_util.write_csv_file(home_dir + save_dir + features_count_dict_file_name, False, True, feature_count_list)

# high_prob_features_pos, high_prob_features_neg = funcs_worry.calc_probs(features_dict, feature_vects_neg, feature_vects_pos)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_pos', False, True, high_prob_features_pos)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_neg', False, True, high_prob_features_neg)

results = []
header = ['strip_thresh', 'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
          'tr_dim_size', 'ts_dim_size',
          'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg']

for strip_thresh in strip_thresholds:

    train_set_vects_pos = feature_vects_pos
    train_set_vects_neg = feature_vects_neg

    test_set_vects = feature_vects_test

    test_set_texts = tweet_texts_test

    print 'test set size:', len(test_set_vects)

    # We need to create a new dicts just for training set. Count all the feature in the test set.
    # This gives the test dict count. Subtract this from the original one to get the training dict.
    # features_count_dict_train = copy.deepcopy(features_count_dict)
    # all_test_set_vects = test_set_vects
    # for i in range(len(all_test_set_vects)):
    #     if (i % 1000) == 0:
    #         print 'creating training dictionary', i
    #     vect = all_test_set_vects[i]
    #     fact = norm_factors_test[i]
    #     for a, r in vect.iteritems():
    #         c_test = r * fact
    #         c_train_and_test = features_count_dict_train[a]
    #         diff = int(c_train_and_test - c_test)
    #         features_count_dict_train[a] = diff

    ################################################################################################################
    if use_even_test_sets:
        if len(test_set_vects_pos) > len(test_set_vects_neg):
            # zip feature_vectors and tweet_texts for sampling
            zipped = zip(test_set_vects_pos, test_set_texts_pos)
            # randomly select elements
            sampled_zipped = random.sample(zipped, len(test_set_vects_neg))
            # unzip sampled elements
            test_set_vects_pos, test_set_texts_pos = list(zip(*sampled_zipped))
            test_set_vects_pos = list(test_set_vects_pos)
            test_set_texts_pos = list(test_set_texts_pos)
        elif len(test_set_vects_neg) > len(test_set_vects_pos):
            # zip feature_vectors and tweet_texts for sampling
            zipped = zip(test_set_vects_neg, test_set_texts_neg)
            # randomly select elements
            sampled_zipped = random.sample(zipped, len(test_set_vects_pos))
            # unzip sampled elements
            test_set_vects_neg, test_set_texts_neg = zip(*sampled_zipped)
            test_set_vects_neg = list(test_set_vects_neg)
            test_set_texts_neg = list(test_set_texts_neg)
        ################################################################################################################

    ################################################################################################################
    # train_set_dim = funcs_worry.get_dimension_size(train_set_vects_pos + train_set_vects_neg)
    # test_set_dim = funcs_worry.get_dimension_size(test_set_vects)
    # train_set_unique_features = [[features_dict_reverse[dim]] for dim in train_set_dim]
    # my_util.write_csv_file(home_dir + save_dir + 'train_features_before_stripping_' + str(strip_thresh), False, True, train_set_unique_features)
    # test_set_unique_features = [[features_dict_reverse[dim]] for dim in test_set_dim]
    # my_util.write_csv_file(home_dir + save_dir + 'test_set_unique_features' + str(strip_thresh), False, True, test_set_unique_features)
    # ################################################################################################################
    #
    #
    # if strip_thresh > 0:
    #     train_set_vects_pos = \
    #         funcs_worry.strip_less_than(train_set_vects_pos, features_count_dict_train, strip_thresh)
    #     train_set_vects_neg = \
    #         funcs_worry.strip_less_than(train_set_vects_neg, features_count_dict_train, strip_thresh)
    #
    # # train sets and test sets are list of dictionaries (called vectors).
    # train_set_dim = funcs_worry.get_dimension_size(train_set_vects_pos + train_set_vects_neg)
    #
    # ################################################################################################################
    # train_set_unique_features = [[features_dict_reverse[dim]] for dim in train_set_dim]
    # my_util.write_csv_file(home_dir + save_dir + 'train_features_bigger_than_' + str(strip_thresh), False, True, train_set_unique_features)
    # ################################################################################################################

    x_train = train_set_vects_pos + train_set_vects_neg
    y_train = [labels['pos']] * len(train_set_vects_pos) + [labels['neg']] * len(train_set_vects_neg)

    x_test = test_set_vects
    y_test = [0] * len(test_set_vects)

    training_sizes = {'pos':len(train_set_vects_pos),'neg':len(train_set_vects_neg)}
    svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, labels, training_sizes)
    #svm_params = '-s 0 -t 0 -c 1 -w1 0.25 -w-1 1.25'
    p_label, p_acc, p_val = funcs_worry.train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)
    print 'calculating validation statistics ...'
    # prediction_result, accuracy, precisions, recalls = \
    #     funcs_worry.calc_prediction_stats_2(y_test, test_set_texts, p_label, labels)

    prediction_result=[]
    for i in range(len(x_test)):
        prediction_result.append([test_set_texts[i], p_label[i], p_val[i][0]])

    my_util.write_csv_file(home_dir+save_dir+result_file_name + '_ST'+str(strip_thresh), False, True, prediction_result)

    # results.append(
    #     [strip_thresh,
    #      len(train_set_vects_pos), len(train_set_vects_neg), len(test_set_vects_pos), len(test_set_vects_neg),
    #      len(train_set_dim), len(test_set_dim), accuracy, precisions['pos'], precisions['neg'], recalls['pos'], recalls['neg']]
    # )

# results = sorted(results, key=itemgetter(header.index('accuracy')))
# results.reverse()

# means = ['mean']
# stdevs = ['stdev']
# for column in range(header.index('tr_size_pos'), len(header)):
#     data = [row[column] for row in results]
#     mean, stdev = math_extra.calc_mean_stdev(data)
#     means = means + [round(mean, 2)]
#     stdevs = stdevs + [round(stdev, 2)]
#
# #append an empty row
# results.append([''] * len(header))
# # add two rows: means and stdev
# results = results + [means] + [stdevs]
# # append the header at the beginning
# results = [header] + results
#
# my_util.write_csv_file(home_dir + save_dir + table_file_name, False, True, results)
__author__ = 'behzadbehzadan'

"""
In this version, a binary classification applies on 'worried' (positive) and 'not worried' (negative) training sets.
The test set has data related to the hand picked data.
"""

import random
import math_extra
import my_util
import os
import funcs_worry
import copy
from operator import itemgetter
import svmutil

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_AMT/test_5/new_norm/'
collection_name_train = 'worried'
labels_features_file_name = 'labels_features'
tweet_texts_file_name = 'all_tweet_texts'
table_file_name = 'table'
# the train_lab is the list of key-phrases that are used to detect negative tweets from positive tweets.
# These phrases must not exist - and, hence, excluded - from the feature space.
#neg_train_labs = funcs_worry.get_negative_phrases(collection_name_train)
neg_train_labs = [collection_name_train]
pos_train_labs = [collection_name_train]

# All of the positive tweets for test set are extracted from the file 'heating'.
# Since the training set is from file 'worried', there is no default intersection between them. So we leave it empty.
pos_train_labs_test_set = []
neg_train_labs_test_set = []
########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
strip_thresholds = [0]
random.seed(7)
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# other labels are associated to any other types of tweets.
class_labels = { 'pos' : +1, 'neg' : -1}#, 'oth' : 0}
#m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
m = 1
#length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
n = 2
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

print 'reading training set source file ...'
tweets_train = my_util.read_csv_file(home_dir + source_dir + 'AmazonMTurk', False, True)
header = tweets_train[0]
worrieds = [t[header.index('Input.TEXT')] for t in tweets_train if t[header.index('Answer.Worry')] == 'worry']
not_worrieds = [t[header.index('Input.TEXT')] for t in tweets_train if t[header.index('Answer.Worry')] == 'notworry']
tell_not_worry = [t[header.index('Input.TEXT')] for t in tweets_train if t[header.index('Answer.Worry')] == 'tellnotworry']
nothing = [t[header.index('Input.TEXT')] for t in tweets_train if t[header.index('Answer.Worry')] == 'nothing']

def select_two_third(tweets):
    # make all letters lower-case --> this is essential when comparing strings and also when using quick_ratio
    tweets = [t.lower() for t in tweets]
    # remove extra spaces that may exist between words, by first splitting the words and then re-joining them.
    tweets = [' '.join(t.split()) for t in tweets]
    clustered_tweets = []#cluster duplicated/similar tweets together
    while len(tweets) > 0:
        t = tweets[0]
        #note: the string must have become lower-case before this stage.
        duplicates = [s for s in tweets if s == t]
        clustered_tweets.append([t, len(duplicates)])
        tweets = [x for x in tweets if x not in duplicates]

    unique_tweets = [t for t, l in clustered_tweets if l >= 2]
    return unique_tweets#, clustered_tweets

worrieds = select_two_third(worrieds)
not_worrieds = select_two_third(not_worrieds)
tell_not_worry = select_two_third(tell_not_worry)
nothing = select_two_third(nothing)
tr_set_size = len(worrieds) + len(not_worrieds) + len(tell_not_worry) + len(nothing)
print 'data size:', tr_set_size

print 'reading test set source file ...'
tweets_test_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_worry', False, True)
tweets_test_worry = [t[0] for t in tweets_test_worry]
tweets_test_worry = [t.lower() for t in tweets_test_worry]
tweets_test_worry = [' '.join(t.split()) for t in tweets_test_worry]
print 'data size:', len(tweets_test_worry)

tweets_test_not_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_not_worry', False, True)
tweets_test_not_worry = [t[0] for t in tweets_test_not_worry]
tweets_test_not_worry = [t.lower() for t in tweets_test_not_worry]
tweets_test_not_worry = [' '.join(t.split()) for t in tweets_test_not_worry]
print 'data size:', len(tweets_test_not_worry)

tweets_test_others = my_util.read_csv_file(home_dir + source_dir + 'test_set_others', False, True)
tweets_test_others = [t[0] for t in tweets_test_others]
tweets_test_others = [t.lower() for t in tweets_test_others]
tweets_test_others = [' '.join(t.split()) for t in tweets_test_others]
print 'data size:', len(tweets_test_others)


########################################################################################################################
########################################################################################################################
positives_test = tweets_test_worry
negatives_test = tweets_test_not_worry + tweets_test_others
########################################################################################################################
########################################################################################################################

print 'creating feature vectors...'

#the very first index is always 1.
if new_normalisation_flag:
    max_index = 0
else:
    max_index = 1

feature_vects_pos1, tweet_texts_pos1, max_index, norm_factors_pos1 = funcs_worry.get_sparse_feature_vector_worry(
    worrieds, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, ['worried', 'worry'], random)

feature_vects_pos2, tweet_texts_pos2, max_index, norm_factors_pos2 = funcs_worry.get_sparse_feature_vector_worry(
    tell_not_worry, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, ['worried', 'worry'], random)

feature_vects_neg1, tweet_texts_neg1, max_index, norm_factors_neg1 = funcs_worry.get_sparse_feature_vector_worry(
    not_worrieds, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, ['worried', 'worry'], random)

feature_vects_neg2, tweet_texts_neg2, max_index, norm_factors_neg2 = funcs_worry.get_sparse_feature_vector_worry(
    nothing, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

feature_vects_pos = feature_vects_pos1 + feature_vects_pos2
feature_vects_neg = feature_vects_neg1 + feature_vects_neg2

tweet_texts_pos = tweet_texts_pos1 + tweet_texts_pos2
tweet_texts_neg = tweet_texts_neg1 + tweet_texts_neg2

norm_factors_pos = norm_factors_pos1 + norm_factors_pos2
norm_factors_neg = norm_factors_neg1 + norm_factors_neg2

feature_vects_test_pos, tweet_texts_test_pos, max_index, norm_factors_test_pos = funcs_worry.get_sparse_feature_vector_worry(
    positives_test, features_dict, features_count_dict, max_index, m, n,  remove_stpwds_for_unigrams, new_normalisation_flag, pos_train_labs_test_set, random)

feature_vects_test_neg, tweet_texts_test_neg, max_index, norm_factors_test_neg = funcs_worry.get_sparse_feature_vector_worry(
    negatives_test, features_dict, features_count_dict, max_index, m, n,  remove_stpwds_for_unigrams, new_normalisation_flag, neg_train_labs_test_set, random)

n_distinct_features = len(features_dict)
print 'feature vectors created!', 'No of distinct features:', len(features_dict)

features_dict_reverse = funcs_worry.get_features_dict_reverse(features_dict)

#create a list from feature_dict in the form of [ ['feature', address], ...] to save in a csv file (tab deliminated)
feature_list = [list(z) for z in zip(features_dict.keys(), features_dict.values())]
my_util.write_csv_file(home_dir + save_dir + 'features_dict', True, True, feature_list)
#create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
feature_count_list = [list(z) for z in zip(features_count_dict.keys(), features_count_dict.values())]
my_util.write_csv_file(home_dir + save_dir + 'features_count_dict_all', False, True, feature_count_list)
#create a list from feature_dict in the form of [ [address, 'feature'], ...] to save in a csv file (tab deliminated)
feature_reverse_list = [list(z) for z in zip(features_dict_reverse.keys(), features_dict_reverse.values())]
my_util.write_csv_file(home_dir + save_dir + 'features_dict_reverse', True, True, feature_reverse_list)


all_statistics = []
header = ['strip_thresh', 'max_ngram', 'n_distinct_features', 'svm_params',
                  'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
                  'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg']

all_statistics.append(header)
for strip_thresh in strip_thresholds:

    statistics = []

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
    # if use_even_test_sets:
    #     if len(test_set_vects_pos) > len(test_set_vects_neg):
    #         # zip feature_vectors and tweet_texts for sampling
    #         zipped = zip(test_set_vects_pos, test_set_texts_pos)
    #         # randomly select elements
    #         sampled_zipped = random.sample(zipped, len(test_set_vects_neg))
    #         # unzip sampled elements
    #         test_set_vects_pos, test_set_texts_pos = list(zip(*sampled_zipped))
    #         test_set_vects_pos = list(test_set_vects_pos)
    #         test_set_texts_pos = list(test_set_texts_pos)
    #     elif len(test_set_vects_neg) > len(test_set_vects_pos):
    #         # zip feature_vectors and tweet_texts for sampling
    #         zipped = zip(test_set_vects_neg, test_set_texts_neg)
    #         # randomly select elements
    #         sampled_zipped = random.sample(zipped, len(test_set_vects_pos))
    #         # unzip sampled elements
    #         test_set_vects_neg, test_set_texts_neg = zip(*sampled_zipped)
    #         test_set_vects_neg = list(test_set_vects_neg)
    #         test_set_texts_neg = list(test_set_texts_neg)
    #     ################################################################################################################

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

    x_train = feature_vects_pos +  feature_vects_neg
    y_train = [class_labels['pos']] * len(feature_vects_pos) + [class_labels['neg']] * len(feature_vects_neg)

    x_test = feature_vects_test_pos + feature_vects_test_neg
    test_set_texts = tweet_texts_test_pos + tweet_texts_test_neg
    y_test = [class_labels['pos']] * len(tweet_texts_test_pos) + [class_labels['neg']] * len(tweet_texts_test_neg)

    training_sizes = {'pos': len(feature_vects_pos), 'neg': len(feature_vects_neg)}
    svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, class_labels, training_sizes)
    p_label, p_acc, p_val = funcs_worry.train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)

    prediction_result, accuracy, precisions, recalls = \
        funcs_worry.calc_prediction_stats_2(y_test, test_set_texts, p_label, p_val, class_labels)

    print 'accuracy', accuracy
    print 'precisions', precisions
    print 'recalls', recalls

    my_util.write_csv_file(home_dir + save_dir + 'result_' + '_ST' + str(strip_thresh), False, True, prediction_result)

    statistics.append(
    [strip_thresh, n, n_distinct_features, svm_params,
    len(feature_vects_pos), len(feature_vects_neg), len(feature_vects_test_pos),  len(feature_vects_test_neg),
    accuracy, precisions['pos'], precisions['neg'], recalls['pos'], recalls['neg']]
    )

    my_util.write_csv_file(home_dir + save_dir + 'statistics' + '_ST' + str(strip_thresh), False, True, statistics)


    statistics = sorted(statistics, key=itemgetter(header.index('accuracy')))
    statistics.reverse()
    means = [''] * header.index('svm_params') + ['mean']#shift the mean to the right, so that it comes under cross_val
    stdevs = [''] * header.index('svm_params') + ['stdev']
    for column in range(header.index('svm_params')+1, len(header)):
        data = [row[column] for row in statistics]
        mean, stdev = math_extra.calc_mean_stdev(data)
        means = means + [round(mean, 2)]
        stdevs = stdevs + [round(stdev, 2)]
    statistics = statistics + [means] + [stdevs]

    all_statistics.append([''] * len(header))#append an empty row
    all_statistics = all_statistics + statistics

my_util.write_csv_file(home_dir + save_dir + 'all_statistics', False, True, all_statistics)




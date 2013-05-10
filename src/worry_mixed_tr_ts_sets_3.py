__author__ = 'behzadbehzadan'

"""
training_set = 'worried' + 'MechTurk'
test_set = 'MechTurk'
cross vlidation applies
"""

import random
import math_extra
import my_util
import os
import funcs_worry
import copy
from operator import itemgetter

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_mixed_sets/worried_and_MTurk_vs_handPicked/test3/'
########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = False
strip_thresholds = [0]
random.seed(7)
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# other labels are associated to any other types of tweets.
labels = { 'pos' : +1, 'neg' : -1}#, 'oth' : 0}
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
cost = 1
# Set the nu parameter for the nu_SVC
# Note: if nu is not small enough, the following error message is shown: "specified nu is infeasible"
nu = 0.05
# Assign different costs to balance unbalanced (different sized) training sets.
balance_sets = True
use_even_test_sets = False
########################################################################################################################

#the very first index is always 1.
if new_normalisation_flag:
    max_index = 0
else:
    max_index = 1


# {'feature' : feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
# when using svm, an address represents a dimension on the input-space. So it is important to keep the address
# consistent for positive and negative sets and also for training and test sets.
features_dict = dict()
# {feature_address (dimension no) : 'feature'} --> used for debugging to visualise features
features_dict_reverse = dict()
#{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
features_count_dict = dict()
# whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

features_dict_training_pos = dict()
features_count_dict_training_pos = dict()

features_dict_training = dict()
features_count_dict_training = dict()
###################################################### functions #######################################################
def read_worried_data():

    print "reading 'worried' source file ..."
    worried_source_data_noRT = my_util.read_csv_file(home_dir + source_dir + 'source_worried_noDup', False, True)
    #worried_source_data = worried_source_data_noRT[:100]
    worried_source_data = [t[0] for t in worried_source_data_noRT]
    worried_source_data = [t.lower() for t in worried_source_data]
    # remove extra spaces that may exist between words. Is good for when finding not worried tweets, as we look
    # for certain strings like 'aint worried' (don't care about one or double space between 'aint' & 'worried')
    worried_source_data = [' '.join(t.split()) for t in worried_source_data]
    print 'training set size:', len(worried_source_data)

    return worried_source_data

def read_amazon_mech_turk_data():

    print 'reading Amazon MTurk source files ...'
    tweets_MTurk = my_util.read_csv_file(home_dir + source_dir + 'AmazonMTurk', False, True)
    header = tweets_MTurk[0]
    worrieds_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'worry']
    not_worrieds_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'notworry']
    tell_not_worry_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'tellnotworry']
    nothing_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'nothing']
    # select those tweets that turkers agreed when annotating --> thresshol==2 means at least two annotators agreed.
    worrieds_MTurk = funcs_worry.filter_same_turked_tweets(worrieds_MTurk, 2)
    not_worrieds_MTurk = funcs_worry.filter_same_turked_tweets(not_worrieds_MTurk, 2)
    tell_not_worry_MTurk = funcs_worry.filter_same_turked_tweets(tell_not_worry_MTurk, 2)
    nothing_MTurk = funcs_worry.filter_same_turked_tweets(nothing_MTurk, 2)
    print 'Amazon MTurk data size:', len(worrieds_MTurk) + len(not_worrieds_MTurk) + len(tell_not_worry_MTurk) + len(nothing_MTurk)

    return worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk

# def read_hand_picked_data():
#
#     print 'reading test set source files ...'
#     tweets_test_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_worry', False, True)
#     tweets_test_worry = [t[0] for t in tweets_test_worry]
#     tweets_test_worry = [t.lower() for t in tweets_test_worry]
#     tweets_test_worry = [' '.join(t.split()) for t in tweets_test_worry]
#     print 'test_set_worry size:', len(tweets_test_worry)
#
#     tweets_test_not_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_not_worry', False, True)
#     tweets_test_not_worry = [t[0] for t in tweets_test_not_worry]
#     tweets_test_not_worry = [t.lower() for t in tweets_test_not_worry]
#     tweets_test_not_worry = [' '.join(t.split()) for t in tweets_test_not_worry]
#     print 'test_set_not_worry size:', len(tweets_test_not_worry)
#
#     tweets_test_others = my_util.read_csv_file(home_dir + source_dir + 'test_set_others', False, True)
#     tweets_test_others = [t[0] for t in tweets_test_others]
#     tweets_test_others = [t.lower() for t in tweets_test_others]
#     tweets_test_others = [' '.join(t.split()) for t in tweets_test_others]
#     print 'test_set_others size:', len(tweets_test_others)
#
#     return tweets_test_worry, tweets_test_not_worry, tweets_test_others

def create_pos_neg_sets_for_worried_data(worried_source_data):

    #reads all tweets (no retweets) to create pos/neg sets
    print "creating pos/neg sets for 'worried' data ..."

    # the train_lab is the list of key-phrases that are used to detect negative tweets from positive tweets.
    # These phrases must be excluded from the feature space.
    neg_phrases = my_util.read_csv_file(home_dir + source_dir + 'neg_phrases_sorted', False, True)
    tell_not_worry_phrases = my_util.read_csv_file(home_dir + source_dir + 'tell_not_worry_phrases', False, True)
    # remove those negative phrases that has positive ('worried') implication!
    neg_train_labs = [x for x in neg_phrases if x not in tell_not_worry_phrases]
    # select those negative phrase that have been used more than a certain number of times.
    neg_train_labs = [p for p, n in neg_train_labs if int(n) > 0]
    neg_train_labs = [t.lower() for t in neg_train_labs]
    neg_train_labs = [' '.join(t.split()) for t in neg_train_labs]

    positives, negatives, n_containing_tweets = funcs_worry.find_pos_neg_tweets(neg_train_labs, worried_source_data)

    # Most of the mixed messages like 'worried about X, but not worried about Y' must be in the negative set.
    # This is because we have looked for every tweet containing negative training labels and put them in the negative set.
    negatives_with_double_keyword = funcs_worry.find_double_keywords('worried', negatives)
    print 'negatives_with_double_keyword:', len(negatives_with_double_keyword)
    negatives = funcs_worry.remove_intersection_from_the_list(negatives, negatives_with_double_keyword)
    print 'finished creating pos/neg sets!'

    return positives, negatives, n_containing_tweets, neg_train_labs


def save_stuff():

    #create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
    feature_count_list_training_pos = [list(z) for z in zip(features_count_dict_training_worried.keys(), features_count_dict_training_worried.values())]
    my_util.write_csv_file(home_dir + save_dir + 'features_count_dict_training_worried', False, True, feature_count_list_training_pos)
    funcs_worry.write_features_and_freqs_to_csv(features_dict_training_pos, feature_count_list_training_pos, -1, home_dir + save_dir + 'features_freq_training_worried')

    #create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
    feature_count_list_training = [list(z) for z in zip(features_count_dict_training.keys(), features_count_dict_training.values())]
    my_util.write_csv_file(home_dir + save_dir + 'features_count_dict_training', False, True, feature_count_list_training)
    funcs_worry.write_features_and_freqs_to_csv(features_dict, features_count_dict, -1, home_dir + save_dir + 'features_freq_training')

    features_dict_reverse = funcs_worry.get_features_dict_reverse(features_dict)
    #create a list from feature_dict in the form of [ ['feature', address], ...] to save in a csv file (tab deliminated)
    feature_list = [list(z) for z in zip(features_dict.keys(), features_dict.values())]
    my_util.write_csv_file(home_dir + save_dir + 'features_dict', True, True, feature_list)
    #create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
    feature_count_list = [list(z) for z in zip(features_count_dict.keys(), features_count_dict.values())]
    my_util.write_csv_file(home_dir + save_dir + 'features_count_dict', False, True, feature_count_list)
    #create a list from feature_dict in the form of [ [address, 'feature'], ...] to save in a csv file (tab deliminated)
    feature_reverse_list = [list(z) for z in zip(features_dict_reverse.keys(), features_dict_reverse.values())]
    my_util.write_csv_file(home_dir + save_dir + 'features_dict_reverse', True, True, feature_reverse_list)


def get_worried_pos_neg_sets(worried_source_data):

    positives, negatives, n_containing_tweets, neg_train_labs = create_pos_neg_sets_for_worried_data(worried_source_data)

    pos_train_labs = ['worried']

    # # find all positive tweets that contain 'worry' phrases:
    # positives_worry = [t for t in positives if 'worry' in t]
    # positives_without_worry = [x for x in positives if x not in positives_worry]
    # print "number of positive 'worried' tweets containing 'worry':", len(positives_worry)
    # positives = positives_worry + random.sample(positives_without_worry, 10000-len(positives_worry))
    # pos_train_labs.extend(['worry'])
    # negatives = random.sample(negatives, 10000)

    positives = random.sample(positives, 10000)
    negatives = random.sample(negatives, 10000)
    #positives = random.sample(positives, len(negatives))

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(home_dir + save_dir + 'tr_worried_neg', False, True, [[t] for t in negatives])
    my_util.write_csv_file(home_dir + save_dir + 'tr_worried_pos', False, True, [[t] for t in positives])
    my_util.write_csv_file(home_dir + save_dir + 'n_neg_phrases_used', False, True, n_containing_tweets)

    return positives, negatives, pos_train_labs, neg_train_labs

def get_MTurk_pos_neg_sets(worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk):

    positives = worrieds_MTurk + tell_not_worry_MTurk
    negatives = not_worrieds_MTurk + nothing_MTurk

    #save (write) pos/neg tweets in a file!
    my_util.write_csv_file(home_dir + save_dir + 'tr_MTurk_neg', False, True, [[t] for t in negatives])
    my_util.write_csv_file(home_dir + save_dir + 'tr_MTurk_pos', False, True, [[t] for t in positives])

    pos_train_labs = []
    neg_train_labs = []

    return positives, negatives, pos_train_labs, neg_train_labs


###################################################### read source data ################################################
worried_source_data = read_worried_data()
worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk = read_amazon_mech_turk_data()
########################################################################################################################

###################################################### get training data ###############################################
positives_worried, negatives_worried, pos_train_labs_worried, neg_train_labs_worried = \
    get_worried_pos_neg_sets(worried_source_data)

positives_MTurk, negatives_MTurk, pos_train_labs_MTurk, neg_train_labs_MTurk = \
    get_MTurk_pos_neg_sets(worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk)

#save (write) pos/neg tweets in a file!
my_util.write_csv_file(home_dir + save_dir + 'tr_worried_and_MTurk_neg', False, True, [[t] for t in positives_worried + positives_MTurk])
my_util.write_csv_file(home_dir + save_dir + 'tr_worried_and_MTurk_pos', False, True, [[t] for t in negatives_worried + negatives_MTurk])
########################################################################################################################

###################################################### create feature vectors ##########################################

print 'creating feature vectors for training set...'

feature_vects_pos_worried, tweet_texts_pos_worried, max_index, norm_factors_pos_worried = funcs_worry.get_sparse_feature_vector_worry(
    positives_worried, features_dict, features_count_dict, max_index, min_ngram, max_ngram, remove_stpwds_for_unigrams, new_normalisation_flag, pos_train_labs_worried, random)

feature_vects_neg_worried, tweet_texts_neg_worried, max_index, norm_factors_neg_worried = funcs_worry.get_sparse_feature_vector_worry(
    negatives_worried, features_dict, features_count_dict, max_index, min_ngram, max_ngram, remove_stpwds_for_unigrams, new_normalisation_flag, neg_train_labs_worried, random)

features_dict_training_worried = copy.deepcopy(features_dict)
features_count_dict_training_worried = copy.deepcopy(features_count_dict)

feature_vects_pos_MTurk, tweet_texts_pos_MTurk, max_index, norm_factors_pos_MTurk = funcs_worry.get_sparse_feature_vector_worry(
    positives_MTurk, features_dict, features_count_dict, max_index, min_ngram, max_ngram, remove_stpwds_for_unigrams, new_normalisation_flag, pos_train_labs_MTurk, random)

feature_vects_neg_MTurk, tweet_texts_neg_MTurk, max_index, norm_factors_neg_MTurk = funcs_worry.get_sparse_feature_vector_worry(
    negatives_MTurk, features_dict, features_count_dict, max_index, min_ngram, max_ngram, remove_stpwds_for_unigrams, new_normalisation_flag, neg_train_labs_MTurk, random)

feature_vects_pos = feature_vects_pos_worried + feature_vects_pos_MTurk
tweet_texts_pos = tweet_texts_pos_worried + tweet_texts_pos_MTurk
norm_factors_pos = norm_factors_pos_worried + norm_factors_pos_MTurk

feature_vects_neg = feature_vects_neg_worried + feature_vects_neg_MTurk
tweet_texts_neg = tweet_texts_neg_worried + tweet_texts_neg_MTurk
norm_factors_neg = norm_factors_neg_worried + norm_factors_neg_MTurk

########################################################################################################################

############################################################# save data ################################################
save_stuff()
########################################################################################################################
n_fold_cross_val=10

test_set_size_pos = len(feature_vects_pos_MTurk) / n_fold_cross_val
test_set_size_neg = len(feature_vects_neg_MTurk) / n_fold_cross_val
all_statistics = []
header = ['strip_thresh', 'cross_val', 'max_ngram', 'n_distinct_features', 'svm_params',
                  'tr_size_pos', 'tr_size_neg', 'ts_size_pos', 'ts_size_neg',
                  'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg']

all_statistics.append(header)
for strip_thresh in strip_thresholds:

    statistics = []
    for c in range(0, n_fold_cross_val):

        strt_pos = c * test_set_size_pos
        strt_neg = c * test_set_size_neg

        print str(n_fold_cross_val) + '-fold cross validation in progress...\n'
        print 'iteration', c + 1, '\n'

        if c < n_fold_cross_val - 1:
            end_pos = (c + 1) * test_set_size_pos
            end_neg = (c + 1) * test_set_size_neg

        else:
            #this is the last part
            end_pos = len(feature_vects_pos)
            end_neg = len(feature_vects_neg)

        test_set_vects_pos = feature_vects_pos[strt_pos: end_pos]
        test_set_vects_neg = feature_vects_neg[strt_neg: end_neg]

        test_set_texts_pos = tweet_texts_pos[strt_pos: end_pos]
        test_set_texts_neg = tweet_texts_neg[strt_neg: end_neg]

        print 'test set size positive:', len(test_set_vects_pos)
        print 'test set size negative:', len(test_set_vects_neg)

        # note that the size of the train-set is not necessarily equal to the size of the whole data set minus the size
        # of the test set. This is because still some duplicated tweets (re-tweets) may exist in the data set!
        train_set_vects_pos = [x for x in feature_vects_pos if x not in test_set_vects_pos]
        train_set_vects_neg = [x for x in feature_vects_neg if x not in test_set_vects_neg]

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



        x_train = train_set_vects_pos + train_set_vects_neg
        y_train = [class_labels['pos']] * len(train_set_vects_pos) + [class_labels['neg']] * len(train_set_vects_neg)

        x_test = test_set_vects_pos + test_set_vects_neg
        test_set_texts = test_set_texts_pos + test_set_texts_neg
        y_test = [class_labels['pos']] * len(test_set_vects_pos) + [class_labels['neg']] * len(test_set_vects_neg)

        training_sizes = {'pos': len(train_set_vects_pos), 'neg': len(train_set_vects_neg)}
        svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, class_labels, training_sizes)
        p_label, p_acc, p_val = funcs_worry.train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)

        prediction_result, accuracy, precisions, recalls = \
            funcs_worry.calc_prediction_stats_2(y_test, test_set_texts, p_label, p_val, class_labels)

        my_util.write_csv_file(home_dir + save_dir + 'result_' + str(c + 1) + '_ST' + str(strip_thresh), False, True, prediction_result)

        statistics.append(
        [strip_thresh, c + 1, n, n_distinct_features, svm_params,
        len(train_set_vects_pos), len(train_set_vects_neg), len(test_set_vects_pos),  len(test_set_vects_neg),
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


# ############################################################## svm cross validation test
# ##############################################################
# ##############################################################
# x_train = feature_vects_pos + feature_vects_neg
# y_train = [class_labels['pos']] * len(feature_vects_pos) + [class_labels['neg']] * len(feature_vects_neg)
# training_sizes = {'pos': len(feature_vects_pos), 'neg': len(feature_vects_neg)}
# svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, class_labels, training_sizes) + ' -v 10'
# prob = svmutil.svm_problem(y_train, x_train)
# param = svmutil.svm_parameter(svm_params)
# m = svmutil.svm_train(prob, param)
# ##############################################################
# ##############################################################
# ##############################################################


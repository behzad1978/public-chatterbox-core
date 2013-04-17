__author__ = 'behzadbehzadan'

"""
This version is a 3 class classification.
"""

import random
import math_extra
import my_util
import os
import funcs_worry
import copy
from operator import itemgetter
import math

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/Worry/Sources/'
saving_dir = '/Chatterbox_UCL_Advance/Worry/test/'
collection_names = ['worried']#, 'relaxed']

labels_features_file_name = 'labels_features'
tweet_texts_file_name = 'all_tweet_texts'
norm_factor_file_name = 'norm_factor'
result_file_name = 'Results/cross_val'
features_dict_file_name = 'features_dict'
features_count_dict_file_name = 'features_count_dict'
table_file_name = 'Results/table'

########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
read_data_from_file = False
# n must be greater or equal to 1 --> when equal to 1, there is no cross validation
n_fold_cross_val = 2
strip_thresholds = [0]#[0, 1, 2, 3, 4, 5, 10, 15, 20]
random.seed(7)
#m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
m = 1
#length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
n = 3
###################################################### libsvm settings #################################################
# The nu_CSV does not take the C parameter (i.e. the cost function). Hence, there is no weight balancing option.
svm_type = 'C_SVC' #'nu_SVC'
# Set the kernel. linear --> 0; polynomial --> 1; radial basis --> 2; sigmoid --> 3; precomputed kernel --> 4
kernel_type = 0
# Set the cost parameter for the C_CSV
cost = 10
# Set the nu parameter for the nu_SVC
# Note: if nu is not small enough, the following error message is shown: "specified nu is infeasible"
nu = 0.05
# Assign different costs to balance unbalanced (different sized) training sets.
balance_sets = True
########################################################################################################################
# dict in the form of {'collection_name' : +1/-1/0}
labels = {}
traiining_flag = {}
test_flag = {}
# dict in the form of {'collection_name' : tweets}
input_tweets = {}
tweets = {}
train_labs = {}
feature_vects = {}
tweet_texts = {}
norm_factors = {}

# class_labels:
pos = 1
neg = -1
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# put the label_value to None to exclude that set taking part in classification (neither training, nor test) .
for collection_name in collection_names:
    if collection_name == 'worried':
        labels['pos'] = {collection_name: eval('pos')}
        labels['neg'] = {collection_name: eval('neg')}
    elif collection_name == 'concerned':
        labels['pos'] = {collection_name: None}
        labels['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        labels['pos'] = {collection_name: None}
        labels['neg'] = {collection_name: None}

for collection_name in collection_names:
    if collection_name == 'worried':
        traiining_flag['pos'] = {collection_name: True}
        traiining_flag['neg'] = {collection_name: True}
    elif collection_name == 'concerned':
        traiining_flag['pos'] = {collection_name: None}
        traiining_flag['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        traiining_flag['pos'] = {collection_name: None}
        traiining_flag['neg'] = {collection_name: None}

for collection_name in collection_names:
    if collection_name == 'worried':
        test_flag['pos'] = {collection_name: True}
        test_flag['neg'] = {collection_name: True}
    elif collection_name == 'concerned':
        test_flag['pos'] = {collection_name: None}
        test_flag['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        test_flag['pos'] = {collection_name: None}
        test_flag['neg'] = {collection_name: None}

def initialise_dict(d):
    for l in labels.keys():
        d[l] = {}

def get_train_labs(collection_names):
    all_train_labs = []
    for collection_name in collection_names:
        extra = []
        neg_train_lab = funcs_worry.get_negative_phrases(collection_name)
        # our collection_name usually ends with 'ed' (like 'worried', 'concerned').
        # We remove 'ed' to get 'worri' (for worries) and 'concern'
        truncated = collection_name[: len(collection_name) - 2]
        if collection_name == 'worried':
            extra = ['worr']
        train_labs[collection_name] = neg_train_lab + [collection_name] + [truncated] + extra
        all_train_labs = all_train_labs + neg_train_lab + [collection_name] + [truncated] + extra
    return train_labs, all_train_labs


# {'feature' : feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
# when using svm, an address represents a dimension on the input-space. So it is important to keep the address
# consistent for positive and negative sets and also for training and test sets.
features_dict = dict()
# {feature_address (dimension no) : 'feature'} --> used for debugging to visualise features
features_dict_reverse = dict()
#{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
features_count_dict = dict()
# whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

# if feature_vectors have been previously created, then we just read them from formerly created saved file.
if read_data_from_file:
    # read labels and features.
    # each row is in the following format: label \t address1:feature_value1 \t address2:feature_value2 \t ...
    labels_features = my_util.read_csv_file(home_dir + source_dir + labels_features_file_name, True, True)
    # read tweet_texts corresponding to the feature_vectors
    tweet_texts = my_util.read_csv_file(home_dir + source_dir + tweet_texts_file_name, False, True)
    # when creating feature_vectors, we count tokens resulted from the tweet_text. We then divide counts by a common
    # factor (normalisation factor) which is either the Nr. of tokens or the Nr. of features.
    # Normalisation factors are needed to create a separate feature_count_dict for the training set the , which is used
    # for stripping less frequent features to reduce the dimensionality of the feature space.
    norm_factors = my_util.read_csv_file(home_dir + source_dir + norm_factor_file_name, False, True)
    if len(labels_features) == len(tweet_texts):
        # iterate through each row i
        for i in range(len(labels_features)):
            l_f = labels_features[i]
            text = tweet_texts[i][0]
            # the first element of the row is the label
            l = int(l_f[0])
            # the rest of the elements are the feature_vector
            f = l_f[1:]
            # read the normalisation factor corresponding to each feature_vector
            n = int(norm_factors[i][0])
            # each feature is a string in the form of address:feature_value --> separate address from feature: [a,v]
            f = [a_v.split(':') for a_v in f]
            # create a dictionary (i.e. the feature_vector) in the form of { address : value }
            vector = { int(a_v[0]): float(a_v[1]) for a_v in f }
            if l == labels['pos']:
                labels_pos.append(l)
                feature_vects_pos.append(vector)
                tweet_texts_pos.append(text)
                norm_factors_pos.append(n)
            elif l == labels['neg']:
                labels_neg.append(l)
                feature_vects_neg.append(vector)
                tweet_texts_neg.append(text)
                norm_factors_neg.append(n)
            # elif l == labels['oth']:
            #     labels_oth.append(l)
            #     feature_vects_oth.append(vector)
            #     tweet_texts_oth.append(text)
            #     norm_factors_oth.append(n)

    all_feature_vects = feature_vects_pos + feature_vects_neg# + feature_vects_oth
    all_labels = labels_pos + labels_neg #+labels_oth
    all_texts = tweet_texts_pos + tweet_texts_neg #+tweet_texts_oth

    #read features_dict file --> csv (tab deliminated)
    feature_list = my_util.read_csv_file(home_dir + source_dir + features_dict_file_name, True, True)
    features_dict = { f : int(a) for f, a in feature_list }
    #read feature_count_dict file --> csv (coma separated)
    feature_count_list = my_util.read_csv_file(home_dir + source_dir + features_count_dict_file_name, False, True)
    features_count_dict = { int(a) : int(c) for a, c in feature_count_list }

else:

    if remove_retweets:
        # read input data.
        for collection_name in collection_names:
            # read unique (not re-tweeted) tweets:
            data = my_util.read_csv_file(home_dir + source_dir + 'source_' + collection_name + '_noDup', False, True)
            data = [t[0] for t in data]
            data = [t.lower() for t in data]
            # remove extra spaces that may exist between words. Is good for when finding not worried tweets, as we look
            # for certain strings like 'aint worried' (don't care about one or double space between 'aint' & 'worried')
            data = [' '.join(t.split()) for t in data]
            input_tweets[collection_name] = data[:100]
    else:
        for collection_name in collection_names:
            #read the source file --> [[text1], [text2], [test3], ...]
            data = my_util.read_csv_file(home_dir + source_dir + 'source_' + collection_name , False, True)
            #create list of texts --> [text1, text2, text3, ...]
            data = [t[0] for t in data]
            data = funcs_worry.remove_retweets(data, use_qr_to_remove_dups)
            # write unique tweets data to file:
            my_util.write_csv_file(home_dir + source_dir + 'source_' + '_noDup', False, True, [[t] for t in data])
            input_tweets[collection_name] = data

    # divide tweets into pos and neg sets
    initialise_dict(tweets)
    for collection_name, data in input_tweets.iteritems():
        positive_tweets, negative_tweets = funcs_worry.find_pos_neg_tweets(collection_name, data)
        # add new collection data
        tweets['pos'].update({collection_name : positive_tweets})
        tweets['neg'].update({collection_name : negative_tweets})
        # write data into csv file:
        my_util.write_csv_file(home_dir + source_dir + collection_name, False, True, [[t] for t in positive_tweets])
        my_util.write_csv_file(home_dir + source_dir + 'not_' + collection_name, False, True, [[t] for t in negative_tweets])

    print 'creating feature vectors...'

    #the very first index is always 1.
    if new_normalisation_flag:
        max_index = 0
    else:
        max_index = 1

    train_labs, all_train_labs = get_train_labs(collection_names)

    # create feature_vectors
    initialise_dict(feature_vects)
    initialise_dict(tweet_texts)
    initialise_dict(norm_factors)
    for label, collectionName_labelValue in labels.iteritems():
        for collection_name, label_value in collectionName_labelValue.iteritems():
            if label_value <> None:
                data = tweets[label][collection_name]
                feature_vects[label][collection_name], tweet_texts[label][collection_name], max_index, norm_factors[label][collection_name] = \
                    funcs_worry.get_sparse_feature_vector_worry(data, features_dict, features_count_dict, max_index, m,
                                                    n, remove_stpwds_for_unigrams, new_normalisation_flag, train_labs)

    print 'feature vectors created!', 'No of distinct features:', len(features_dict)

    #The following lines save features and labels in files to be read when we want to read them from file and not
    # creating again.
    all_feature_vects, all_labels, all_texts, all_norm_factors = [], [], [], []
    for label, collectionName_labelValue in labels.iteritems():
        for collection_name, label_value in collectionName_labelValue.iteritems():
            if label_value <> None:
                all_feature_vects = all_feature_vects + feature_vects[label][collection_name]
                all_labels = all_labels + [label_value] * len(feature_vects[label][collection_name])
                all_texts = all_texts + tweet_texts[label][collection_name]
                all_norm_factors = all_norm_factors + norm_factors[label][collection_name]

    funcs_worry.write_labels_features_in_libsvm_form(all_labels, all_feature_vects, home_dir + saving_dir + labels_features_file_name)
    my_util.write_csv_file(home_dir + saving_dir + tweet_texts_file_name, False, True, [[t] for t in all_texts])
    my_util.write_csv_file(home_dir + saving_dir + norm_factor_file_name, False, True, [[n] for n in all_norm_factors])
    #create a list from feature_dict in the form of [ ['feature', address], ...] to save in a csv file (tab deliminated)
    feature_list = [list(z) for z in zip(features_dict.keys(), features_dict.values())]
    my_util.write_csv_file(home_dir + source_dir + features_dict_file_name, True, True, feature_list)
    #create a list from feature_count_dict in the form of [ [address, freq], ...] to save in a csv file
    feature_count_list = [list(z) for z in zip(features_count_dict.keys(), features_count_dict.values())]
    my_util.write_csv_file(home_dir + source_dir + features_count_dict_file_name, False, True, feature_count_list)

    # visualising_thresh = 50
# funcs.write_features_and_freqs_to_csv(feature_vects_pos, features_count_dict_pos, visualising_thresh, source_dir + file_dir + collection_name + "_count_pos")
# funcs.write_features_and_freqs_to_csv(feature_vects_neg, features_count_dict_neg, visualising_thresh, source_dir + file_dir + collection_name + "_count_neg")

# def find_double_keywords(the_list):
#     worried_worried = []
#     for t in the_list:
#         t = t.split()
#         keyword = collection_name
#         count = len([x for x in t if x == keyword])
#         if count >= 2:
#             t = ' '.join(t)
#             worried_worried.append([t])
#     return worried_worried


# keyword_keyword_pos = find_double_keywords(tweet_texts_pos)
# keyword_keyword_neg = find_double_keywords(tweet_texts_neg)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'pos', False, True, keyword_keyword_pos)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'neg', False, True, keyword_keyword_neg)


# high_prob_features_pos, high_prob_features_neg = funcs_worry.calc_probs(features_dict, feature_vects_neg, feature_vects_pos)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_pos', False, True, high_prob_features_pos)
# my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_neg', False, True, high_prob_features_neg)


def shuffle_features_texts_normFactors(list1, list2, list3):
    if len(list1) == len(list2) == len(list3):
        zipped = zip(list1, list2, list3)
        random.shuffle(zipped)
        unzipped = zip(*zipped)
        list1 = list(unzipped[0])
        list2 = list(unzipped[1])
        list3 = list(unzipped[2])
    else:
        raise ValueError('the two list are not equal size!')
        #note that zip returns a new object and does not pass reference
    return list1, list2, list3

# aggregate data that are labeled the same from different collections.
# we will then divide these data for cross validation.
# alternatively, we could divide data for each collection and then aggregate data with the same label. But, because we
# shuffle data, that shouldn't make a big difference.
# same_label_vects, same_label_tweet_texts, same_label_norm_factors  = {}, {}, {}
# initialise_dict(same_label_vects)
# initialise_dict(same_label_tweet_texts)
# initialise_dict(same_label_norm_factors)
# for label, collectionName_labelValue in labels.iteritems():
#     for collection_name, label_value in collectionName_labelValue.iteritems():
#         if label_value <> None:
#             same_label_vects[label].update(feature_vects[label][collection_name])
#             same_label_tweet_texts[label].update(tweet_texts[label][collection_name])
#             same_label_norm_factors[label].update(norm_factors[label][collection_name])

# shuffle data, for each collections and assigned labels separately.
for label, collectionName_labelValue in labels.iteritems():
    for collection_name, label_value in collectionName_labelValue.iteritems():
        if label_value <> None:
            feature_vects[label][collection_name], tweet_texts[label][collection_name], \
            norm_factors[label][collection_name] = \
            shuffle_features_texts_normFactors(feature_vects[label][collection_name],
            tweet_texts[label][collection_name], norm_factors[label][collection_name])

results = []
n_fold_CV = '%d_fold_CV' % n_fold_cross_val
header = ['strip_thresh',
          n_fold_CV, 'tr_size_pos', 'tr_size_neg', 'tr_size_oth', 'ts_size_pos', 'ts_size_neg', 'ts_size_oth',
          'accuracy', 'precision_pos', 'precision_neg', 'precision_zero', 'recall_pos', 'recall_neg', 'recall_zero']

results.append(header)

for strip_thresh in strip_thresholds:

    strt_pos = 0
    end_pos = 0
    results_CrossVal = []
    for n in range(1, n_fold_cross_val+1):

        test_set_vects, test_set_tweet_texts, test_set_norm_factors = {}, {}, {}
        initialise_dict(test_set_vects)
        initialise_dict(test_set_tweet_texts)
        initialise_dict(test_set_norm_factors)
        train_set_vects, train_set_tweet_texts, train_set_norm_factors = {}, {}, {}
        initialise_dict(train_set_vects)
        initialise_dict(train_set_tweet_texts)
        initialise_dict(train_set_norm_factors)

        print str(n_fold_cross_val) + '-fold cross validation in progress...\n'
        print 'iteration', n, '\n'

        for label, collectionName_labelValue in traiining_flag.iteritems():
            for collection_name, flag in collectionName_labelValue.iteritems():
                # True flag means the data is used for training set.
                if flag:

                    if n_fold_cross_val == 1:
                        length = 0
                    else:
                        length = int(math.ceil(float(len(feature_vects[label][collection_name])) / n_fold_cross_val))
                    strt_pos = (n-1) * length
                    end_pos = n * length

                    # note that when n_fold_cross_val == 1, the test-sets are empty as start_pos == end_pos
                    test_set_vects[label][collection_name] = feature_vects[label][collection_name][strt_pos : end_pos]
                    test_set_tweet_texts[label][collection_name] = tweet_texts[label][collection_name][strt_pos:end_pos]
                    test_set_norm_factors[label][collection_name]=norm_factors[label][collection_name][strt_pos:end_pos]

                    print 'test set size ' + label + ' :', len(test_set_vects[label][collection_name])

                    # note: train-set size is not necessarily equal to the size of the whole data set minus the size of the test set.
                    # This is because still some duplicated tweets (re-tweets) may exist in the data set!
                    train_set_vects[label][collection_name] = [x for x in feature_vects[label][collection_name] if
                                                               x not in test_set_vects[label][collection_name]]
                    train_set_tweet_texts[label][collection_name] = [x for x in tweet_texts[label][collection_name] if
                                                                  x not in test_set_tweet_texts[label][collection_name]]
                    train_set_norm_factors[label][collection_name] = [x for x in norm_factors[label][collection_name] if
                                                                 x not in test_set_norm_factors[label][collection_name]]


        x_train, y_train = [], []
        training_sizes = {}
        x_test, y_test = [], []
        texts_test =[]
        normFactors_test=[]
        for label, collectionName_vects in train_set_vects.iteritems():
            for collection_name, vects in collectionName_vects.iteritems():

                x_train = x_train + train_set_vects[label][collection_name]
                y_train = y_train + [labels[label][collection_name]] * len(train_set_vects[label][collection_name])
                # store the size of all class sizes to create different weights for uneven training sets.
                try:
                    training_sizes[label] = training_sizes[label] + len(train_set_vects[label][collection_name])
                except KeyError:
                    training_sizes[label] = len(train_set_vects[label][collection_name])

                x_test = x_test + test_set_vects[label][collection_name]
                y_test = y_test + [labels[label][collection_name]] * len(test_set_vects[label][collection_name])
                texts_test = texts_test + test_set_tweet_texts[label][collection_name]
                normFactors_test = normFactors_test + test_set_norm_factors[label][collection_name]

        # add those elements that must be in the test set, but not in the training set.
        for label, collectionName_flag in test_flag.iteritems():
            for collection_name, flag in collectionName_flag.iteritems():
                if flag and (traiining_flag[label][collection_name] == False):
                    x_test = x_test + feature_vects[label][collection_name]
                    texts_test = texts_test + tweet_texts[label][collection_name]
                    normFactors_test = normFactors_test + norm_factors[label][collection_name]

        # we need to create two new dicts: one for training and one for test. count all the feature
        #in the test set. this gives the test dict count. subtract this from the original one to get the training dict.
        features_count_dict_train = copy.deepcopy(features_count_dict)
        for i, vect in enumerate(x_test):
            fact = normFactors_test[i]
            for a, r in vect.iteritems():
                c_test = r * fact
                c_train_and_test = features_count_dict_train[a]
                diff = int(c_train_and_test - c_test)
                features_count_dict_train[a] = diff

        if strip_thresh > 0:
            x_train = funcs_worry.strip_less_than(x_train, features_count_dict_train, strip_thresh)
            x_test = funcs_worry.strip_less_than(x_test, features_count_dict_train, strip_thresh)

        class_labels = { k : eval(k) for k in labels.keys() }
        svm_params = funcs_worry.get_params(svm_type, kernel_type, cost, nu, balance_sets, class_labels, training_sizes)
        p_label, p_acc, p_val = funcs_worry.train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)
        prediction_result, accuracy, precisions, recalls = \
            funcs_worry.calc_prediction_stats_2(y_test, test_set_texts, p_label, labels)

        my_util.write_csv_file(home_dir + saving_dir + result_file_name + str(n + 1) + '_' + str(accuracy) + '%', False,
                               True, prediction_result)

        results_CrossVal.append(
            [strip_thresh, n + 1,
             len(train_set_vects_pos), len(train_set_vects_neg), len(train_set_vects_pos_oth),
             len(test_set_vects_pos), len(test_set_vects_neg), len(test_set_vects_pos_oth),
             accuracy, precisions['pos'], precisions['neg'], precisions['oth'], recalls['pos'], recalls['neg'], recalls['oth']]
        )

    results_CrossVal = sorted(results_CrossVal, key=itemgetter(header.index('accuracy')))
    results_CrossVal.reverse()
    means = [''] * header.index(n_fold_CV) + ['mean']#shift the mean to the right, so that it comes under n_fold_CV

    stdevs = [''] * header.index(n_fold_CV) + ['stdev']
    for column in range(header.index('tr_size_pos'), len(header)):
        data = [row[column] for row in results_CrossVal]
        mean, stdev = math_extra.calc_mean_stdev(data)
        means = means + [round(mean, 2)]
        stdevs = stdevs + [round(stdev, 2)]
    results_CrossVal = results_CrossVal + [means] + [stdevs]

    results.append([''] * len(header))#append an empty row
    results = results + results_CrossVal

my_util.write_csv_file(home_dir + saving_dir + table_file_name + '_' + str(means[header.index('accuracy')]) + '%',
                       False, True, results)
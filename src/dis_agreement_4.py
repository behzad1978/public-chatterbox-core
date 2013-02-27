__author__ = 'behzadbehzadan'

import my_util
import os
import funcs
import random
import liblinearutil
import svmutil
import math_extra
from operator import itemgetter
import copy

############################################### frequent settings ######################################################
n_fold_cross_val = 5
random.seed(7)
separate_seed_reply_features_flag = True
train_with_liblinear = False
train_with_svm = True
duplicate_disagreed_set = False
#strip features less than 'strip_thresh'. If set to 0, nothing happens. If set to 1, single features are removed.
strip_thresholds = [0]#[0, 1, 2, 3, 4, 5, 10, 15, 20]
l_agr = +1
l_dis = -1
l_oth = 0
#svm params:
#gammas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5]#radial basis parameter
########################################################################################################################

###################################################### file names ######################################################
home_dir = os.path.expanduser('~')
source_dir = home_dir + '/Chatterbox_UCL_Advance/Agreement_Disagreement/convs_123/'

#csv files containing line by line tweets in the form of seed/reply, seed/reply, ..
agreed_file_name = 'agreed_pairs'
positive_answers_file_name = 'positive_answers'
supportive_pairs_file_name = 'supportives'
disagreed_file_name = 'disagreed_pairs'
negative_answers_file_name = 'negative_answers'
offensive_sarcastic_tweets = 'offensives'
followups_file_name = 'followups'

training_file_name = 'labels_and_features'#file saving training set (labels | feature-vectors) that could be read again.
features_file_name = 'features_and_freqs'#file saving features (ngrams) in one column and freq of occurrence in another.
seed_reply_file_name = 'seed_reply_texts'#file saving (seed | reply) texts side by side; to be read for visualisation.
test_result_name = 'test/test_set'#file saving the actual predicted labels and values on the test set.
tabel_result_name = 'test/tabel'
########################################################################################################################

def write_features_and_freqs_to_csv(features_dict, features_count_dict):
    feature_freq_list = []
    for f, a in features_dict.iteritems():
        c = features_count_dict[a]
        if c > 1:
            feature_freq_list.append([f, c])
        else:
            pass
    my_util.write_csv_file(source_dir + features_file_name, False, True, feature_freq_list)

def write_labels_and_features_to_csv(labels, features):
    """
    this function creates a tab deliminator csv file of the labels and features in the form of:
    label dimention_nr1:feature1 dimention_nr2:feature2 ...
    """
    #labels --> [+1,+1,+1...,+1] or [-1,-1,-1,...,-1]
    #features --> [dict1, dict2, dict3, ...]
    #dicts ---> {feature_address1 : feature_freq1, feature_address2 : feature_freq2, ...]
    final_list = []
    if len(labels) == len(features):
        for i in range(0, len(labels)):
            l = labels[i]
            feature_dict = features[i]
            feature_list = [str(k) + ":" + str(v) for k, v in feature_dict.iteritems()]
            the_list = [str(l)] + feature_list
            final_list.append(the_list)
    my_util.write_csv_file(source_dir + training_file_name, True, True, final_list)

def read_data():
    #original agreed tweets read line by line from a csv file. The first row is the header.
    agreed_pairs = my_util.read_csv_file(source_dir + agreed_file_name, True)
    positive_answers_pairs = my_util.read_csv_file(source_dir + positive_answers_file_name, True)
    supportive_pairs = my_util.read_csv_file(source_dir + supportive_pairs_file_name, True)
    #original disagreed tweets read line by line from a csv file. The first row is the header.
    disagreed_pairs = my_util.read_csv_file(source_dir + disagreed_file_name, True)
    negative_answers_pairs = my_util.read_csv_file(source_dir + negative_answers_file_name, True)
    offensive_pairs = my_util.read_csv_file(source_dir + offensive_sarcastic_tweets, True)
    other_pairs = my_util.read_csv_file(source_dir + followups_file_name, True)

    header = agreed_pairs[0]
    conv_id_indx = header.index('conv_id')#onv_id counts seed/reply pairs and is unique for both seed & reply.
    tweet_id_indx = header.index('tweet_id')#tweet_id is a unique no. associated to each single tweet.
    s_r_indx = header.index('S/R')#S/R stickers implying whether the tweet is seed or reply.
    text_indx = header.index('text')#the tweet texts.

    ########################################################################################################################
    agreed_pairs = agreed_pairs + positive_answers_pairs[1:] + supportive_pairs[1:]
    disagreed_pairs = disagreed_pairs + negative_answers_pairs[1:] + offensive_pairs[1:]
    ########################################################################################################################

    agreed_pairs = funcs.unicode_to_float(agreed_pairs, conv_id_indx, 1)
    agreed_pairs = funcs.unicode_to_float(agreed_pairs, tweet_id_indx, 1)
    disagreed_pairs = funcs.unicode_to_float(disagreed_pairs, conv_id_indx, 1)
    disagreed_pairs = funcs.unicode_to_float(disagreed_pairs, tweet_id_indx, 1)
    other_pairs = funcs.unicode_to_float(other_pairs, conv_id_indx, 1)
    other_pairs = funcs.unicode_to_float(other_pairs, tweet_id_indx, 1)

    agreed_pairs = funcs.code_0_1_for_seed_reply(agreed_pairs, s_r_indx, 1)
    disagreed_pairs = funcs.code_0_1_for_seed_reply(disagreed_pairs, s_r_indx, 1)
    other_pairs = funcs.code_0_1_for_seed_reply(other_pairs, s_r_indx, 1)

    is_data_missed, message, seed_reply_agreeds = funcs.make_seeds_replies(agreed_pairs, conv_id_indx, s_r_indx)
    is_data_missed, message, seed_reply_disagreeds = funcs.make_seeds_replies(disagreed_pairs, conv_id_indx, s_r_indx)
    is_data_missed, message, seed_reply_others = funcs.make_seeds_replies(other_pairs, conv_id_indx, s_r_indx)

    return seed_reply_agreeds, seed_reply_disagreeds, seed_reply_others, text_indx

seed_reply_agreeds, seed_reply_disagreeds, seed_reply_others, text_indx = read_data()

if (l_agr == +1 and l_dis == l_oth == -1):
    tabel_result_name = tabel_result_name + '_agr_rest'
    if duplicate_disagreed_set:
        seed_reply_others = random.sample(seed_reply_others, len(seed_reply_agreeds)/2)
    else:
        #we want: agreed set size = disagreed set size + others set size
        seed_reply_others = random.sample(seed_reply_others, len(seed_reply_agreeds)-len(seed_reply_disagreeds))
if (l_dis == -1 and l_agr == l_oth == +1):
    tabel_result_name = tabel_result_name + '_dis_rest'
    if duplicate_disagreed_set:
        seed_reply_agreeds = random.sample(seed_reply_agreeds, len(seed_reply_agreeds)/2)
        seed_reply_others = random.sample(seed_reply_others, len(seed_reply_agreeds)/2)
    else:
        seed_reply_agreeds=random.sample(seed_reply_agreeds, len(seed_reply_disagreeds)/2)
        seed_reply_others = random.sample(seed_reply_others, len(seed_reply_disagreeds)/2)
if (l_agr == +1 and l_dis == -1 and l_oth == 0):
    tabel_result_name = tabel_result_name + '_3-Class'

table_name_flag=True

random.shuffle(seed_reply_agreeds)
random.shuffle(seed_reply_disagreeds)
random.shuffle(seed_reply_others)

test_set_size_agreeds = len(seed_reply_agreeds) / n_fold_cross_val
test_set_size_disagreeds = len(seed_reply_disagreeds) / n_fold_cross_val
test_set_size_others = len(seed_reply_others) / n_fold_cross_val
results = []
n_fold_CV = '%d_fold_CV' %n_fold_cross_val
header = ['strip_thresh',
          n_fold_CV, 'tr_size_agr', 'tr_size_dis', 'tr_size_oth', 'ts_size_agr', 'ts_size_dis', 'ts_size_oth',
          'accuracy', 'precision_pos', 'precision_neg', 'precision_zero', 'recall_pos', 'recall_neg', 'recall_zero']

results.append(header)

#for g in gammas:
for strip_thresh in strip_thresholds:

    results_CrossVal=[]
    for n in range(0, n_fold_cross_val):

        strt_agr = n * test_set_size_agreeds
        strt_dis = n * test_set_size_disagreeds
        strt_other=n * test_set_size_others

        print str(n_fold_cross_val) + '-fold cross validation in progress...'
        print 'iteration', n, '\n'

        if n < n_fold_cross_val-1:
            end_agr = (n + 1) * test_set_size_agreeds
            end_dis = (n + 1) * test_set_size_disagreeds
            end_other = (n + 1) * test_set_size_others

        else:#last part
            end_agr = len(seed_reply_agreeds)
            end_dis = len(seed_reply_disagreeds)
            end_other = len(seed_reply_others)

        test_set_agr = seed_reply_agreeds[strt_agr : end_agr]
        test_set_dis = seed_reply_disagreeds[strt_dis : end_dis]
        test_set_others = seed_reply_others[strt_other : end_other]

        print 'test set size agreed:', len(test_set_agr)
        print 'test set size disagreed:', len(test_set_dis)
        print 'test set size others', len(test_set_others)

        train_set_agr = [x for x in seed_reply_agreeds if x not in test_set_agr]
        train_set_dis = [x for x in seed_reply_disagreeds if x not in test_set_dis]
        train_set_others = [x for x in seed_reply_others if x not in test_set_others]

        if duplicate_disagreed_set:
            size=None
            if (l_agr == +1 and l_dis == l_oth == -1):
                size = len(train_set_agr)/2 - len(train_set_dis)
            if (l_dis == -1 and l_agr == l_oth == +1):
                size = len(train_set_agr) + len(train_set_others) - len(train_set_dis)
            if (l_agr == +1 and l_dis == -1 and l_oth == 0):
                size = len(train_set_agr) - len(train_set_dis)

            train_set_dis = funcs.duplicate(train_set_dis, size, random)

        #{'feature':feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
        features_dict = dict()
        #{'feature_address (dimension no) : feature} --> used for debugging to visualise features
        features_dict_reverse = dict()
        #{feature_address (dimension no.) : freq_count} --> freq_count: absolute frequ of ngram occurring in token.
        features_count_dict = dict()
        #whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]
        max_index = 1

        feature_vects_agr_train, s_r_texts_agr_train, \
        feature_vects_dis_train, s_r_texts_dis_train, \
        feature_vects_others_train, s_r_texts_others_train, \
        max_index = \
            funcs.create_labels_and_features(
                features_dict, features_count_dict, max_index,
                train_set_agr, train_set_dis, train_set_others,
                text_indx,
                separate_seed_reply_features_flag)

        features_dict_train = copy.deepcopy(features_dict)
        features_count_dict_train = copy.deepcopy(features_count_dict)
        features_dict_reverse_train = funcs.get_features_dict_reverse(features_dict_train)

        if strip_thresh > 0:
            feature_vects_agr_train = funcs.strip_less_than(feature_vects_agr_train, features_count_dict_train, strip_thresh)
            feature_vects_dis_train = funcs.strip_less_than(feature_vects_dis_train, features_count_dict_train, strip_thresh)
            feature_vects_others_train = funcs.strip_less_than(feature_vects_others_train, features_count_dict_train, strip_thresh)

        feature_vects_agr_test, s_r_texts_agr_test, \
        feature_vects_dis_test, s_r_texts_dis_test, \
        feature_vects_others_test, s_r_texts_others_test, \
        max_index = \
            funcs.create_labels_and_features(
            features_dict, features_count_dict, max_index,
            test_set_agr, test_set_dis, test_set_others,
            text_indx,
            separate_seed_reply_features_flag)

        features_dict_reverse = funcs.get_features_dict_reverse(features_dict)

        features_dict_test, features_count_dict_test = funcs.make_feature_count_dict_test(
            features_count_dict, features_count_dict_train, features_dict_reverse)

        if strip_thresh > 0:
            feature_vects_agr_test = funcs.strip_less_than(feature_vects_agr_test, features_count_dict_test, strip_thresh)
            feature_vects_dis_test = funcs.strip_less_than(feature_vects_dis_test, features_count_dict_test, strip_thresh)
            feature_vects_others_test = funcs.strip_less_than(feature_vects_others_test, features_count_dict_test, strip_thresh)

        if train_with_liblinear and train_with_svm:
            raise Exception('both of the liblinear and libsvm flags are True! Just one must be True!')

        thresh_pos = 0
        thresh_neg = 0
        if train_with_liblinear:
            if table_name_flag == True:
                tabel_result_name = tabel_result_name + '_liblinear'
                table_name_flag = False
            if (l_agr==+1 and l_dis==l_oth==-1) or (l_dis==-1 and l_agr==l_oth==+1):

                x_train = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                y_train = [l_agr] * len(feature_vects_agr_train)+[l_dis] * len(feature_vects_dis_train) + [l_oth] * len(
                    feature_vects_others_train)

                x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [l_oth] * len(
                    feature_vects_others_test)

            if (l_agr==+1 and l_dis==-1 and l_oth==0):
                #this is a binary classification: no training on the other-set. However, the other-set is used for validation.

                #threshold values are used for liblinear classification judgment!
                thresh_pos = 0.19
                thresh_neg = -0.19
                feature_vects_others_train=[]
                x_train = feature_vects_agr_train + feature_vects_dis_train
                s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train
                y_train = [l_agr] * len(feature_vects_agr_train) + [l_dis] * len(feature_vects_dis_train)

                #since the size of other test is bigger than disagreement set, we shrink the other-set to become equal to disagreed set.
                feature_vects_others_test = feature_vects_others_test[0 : len(feature_vects_dis_test)]
                s_r_texts_others_test = s_r_texts_others_test[0 : len(feature_vects_dis_test)]
                x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [l_oth] * len(
                    feature_vects_others_test)

            print 'training in progress with liblinear ...'
            prob = liblinearutil.problem(y_train, x_train)
            param = liblinearutil.parameter('-c 1 -B 1')
            m = liblinearutil.train(prob, param)
            #p_labels --> classification labels predicted by the system.
            #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
            #p_val --> classification values predicted by the system.
            p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test, m)

        elif train_with_svm:
            if table_name_flag:
                tabel_result_name = tabel_result_name + '_libsvm'
                table_name_flag = False
            if (l_agr == +1 and l_dis == l_oth == -1) or (l_dis == -1 and l_agr == l_oth == +1):

                x_train = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                y_train = [l_agr] * len(feature_vects_agr_train) + [l_dis] * len(feature_vects_dis_train) + [l_oth] * len(
                    feature_vects_others_train)

                x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [l_oth] * len(
                    feature_vects_others_test)

            if (l_agr == +1 and l_dis == -1 and l_oth == 0):

                #this is a 3-class classification
                x_train = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                y_train = [l_agr] * len(feature_vects_agr_train) + [l_dis] * len(feature_vects_dis_train) + [l_oth] * len(
                    feature_vects_others_train)

                x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [l_oth] * len(
                    feature_vects_others_test)

            #-s 0 --> C-SVC (multi-class classification(, s 1 --> nu-SVC (multi-class classification) --> default: -n = 0.5
            #-t 0 --> linear kernel
            #-w option used to handle unbalanced data is for C-SVC, not nu-SVC!
            svm_params = ' -s 0 -c 1 -t 0 -w1 1 -w-1 2.7 -w0 1'
            #svm_params = '-g ' + str(g) + svm_params#adding the gamma value when -t 2
            #svm_params = '-s 1 -t 0 -n 0.5'
            prob = svmutil.svm_problem(y_train, x_train)
            param = svmutil.svm_parameter(svm_params)
            m = svmutil.svm_train(prob, param)
            #p_labels --> classification labels predicted by the system.
            #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
            #p_val --> classification values predicted by the system.
            p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, m)

        prediction_result, accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero = \
        funcs.calc_prediction_stats(train_with_liblinear, thresh_pos, thresh_neg, y_test, s_r_texts_test, p_label, p_val)

        my_util.write_csv_file(source_dir + test_result_name + str(n+1) + '_' + str(accuracy) + '%', False, True, prediction_result)
        results_CrossVal.append(
            [strip_thresh, n + 1,
            len(feature_vects_agr_train), len(feature_vects_dis_train), len(feature_vects_others_train),
            len(feature_vects_agr_test), len(feature_vects_dis_test), len(feature_vects_others_test),
            accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero]
        )

    results_CrossVal = sorted(results_CrossVal, key=itemgetter(header.index('accuracy')))
    results_CrossVal.reverse()
    means = [''] * header.index(n_fold_CV) + ['mean']#shift the mean to the right, so that it comes under n_fold_CV
    stdevs = [''] * header.index(n_fold_CV) + ['stdev']
    for column in range(header.index('tr_size_agr'), len(header)):
        data = [row[column] for row in results_CrossVal]
        mean, stdev = math_extra.calc_mean_stdev(data)
        means = means + [round(mean, 2)]
        stdevs = stdevs + [round(stdev, 2)]
    results_CrossVal = results_CrossVal + [means] + [stdevs]

    results.append(['']*len(header))#append an empty row
    results =  results + results_CrossVal

# print '\nmax:', results[0][header.index('accuracy')], 'iteration:', results[0][0]
# print 'min:', results[len(results) - 1][header.index('accuracy')], 'iteration:', results[len(results) - 1][0]
# print 'mean:', results[len(results) - 2][header.index('accuracy')]
# print 'stdev:', results[len(results) - 1][header.index('accuracy')]

my_util.write_csv_file(source_dir + tabel_result_name, False, True, results)
import random

__author__ = 'behzadbehzadan'

"""
In the older version, there were two dicts: one for positive set and another for negative set. This doesn't seem to have
any advantage. So, in this version we only use one dict for both of the negative and positive sets. Only for trainig and
test set we use two different dicts to strip non-frequent tokens independently from training and test sets.
"""

import my_util
import os
import funcs
import operator
import re
import unicodecsv

source_dir = os.path.expanduser('~')
file_dir = '/Chatterbox_UCL_Advance/Worry/'
collection_name = 'worried'
source_file = 'source' + '_' + collection_name
source_file_noDup = source_file + '_noDup'
labels_features_file_name = 'labels_features'
tweet_texts_file_name = 'all_tweet_texts'

########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
read_labels_and_features_from_file = False
n_fold_cross_val = 10
strip_thresholds = [0]#[0, 1, 2, 3, 4, 5, 10, 15, 20]
#positive labels are associated to worried/concerned/stressed... tweets.
l_pos = +1
#negative labels are associated to NOT worried/concerned/stressed... tweets.
l_neg = -1
#other labels are associated to any other types of tweets.
l_oth = 0
########################################################################################################################

if remove_retweets:
    try:
        tweets_noDup = my_util.read_csv_file(source_dir + file_dir + source_file_noDup, False, True)
        tweets = [t[0] for t in tweets_noDup]
        tweets = [t.lower() for t in tweets]
        tweets = [' '.join(t.split()) for t in tweets]#this part removes extra spaces that may exist between words.
    except IOError:
        #read the source file --> [[text1], [text2], [test3], ...]
        tweets = my_util.read_csv_file(source_dir + file_dir + source_file, False, True)
        #create list of texts --> [text1, text2, text3, ...]
        tweets = [t[0] for t in tweets]
        #make all letters lower-case --> this is essential when comparing strings and also when using quick_ratio
        tweets = [t.lower() for t in tweets]
        #put a space between any non punct char and a punct char
        tweets = [re.sub(r"(?u)(\w)(\W)", r"\1 \2", t) for t in tweets]
        #put a space between any punct char and a non punct char
        tweets = [re.sub(r"(?u)(\W)([\w@#])", r"\1 \2", t) for t in tweets]
        #remove extra spaces that may exist between words, by first splitting the words and then re-joining them.
        tweets = [' '.join(t.split())]
        #remove duplicates by direct comparison of strings
        tweets = funcs.remove_duplicate_tweets(tweets, False, None)
        #remove duplicates by direct comparison of the truncated strings
        tweets = funcs.truncate_and_remove_duplicates(tweets, 4)
        if use_qr_to_remove_dups:
            tweets = funcs.remove_duplicate_tweets(tweets, True, 0.89)
        my_util.write_csv_file(source_dir + file_dir + source_file_noDup, False, True, [[t] for t in tweets])

labels_pos=[]
labels_neg=[]
labels_oth=[]
all_labels=[]
feature_vects_pos=[]
feature_vects_neg=[]
feature_vects_oth=[]
all_feature_vects=[]
tweet_texts_pos=[]
tweet_texts_neg=[]
tweet_texts_oth=[]
all_texts=[]

if read_labels_and_features_from_file:
    # each row is in the following format: label \t address1:feature_value1 \t address2:feature_value2 \t ...
    rows = my_util.read_csv_file(source_dir + file_dir + labels_features_file_name, True, True)
    texts= my_util.read_csv_file(source_dir + file_dir + tweet_texts_file_name, False, True)
    if len(rows) == len(texts):
        for i in range(len(rows)):
            row = rows[i]
            text= texts[i]
            l = int(row[0])
            # each row is a string in the form of address:feature_value --> separate address from feature: [a,v]
            row =[ a_v.split(':') for a_v in row[1:] ]
            vector = { int(a_v[0]) : float(a_v[1]) for a_v in row }
            if l==l_pos:
                labels_pos.append(l)
                feature_vects_pos.append(vector)
                tweet_texts_pos.append(text)
            elif l==l_neg:
                labels_neg.append(l)
                feature_vects_neg.append(vector)
                tweet_texts_neg.append(text)
            elif l==l_oth:
                labels_oth.append(l)
                feature_vects_oth.append(vector)
                tweet_texts_oth.append(text)

    all_feature_vects = feature_vects_pos + feature_vects_neg# + feature_vects_oth
    all_labels = labels_pos + labels_neg #+labels_oth
    all_texts = tweet_texts_pos + tweet_texts_neg #+tweet_texts_oth

else:

    nots = ["never", "don't", "dont", "no", "not", "ain", "ainn", "aint", "ain't", "aren't", "arent", "isn't", "isnt",
                "wasn't", "wasnt", "weren't", "werent", "haven't", "havent", "hasn't", "hasnt", "won't", "wont", "can not",
                "cannot", "couldn't", "couldnt", "shouldn't", "shouldnt", "wouldn't", "wouldnt"]

    verbs = ["be", "been", "get"]

    no_signs = nots + [x + ' ' + y for x in nots for y in verbs]

    adverbs = ['as', 'so', 'so much', 'to', 'too', 'too much', 'very much', 'that much', 'this much', 'completely', 'totally',
               'entirely', 'extremely', 'nobody', 'anybody', 'anyone', 'ever', 'normally', 'really', "even"]

    #more_no_signs = ['worried at all', 'stop worrying about']
    more_no_signs = ['concerned at all', 'stop concerning about']

    no_signs = no_signs + [ x + ' ' + y for x in no_signs for y in adverbs]
    no_signs = [x + ' ' + collection_name for x in no_signs]
    no_signs = no_signs + more_no_signs

    #select tweets containing negative signs and put them in the negative set.
    positives = tweets[:]
    negatives = []
    for s in no_signs:
        temp = [t for t in positives if s in t]
        negatives = negatives + temp
        positives = [t for t in positives if t not in temp]

    print collection_name + ':', len(positives)
    print 'not_' + collection_name + ':', len(negatives)

    my_util.write_csv_file(source_dir + file_dir + 'not_' + collection_name, False, True, [[t] for t in negatives])
    my_util.write_csv_file(source_dir + file_dir + collection_name, False, True, [[t] for t in positives])

    print 'creating feature vectors...'

    # {'feature':feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
    # when using svm, an address represents a dimension on the space.
    # So it is important to keep the address consistent for positive and negative sets and also for training and test sets.
    features_dict = dict()
    #{'feature_address (dimension no) : feature} --> used for debugging to visualise features
    features_dict_reverse = dict()
    #{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
    features_count_dict = dict()
    #whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

    if new_normalisation_flag:
        #the very first index is always 1.
        max_index = 0
    else:
        max_index = 1
    #m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
    m=1
    #length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
    n=3

    feature_vects_pos, tweet_texts_pos, max_index = funcs.get_sparse_feature_vector_worry(positives, features_dict,
                                   features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)

    feature_vects_neg, tweet_texts_neg, max_index = funcs.get_sparse_feature_vector_worry(negatives, features_dict,
                                   features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)

    #feature_vects_oth, tweet_texts_oth, max_index = funcs.get_sparse_feature_vector_worry(others, features_dict,
    #                             features_count_dict, max_index, m, n,  remove_stpwds_for_unigrams, new_normalisation_flag)

    print 'feature vectors created!', 'No of features:', len(features_dict)

    labels_pos = [l_pos] * len(feature_vects_pos)
    labels_neg = [l_neg] * len(feature_vects_neg)
    #labels_pos = [l_oth] * len(feature_vects_oth)

    all_feature_vects = feature_vects_pos + feature_vects_neg# + feature_vects_oth
    all_labels = labels_pos + labels_neg #+labels_oth
    all_texts = tweet_texts_pos + tweet_texts_neg #+tweet_texts_oth

    funcs.write_labels_and_features_to_csv(all_labels, all_feature_vects, source_dir + file_dir + labels_features_file_name)
    my_util.write_csv_file(source_dir + file_dir + tweet_texts_file_name, False, True, [[t] for t in all_texts])

# visualising_thresh = 50
# funcs.write_features_and_freqs_to_csv(feature_vects_pos, features_count_dict_pos, visualising_thresh, source_dir + file_dir + collection_name + "_count_pos")
# funcs.write_features_and_freqs_to_csv(feature_vects_neg, features_count_dict_neg, visualising_thresh, source_dir + file_dir + collection_name + "_count_neg")

def find_double_keywords(the_list):
    worried_worried = []
    for t in the_list:
        t = t.split()
        keyword = collection_name
        count = len([x for x in t if x==keyword])
        if count>=2:
            t = ' '.join(t)
            worried_worried.append([t])
    return worried_worried

keyword_keyword_pos = find_double_keywords(tweet_texts_pos)
keyword_keyword_neg = find_double_keywords(tweet_texts_neg)
my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'pos', False, True, keyword_keyword_pos)
my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'neg', False, True, keyword_keyword_neg)

def calc_probs():

    print 'calculating probabilities...'
    prob_thresh = 0.69
    high_prob_features_pos=[]
    high_prob_features_neg=[]
    c=0
    for f, a in features_dict.iteritems():

        c+=1
        m = operator.mod(c, 1000)
        if m==0:
            print c

        neg_tweets_containing_f = [v for v in feature_vects_neg if a in v]#Note: a in v is exactly the same as v.has_key(a)
        pos_tweets_containing_f = [v for v in feature_vects_pos if a in v]
        p = len(pos_tweets_containing_f)
        n = len(neg_tweets_containing_f)
        smoothing_fact = 5

        if n !=0:
            prob_pos_given_f = float(p)/(p + n + smoothing_fact)
            if prob_pos_given_f > prob_thresh:
                high_prob_features_pos.append([f, prob_pos_given_f])
                #print prob_pos_given_f, f

        if p!=0:
            prob_neg_given_f = float(n)/(p + n + smoothing_fact)
            if prob_neg_given_f > prob_thresh:
                high_prob_features_neg.append([f, prob_neg_given_f])
                #print prob_neg_given_f, f

    my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_pos', False, True, high_prob_features_pos)
    my_util.write_csv_file(source_dir + file_dir + collection_name + '_high_probs_neg', False, True, high_prob_features_neg)

def shuffle_features_and_texts(list1, list2):
    if len(list1) == len(list2):
        zipped = zip(list1, list2)
        random.shuffle(zipped)
        unzipped = zip(*zipped)
        list1 = list(unzipped[0])
        list2 = list(unzipped[1])
    else:
        raise ValueError('the two list are not equal size!')
    #note that zip returns a new object and does not pass reference
    return list1, list2

feature_vects_pos, tweet_texts_pos = shuffle_features_and_texts(feature_vects_pos, tweet_texts_pos)
feature_vects_neg, tweet_texts_neg = shuffle_features_and_texts(feature_vects_neg, tweet_texts_neg)
feature_vects_oth, tweet_texts_oth = shuffle_features_and_texts(feature_vects_oth, tweet_texts_oth)

test_set_size_pos = len(feature_vects_pos) / n_fold_cross_val
test_set_size_neg = len(feature_vects_neg) / n_fold_cross_val
test_set_size_oth = len(feature_vects_oth) / n_fold_cross_val
results = []
n_fold_CV = '%d_fold_CV' % n_fold_cross_val
header = ['strip_thresh',
          n_fold_CV, 'tr_size_agr', 'tr_size_dis', 'tr_size_oth', 'ts_size_agr', 'ts_size_dis', 'ts_size_oth',
          'accuracy', 'precision_pos', 'precision_neg', 'precision_zero', 'recall_pos', 'recall_neg', 'recall_zero']

results.append(header)

for strip_thresh in strip_thresholds:

    results_CrossVal = []
    for n in range(0, n_fold_cross_val):

        strt_pos = n * test_set_size_pos
        strt_neg = n * test_set_size_neg
        strt_oth = n * test_set_size_oth

        print str(n_fold_cross_val) + '-fold cross validation in progress...'
        print 'iteration', n, '\n'

        if n < n_fold_cross_val - 1:
            end_pos = (n + 1) * test_set_size_pos
            end_neg = (n + 1) * test_set_size_neg
            end_oth = (n + 1) * test_set_size_oth

        else:#last part
            end_pos = len(feature_vects_pos)
            end_neg = len(feature_vects_neg)
            end_oth = len(feature_vects_oth)

        test_set_vects_pos = feature_vects_pos[strt_pos: end_pos]
        test_set_vects_neg = feature_vects_neg[strt_neg: end_neg]
        test_set_vects_oth = feature_vects_oth[strt_oth: end_oth]

        test_set_texts_pos = tweet_texts_pos[strt_pos: end_pos]
        test_set_texts_neg = tweet_texts_neg[strt_neg: end_neg]
        test_set_texts_oth = tweet_texts_oth[strt_oth: end_oth]

        print 'test set size positive:', len(test_set_vects_pos)
        print 'test set size negative:', len(test_set_vects_neg)
        print 'test set size others', len(test_set_vects_oth)

        train_set_vects_pos = [x for x in feature_vects_pos if x not in test_set_vects_pos]
        train_set_vects_dis = [x for x in feature_vects_neg if x not in test_set_vects_neg]
        train_set_vects_oth = [x for x in feature_vects_oth if x not in test_set_vects_oth]

        # we need to create two new dicts: one for training and one for test. count all the feature
        #in the test set. this gives the test dict count. subtract this from the original one to get the trainig dict.


        features_dict_train = copy.deepcopy(features_dict)
        features_count_dict_train = copy.deepcopy(features_count_dict)
        features_dict_reverse_train = funcs.get_features_dict_reverse(features_dict_train)

        if strip_thresh > 0:
            feature_vects_agr_train = funcs.strip_less_than(feature_vects_agr_train, features_count_dict_train,
                                                            strip_thresh)
            feature_vects_dis_train = funcs.strip_less_than(feature_vects_dis_train, features_count_dict_train,
                                                            strip_thresh)
            feature_vects_others_train = funcs.strip_less_than(feature_vects_others_train, features_count_dict_train,
                                                               strip_thresh)

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
            feature_vects_agr_test = funcs.strip_less_than(feature_vects_agr_test, features_count_dict_test,
                                                           strip_thresh)
            feature_vects_dis_test = funcs.strip_less_than(feature_vects_dis_test, features_count_dict_test,
                                                           strip_thresh)
            feature_vects_others_test = funcs.strip_less_than(feature_vects_others_test, features_count_dict_test,
                                                              strip_thresh)

        if train_with_liblinear == train_with_svm:
            raise Exception('The liblinear and libsvm flags are either both True or both False!')


        elif train_with_svm:

            if table_name_flag:
                tabel_result_name = tabel_result_name + '_libsvm'
                table_name_flag = False

            if (l_agr == +1 and l_dis == l_oth == -1) or (l_dis == -1 and l_agr == l_oth == +1):
                x_train = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                y_train = [l_agr] * len(feature_vects_agr_train) + [l_dis] * len(feature_vects_dis_train) + [
                    l_oth] * len(
                    feature_vects_others_train)

                x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [l_oth] * len(
                    feature_vects_others_test)

                # s 1 --> nu-SVC (multi-class classification) --> default: -n = 0.5
                svm_params = '-s 1 -t 0 -n 0.5'
                p_label, p_acc, p_val = train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)
                prediction_result, accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero = \
                    funcs.calc_prediction_stats(None, None, y_test, s_r_texts_test, p_label, [])

            if (l_agr == +1 and l_dis == -1 and l_oth == 0):

                if one_against_one == False:

                    #this is a 3-class classification
                    x_train = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                    s_r_texts_train = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                    y_train = [l_agr] * len(feature_vects_agr_train) + [l_dis] * len(feature_vects_dis_train) + [
                        l_oth] * len(
                        feature_vects_others_train)

                    x_test = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                    s_r_texts_test = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                    y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [
                        l_oth] * len(
                        feature_vects_others_test)

                    #-s 0 --> C-SVC (multi-class classification)
                    #-t 0 --> linear kernel
                    #-w option used to handle unbalanced data is for C-SVC, not nu-SVC!
                    svm_params = ' -s 0 -c 1 -t 0 -w1 1 -w-1 2.7 -w0 1'
                    p_label, p_acc, p_val = train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params)
                    prediction_result, accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero = \
                        funcs.calc_prediction_stats(None, None, y_test, s_r_texts_test, p_label, [])

                elif one_against_one == True:

                    #first classify agreed from the rest (i.e. disagreed and others)!
                    x_train_1 = feature_vects_agr_train + feature_vects_dis_train + feature_vects_others_train
                    s_r_texts_train_1 = s_r_texts_agr_train + s_r_texts_dis_train + s_r_texts_others_train
                    l_rest = l_dis
                    y_train_1 = [l_agr] * len(feature_vects_agr_train) + [l_rest] * len(feature_vects_dis_train) + [
                        l_rest] * len(feature_vects_others_train)

                    x_test_1 = feature_vects_agr_test + feature_vects_dis_test + feature_vects_others_test
                    s_r_texts_test_1 = s_r_texts_agr_test + s_r_texts_dis_test + s_r_texts_others_test
                    y_test_1 = [l_agr] * len(feature_vects_agr_test) + [l_rest] * len(feature_vects_dis_test) + [
                        l_rest] * len(feature_vects_others_test)

                    #the final y
                    y_test = [l_agr] * len(feature_vects_agr_test) + [l_dis] * len(feature_vects_dis_test) + [
                        l_oth] * len(feature_vects_others_test)

                    # s 1 --> nu-SVC (multi-class classification) --> default: -n = 0.5
                    svm_params = '-s 1 -t 0 -n 0.5'
                    p_label_1, p_acc, p_val = train_and_test_with_libsvm(y_train_1, x_train_1, y_test_1, x_test_1,
                                                                         svm_params)

                    #make a side by side matrix
                    nRows = len(y_test)
                    x_sr_y1_p1_y = [[x_test_1[i], s_r_texts_test_1[i], y_test_1[i], p_label_1[i], y_test[i]] for i in
                                    xrange(nRows)]

                    prediction_result_1, accuracy_1, precision_pos_1, precision_neg_1, precision_zero_1, recall_pos_1, recall_neg_1, recall_zero_1 = \
                        funcs.calc_prediction_stats(None, None, y_test_1, s_r_texts_test_1, p_label_1, [])

                    #extra step to equalise the two classification set sizes.
                    min_dis_others = min(len(feature_vects_dis_train), len(feature_vects_others_train))
                    feature_vects_dis_train = feature_vects_dis_train[0:min_dis_others]
                    feature_vects_others_train = feature_vects_others_train[0:min_dis_others]

                    #second: classify dispos from others. This is count as a separate classification;
                    #the training set does not need to have any intersection with the first classification stage.
                    x_train_2 = feature_vects_dis_train + feature_vects_others_train
                    y_train_2 = [l_dis] * len(feature_vects_dis_train) + [l_oth] * len(feature_vects_others_train)

                    #the test set is the correct predictions in the stage one that does not contain agreed tweets.
                    nCol = len(x_sr_y1_p1_y[0])
                    columns = [[row[i] for row in x_sr_y1_p1_y if (row[2] == row[3] == l_rest)] for i in xrange(nCol)]
                    x_test_2 = columns[0]
                    s_r_texts_test_2 = columns[1]
                    y_test_2 = columns[4]

                    p_label_2, p_acc, p_val = train_and_test_with_libsvm(y_train_2, x_train_2, y_test_2, x_test_2,
                                                                         svm_params)

                    #make a side by side matrix
                    x_sr_y2_p2 = [[x_test_2[i], s_r_texts_test_2[i], y_test_2[i], p_label_2[i]] for i in
                                  xrange(len(y_test_2))]

                    prediction_result_2, accuracy_2, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero = \
                        funcs.calc_prediction_stats(None, None, y_test_2, s_r_texts_test_2, p_label_2, [])

                    #copy the predicted labels obtained in stage two to the corresponding labels in stage one.
                    for u in x_sr_y1_p1_y:
                        for v in x_sr_y2_p2:
                            if u[:1] == v[:1]:
                                u[3] = v[3]

                    s_r_texts_test = [row[1] for row in x_sr_y1_p1_y]
                    y_test = [row[4] for row in x_sr_y1_p1_y]
                    p_label = [row[3] for row in x_sr_y1_p1_y]

                    #note that the precision_neg is a bit distorted since a false negative in the first stage could in
                    #belong to either the negative set or the zero set that will be classified in the second stage.
                    prediction_result, accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero = \
                        funcs.calc_prediction_stats(None, None, y_test, s_r_texts_test, p_label, [])

                    my_util.write_csv_file(
                        source_dir + result_name + str(n + 1) + '_' + 'agr-rest' + '_' + str(accuracy_1) + '%', False,
                        True, prediction_result_1)
                    my_util.write_csv_file(
                        source_dir + result_name + str(n + 1) + '_' + 'dis-oth' + '_' + str(accuracy_2) + '%', False,
                        True, prediction_result_2)
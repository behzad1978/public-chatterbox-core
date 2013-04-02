import random

__author__ = 'behzadbehzadan'

"""
In the older version, there were two dicts: one for positive set and another for negative set. This doesn't seem to have
any advantage. So, in this version we only use one dict for both of the negative and positive sets. Only for trainig and
test set we use two different dicts to strip non-frequent tokens independently from training and test sets.
"""

import my_util
import os
import funcs_worry
import operator
import re
import copy
import svmutil

source_dir = os.path.expanduser('~')
file_dir = '/Chatterbox_UCL_Advance/Worry/'
collection_name = 'worried'
source_file = 'source' + '_' + collection_name
source_file_noDup = source_file + '_noDup'
labels_features_file_name = 'labels_features'
tweet_texts_file_name = 'all_tweet_texts'
n_of_features_file_name = 'n_of_features'
result_file_name = 'results_worry'

########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = True
read_labels_and_features_from_file = False
n_fold_cross_val = 2
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
        tweets = funcs_worry.remove_duplicate_tweets(tweets, False, None)
        #remove duplicates by direct comparison of the truncated strings
        tweets = funcs_worry.truncate_and_remove_duplicates(tweets, 4)
        if use_qr_to_remove_dups:
            tweets = funcs_worry.remove_duplicate_tweets(tweets, True, 0.89)
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
n_of_features_pos=[]
n_of_features_neg=[]
n_of_features_oth=[]

if read_labels_and_features_from_file:
    # each row is in the following format: label \t address1:feature_value1 \t address2:feature_value2 \t ...
    rows = my_util.read_csv_file(source_dir + file_dir + labels_features_file_name, True, True)
    texts= my_util.read_csv_file(source_dir + file_dir + tweet_texts_file_name, False, True)
    n_of_features= my_util.read_csv_file(source_dir + file_dir + n_of_features_file_name, False, True)
    if len(rows) == len(texts):
        for i in range(len(rows)):
            row = rows[i]
            text= texts[i]
            l = int(row[0])
            n = n_of_features[i]
            # each row is a string in the form of address:feature_value --> separate address from feature: [a,v]
            row =[ a_v.split(':') for a_v in row[1:] ]
            vector = { int(a_v[0]) : float(a_v[1]) for a_v in row }
            if l==l_pos:
                labels_pos.append(l)
                feature_vects_pos.append(vector)
                tweet_texts_pos.append(text)
                n_of_features_pos.append(n)
            elif l==l_neg:
                labels_neg.append(l)
                feature_vects_neg.append(vector)
                tweet_texts_neg.append(text)
                n_of_features_neg.append(n)
            elif l==l_oth:
                labels_oth.append(l)
                feature_vects_oth.append(vector)
                tweet_texts_oth.append(text)
                n_of_features_oth.append(n)

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
    positives = tweets[:1000]
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

    # {'feature' : feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
    # when using svm, an address represents a dimension on the input-space. So it is important to keep the address
    # consistent for positive and negative sets and also for training and test sets.
    features_dict = dict()
    # {feature_address (dimension no) : 'feature'} --> used for debugging to visualise features
    features_dict_reverse = dict()
    #{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
    features_count_dict = dict()
    # whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

    if new_normalisation_flag:
        #the very first index is always 1.

        max_index = 0
    else:
        max_index = 1
    #m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
    m=1
    #length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
    n=3

    feature_vects_pos, tweet_texts_pos, max_index, n_of_features_pos = funcs_worry.get_sparse_feature_vector_worry(positives, features_dict,
                                   features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)

    feature_vects_neg, tweet_texts_neg, max_index, n_of_features_neg = funcs_worry.get_sparse_feature_vector_worry(negatives, features_dict,
                                   features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)

    #feature_vects_oth, tweet_texts_oth, max_index, n_of_features_oth = funcs.get_sparse_feature_vector_worry(others, features_dict,
    #                             features_count_dict, max_index, m, n,  remove_stpwds_for_unigrams, new_normalisation_flag)

    print 'feature vectors created!', 'No of features:', len(features_dict)

    labels_pos = [l_pos] * len(feature_vects_pos)
    labels_neg = [l_neg] * len(feature_vects_neg)
    #labels_pos = [l_oth] * len(feature_vects_oth)

    all_feature_vects = feature_vects_pos + feature_vects_neg# + feature_vects_oth
    all_labels = labels_pos + labels_neg #+labels_oth
    all_texts = tweet_texts_pos + tweet_texts_neg #+tweet_texts_oth
    all_n_of_features = n_of_features_pos + n_of_features_neg #+ n_of_features_oth

    funcs_worry.write_labels_and_features_to_csv(all_labels, all_feature_vects, source_dir + file_dir + labels_features_file_name)
    my_util.write_csv_file(source_dir + file_dir + tweet_texts_file_name, False, True, [[t] for t in all_texts])
    my_util.write_csv_file(source_dir + file_dir + n_of_features_file_name, False, True, [[n] for n in all_n_of_features])

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

def shuffle_features_texts_n(list1, list2, list3):
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

feature_vects_pos, tweet_texts_pos, n_of_features_pos = shuffle_features_texts_n(feature_vects_pos, tweet_texts_pos, n_of_features_pos)
feature_vects_neg, tweet_texts_neg, n_of_features_neg = shuffle_features_texts_n(feature_vects_neg, tweet_texts_neg, n_of_features_neg)
#feature_vects_oth, tweet_texts_oth, n_of_features_oth = shuffle_features_texts_n(feature_vects_oth, tweet_texts_oth, n_of_features_oth)
feature_vects_oth =[]
tweet_texts_oth=[]
n_of_features_oth=[]

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
        train_set_vects_neg = [x for x in feature_vects_neg if x not in test_set_vects_neg]
        train_set_vects_oth = [x for x in feature_vects_oth if x not in test_set_vects_oth]

        # we need to create two new dicts: one for training and one for test. count all the feature
        #in the test set. this gives the test dict count. subtract this from the original one to get the trainig dict.
        features_count_dict_train = copy.deepcopy(features_count_dict)
        all_test_set_vects = test_set_vects_pos + test_set_vects_neg + test_set_vects_oth
        all_norm_factors = n_of_features_pos + n_of_features_neg + n_of_features_oth
        for i in range(len(all_test_set_vects)):
            vect = all_test_set_vects[i]
            fact = all_norm_factors[i]
            for a, r in vect.iteritems():
                c_test = r*fact
                c_train_and_test = features_count_dict_train[a]
                diff = c_train_and_test - c_test
                features_count_dict_train[a] = diff

        if strip_thresh > 0:
            train_set_vects_pos = funcs_worry.strip_less_than(train_set_vects_pos, features_count_dict_train, strip_thresh)
            train_set_vects_neg = funcs_worry.strip_less_than(train_set_vects_neg, features_count_dict_train, strip_thresh)
            train_set_vects_oth = funcs_worry.strip_less_than(train_set_vects_oth, features_count_dict_train, strip_thresh)
            test_set_vects_pos = funcs_worry.strip_less_than(test_set_vects_pos, features_count_dict_train, strip_thresh)
            test_set_vects_neg = funcs_worry.strip_less_than(test_set_vects_neg, features_count_dict_train, strip_thresh)
            test_set_vects_oth = funcs_worry.strip_less_than(test_set_vects_oth, features_count_dict_train, strip_thresh)


        x_train = train_set_vects_pos + train_set_vects_neg + train_set_vects_oth
        y_train = [l_pos]*len(train_set_vects_pos) + [l_neg]*len(train_set_vects_neg) + [l_oth]*len(train_set_vects_oth)

        x_test = test_set_vects_pos + test_set_vects_neg + test_set_vects_oth
        test_set_texts = test_set_texts_pos + test_set_texts_neg + test_set_texts_oth
        y_test = [l_pos]*len(test_set_vects_pos) + [l_neg]*len(test_set_vects_neg) + [l_oth]*len(test_set_vects_oth)

        #-s 0 --> C-SVC (multi-class classification)
        #-s 1 --> nu-SVC (multi-class classification) --> default: -n = 0.5
        #-t 0 --> linear kernel
        #-w option used to handle unbalanced data is for C-SVC, not nu-SVC!
        svm_params = '-s 1 -t 0 -n 0.5'# -w0 1'
        #svm_params = '-s 0 -c 1 -t 0 -w1 1 -w-1 3.55'# -w0 1'
        prob = svmutil.svm_problem(y_train, x_train)
        param = svmutil.svm_parameter(svm_params)
        m = svmutil.svm_train(prob, param)

        #p_labels --> classification labels predicted by the system.
        #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
        #p_val --> classification values predicted by the system.
        p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, m)
        prediction_result, accuracy, precisions, recalls = funcs_worry.calc_prediction_stats(y_test, test_set_texts, p_label, [l_pos, l_neg, l_oth])

        my_util.write_csv_file(source_dir + result_file_name + str(n + 1) + '_' + str(accuracy) + '%', False, True, prediction_result)

#         results_CrossVal.append(
#             [strip_thresh, n + 1,
#              len(feature_vects_agr_train), len(feature_vects_dis_train), len(feature_vects_others_train),
#              len(feature_vects_agr_test), len(feature_vects_dis_test), len(feature_vects_others_test),
#              accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero]
#         )
#
# results_CrossVal = sorted(results_CrossVal, key=itemgetter(header.index('accuracy')))
# results_CrossVal.reverse()
# means = [''] * header.index(n_fold_CV) + ['mean']#shift the mean to the right, so that it comes under n_fold_CV
# stdevs = [''] * header.index(n_fold_CV) + ['stdev']
# for column in range(header.index('tr_size_agr'), len(header)):
#     data = [row[column] for row in results_CrossVal]
#     mean, stdev = math_extra.calc_mean_stdev(data)
#     means = means + [round(mean, 2)]
#     stdevs = stdevs + [round(stdev, 2)]
# results_CrossVal = results_CrossVal + [means] + [stdevs]
#
# results.append([''] * len(header))#append an empty row
# results = results + results_CrossVal
#
# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

from operator import itemgetter
import re
import my_util
import difflib
import stopwords
import svmutil

#list of URL shorteners.
shorteners = ["t.co", "goo.gl", "img.ly", "bit.ly", "is.gd", "tinyurl.com", "is.gd", "tr.im", "ow.ly", "cli.gs",
              "twurl.nl", "Digg.com", "u.mavrev.com", "tiny.cc", "short.to", "BudURL.com", "snipr.com", "6url.com",
              "snipurl.com", "Just.as", "snurl.com", "su.pr", "kl.am", "adjix.com", "xrl.us", "notlong.com", "short.ie",
              "sn.im", "idek.net", "AltURL.com", "xr.com", "twurl.cc", "Yep.it", "RedirX.com", "to.ly", "fon.gs",
              "x.se", "twirl.at", "ru.ly"]

read_training_data_from_file = False

def exclude_url(split_text):
    #drop the URLs
    no_url_tweet_text = []
    for t in split_text:
        hasshortener = False
        for short in shorteners:
            if short in t:
                hasshortener = True
        if hasshortener:
            pass
        elif "http" in t:
            pass
        elif "://" in t:
            pass
        elif 'www' in t:
            pass
        else:
            no_url_tweet_text.append(t)
    return no_url_tweet_text

def check_feature(f, stopword_flag):
    global train_labs, shorteners
    #print train_labs
    if f == "" or f == None:
        return None

    if stopword_flag:
        stp = stopwords.stopwords['en']
        if f in stp:
            return False
    if f[0] == "@":
        return False
    if "http" in f:
        return False
    if "://" in f:
        return False
    if f == "RT":
        return False
    if f == "rt":
        return False
    # if f in train_labs:
    #     if random.randint(0, 10) == 5:
    #         return True
    #     else:
    #         return False
    if "www" in f:
        return False
    for short in shorteners:
        if short in f:
            return False
    else:
        return True


def check_features(f_list, stopword_flag):
    #print f_list
    for f in f_list:
        if not (check_feature(f, stopword_flag)):
            return False
    return True

def add_to_dict(t, the_length, vector, features_dict, features_count_dict, max_index):
    try:
        #when using svm, the indx is a dimension in the space!
        indx = features_dict[t]
    except KeyError:
        features_dict[t] = max_index
        indx = max_index
        max_index = max_index + 1
    try:
        vector[indx] = vector[indx] + (1.0 / the_length)
    except KeyError:
        vector[indx] = (1.0 / the_length)
    try:
        features_count_dict[indx] = features_count_dict[indx] + 1
    except KeyError:
        features_count_dict[indx] = 1

    return max_index

def get_ngrams_worry(tweet, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag):
    """
    this provides a term-frequency vector
    """
    vector = dict()
    #print tweet
    split_tweet_text = tweet.lower().split()

    no_url_tweet_text = exclude_url(split_tweet_text)
    tweet_text = " ".join(no_url_tweet_text)
    #print tweet_text
    ##put a space between any non punct char and a punct char
    tweet_text = re.sub(r"([^'\".,;:/?\!@#£$%^&*()_\-=+`~])(['\".,;:/?\!@#£$%^&*()_\-=+`~])", r"\1 \2", tweet_text)
    #print tweet_text
    ##put a space between any punct char and a non punct char
    tweet_text = re.sub(r"(['\".,;:/?\!£$%^&*()_\-=+`~])([^'\".,;:/?\!@#£$%^&*()_\-=+`~#@])", r"\1 \2", tweet_text)
    #print tweet_text
    #stick heart symbols back together
    tweet_text = re.sub(" < 3 ", " <3 ", tweet_text)
    tweet_text = re.sub(" : d ", " :d ", tweet_text)
    tweet_text = re.sub(" : p ", " :p ", tweet_text)

    #print tweet_text
    #stick n't back together
    tweet_text = re.sub(r"(\w)n ' t\b", r"\1 n't", tweet_text)
    #print tweet_text
    tweet_text = re.sub(r" ([:;][-]?) ([DP]) ", r"\1\2", tweet_text)
    #print tweet_text
    #concatinate any letter that is repeated more than 3 times. like: 'yesssss' to 'yesss' or 'nooooo' to 'nooo'.
    #\s represents any non-whitespace character. \1 matches the same character 1 time.
    tweet_text = re.sub(r"(\S)\1\1+", r"\1\1\1", tweet_text)

    tokens = tweet_text.split()

    #i --> the length of the token
    #j --> starting index of the token
    norm_factor = 0
    n_of_features = 0
    #sometimes tokens may be empty --> eg: when tweet is just a url --> we exclude the url and it results an empty list.
    if len(tokens) > 0:
        for i in range(m, n + 1):
            stpwd_flag = False
            if remove_stpwds_for_unigrams:
                if i == 1:
                    stpwd_flag = True
            for j in xrange(0, len(tokens) - (i - 1)):
                if check_features(tokens[j:j + i], stpwd_flag):
                    n_of_features += 1
                    t = " ".join(tokens[j:j + i])

                    if new_normalisation_flag:
                        if t not in features_dict:
                            max_index += 1
                            features_dict[t] = max_index
                            features_count_dict[max_index] = 0

                        #a is the address or dimension number
                        a = features_dict[t]

                        if a not in vector:
                            vector[a] = 0

                        vector[a] += 1
                        features_count_dict[a] += 1
                    else:
                        #note: vector, features_dict, features_count_dict are passed by reference
                        #only max_index must be returned, as its value changes inside the method.
                        max_index = add_to_dict(t, len(tokens), vector, features_dict, features_count_dict, max_index)
                        norm_factor = len(tokens)

        # The following line is performed when the i-loop is finished. This is because, for a given tweet_text, the
        # representing feature-vector must include values of all ngrams extracted from the text.
        if new_normalisation_flag:
            # Divide all elements of the vector by the number of traversed features.
            # This is more correct than the division by len(tokens), as the no. of ngrams (addressed in a vector) is not
            # necessarily equal to the number of features.
            vector = {a : float(c)/n_of_features for a, c in vector.iteritems()}
            norm_factor = n_of_features

    return vector, max_index, norm_factor


def get_sparse_feature_vector_worry(tweet_list, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag):

    feature_vectors = []
    tweet_texts = []
    normal_factors = []
    for ind, tweet in enumerate(tweet_list):
        vector, max_index, normal_factor = get_ngrams_worry(tweet, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)
        feature_vectors.append(vector)
        #in general the tweet_texts should be tweet_list itself. We return this in case the order of the vector changes
        # during the loop.
        tweet_texts.append(tweet)
        normal_factors.append(normal_factor)

        if (ind % 1000) == 0:
            print ind

    return feature_vectors, tweet_texts, max_index, normal_factors

def write_features_and_freqs_to_csv(features_dict, features_count_dict, thresh, filename):
    feature_freq_list = []
    for f, a in features_dict.iteritems():
        c = features_count_dict[a]
        if c > thresh:
            feature_freq_list.append([f, c])
        else:
            pass
    my_util.write_csv_file(filename, False, True, feature_freq_list)

def get_features_dict_reverse(features_dict):
    """
    create a reverse dictionary to ba able to call the address of a feature and get the feature itself.
    mainly used for debug purpose.
    """
    features_dict_reverse = {v:f for f,v in features_dict.iteritems()}
    return features_dict_reverse

def make_feature_count_dict_test(features_count_dict, features_count_dict_train, features_dict_reverse):
    # method creating the test-set dictionary!

    features_dict_test = dict() #only contains features of the test set
    features_count_dict_test = dict()

    for ref, c in features_count_dict.iteritems():
        add_to_dict = False
        if features_count_dict_train.has_key(ref):
            #if the feature also exist in the training set,
            #subtract the values to obtain the count of the feature only in the test set.
            c_train = features_count_dict_train[ref]
            if c - c_train <> 0:#if the diff is 0, the feature only belongs to training set; so, don't add to test set.
                add_to_dict = True
                c_test = c - c_train
        else:
            #if the feature does not exist in the training set, then it belongs to the test set.
            add_to_dict = True
            c_test = c
        if add_to_dict:
            features_count_dict_test[ref] = c_test
            f = features_dict_reverse[ref]
            features_dict_test[f] = ref
    return features_dict_test, features_count_dict_test

def strip_less_than(vectors, features_count_dict, m):
    out_vectors = []
    for vector in vectors:
        new_vector = dict()
        for k, v in vector.items():
            if features_count_dict[k] > m:
                new_vector[k] = v
            #else:
            #print 'single_feature:', features_dict_reverse[k]
        out_vectors.append(new_vector)
    return out_vectors


def duplicate(set1, size, rand):
    #set2 is bigger
    #note that random.sample cannot be used if the size of set2 is more than two times bigger than set1
    if size > 0:
        length = len(set1)
        for i in xrange(0, size):
            r = rand.randint(0, length - 1)
            duplicated_vect = set1[r]
            set1.append(duplicated_vect)
    return set1

def remove_duplicates(set_of_seed_reply_tupples):
    noDuplicate = set_of_seed_reply_tupples[:]
    for sr in set_of_seed_reply_tupples:
        duplicates = [sr1 for sr1 in set_of_seed_reply_tupples if sr == sr1]
        dup = duplicates.pop(0)
        if len(duplicates) > 0:
            noDuplicate = [sr for sr in noDuplicate if sr not in duplicates]
            noDuplicate.append(dup)
    return noDuplicate

def calc_prediction_stats(y_test, tweet_texts, p_label, labels):
    #labels is a list containing labels, like: [0,+1,-1]
    prediction_result=[]
    header = ['text', 'original_label', 'predicted_label', 'prediction_success']
    prediction_result.append(header)
    true_counts =[0]*len(labels)
    false_counts=[0]*len(labels)
    n_samples = [0]*len(labels)

    for i in range(len(tweet_texts)):
        text = tweet_texts[i]
        original_label = y_test[i]
        predicted_label = int(p_label[i])
        prediction_success = 'wrong'

        for j in range(len(labels)):
            if original_label == labels[j]:
                n_samples[j] +=1
                if predicted_label == original_label:
                    prediction_success = 'correct'
                    true_counts[j] +=1
                    break

            if predicted_label == labels[j]:
                false_counts[j] +=1

        prediction_result.append([text, original_label, predicted_label, prediction_success])

    accuracy = round(float(sum(true_counts)) / len(y_test), 2)

    # do-something if x, else do-something else.
    precisions = [ round(float(true_counts[i]) / (true_counts[i]+false_counts[i]), 2) if
                   (true_counts[i] + false_counts[i]) <>0 else 0 for i in range(len(labels)) ]

    recalls = [ round(float(true_counts[i]) / n_samples[i], 2) if n_samples[i] <> 0 else 0 for i in range(len(labels)) ]

    return prediction_result, accuracy, precisions, recalls


def calc_prediction_stats_2(y_test, tweet_texts, p_label, class_labels):
    # y_test is the list of desired labels which are supposed to be predicted.
    # tweet_texts is the list of tweet_texts in the test set, corresponding (1-1) to labels in the y_test.
    # p_label is a list of predicted labels
    # class_labels is a dict containing classification label types: {'pos':1, 'neg':-1, 'oth':0}
    prediction_result = []
    header = ['text', 'original_label', 'predicted_label', 'prediction_success']
    prediction_result.append(header)
    # true_counts records the number of correct predictions for each label-type in the whole test-set.
    true_counts = dict(zip(class_labels.keys(), [0] * len(class_labels)))
    # false_counts records the number of wrong predictions for each label-type in the whole test-set.
    false_counts = dict(zip(class_labels.keys(), [0] * len(class_labels)))
    # n_samples gives the size of different classes in the test set (how many of each label-type exists in the y_test).
    n_samples = dict(zip(class_labels.keys(), [0] * len(class_labels)))

    # iterate through every element in the test set.
    for i in range(len(tweet_texts)):
        text = tweet_texts[i]
        original_label = y_test[i]
        predicted_label = int(p_label[i])
        prediction_success = 'wrong'

        for l, j in class_labels.iteritems():
            if original_label == j:
                n_samples[l] += 1
                if predicted_label == original_label:
                    prediction_success = 'correct'
                    true_counts[l] += 1
                    break

            if predicted_label == j:
                false_counts[l] += 1

        # when saved in csv format, this is a file showing the predicted state of each element in the test set.
        prediction_result.append([text, original_label, predicted_label, prediction_success])

    accuracy = round(float(sum(true_counts.values())) / len(y_test), 2)

    # for every class in the test set calculate the respective precision and recall values.
    # do-something if x, else do-something else.
    precisions = { l : round(float(true_counts[l]) / (true_counts[l] + false_counts[l]), 2) if
                  (true_counts[l] + false_counts[l]) <> 0 else 0 for l in class_labels.keys()}

    recalls = { l : round(float(true_counts[l]) / n_samples[l], 2) if
               n_samples[l] <> 0 else 0 for l in class_labels.keys() }

    return prediction_result, accuracy, precisions, recalls


def remove_duplicate_tweets(tweets_list, use_quick_ratio, thresh):
    clustered_tweets = []#cluster duplicated/similar tweets together
    while len(tweets_list) > 0:
        t = tweets_list[0]
        if use_quick_ratio:
            #using the quick_ratio() is very time consuming!
            #note that without lower-casing the strings, the quick_ratio() does not work correctly.
            #so the string must have become lower-case before.
            duplicates = [s for s in tweets_list if difflib.SequenceMatcher(None, s, t).quick_ratio() > thresh]
        else:
            #note: the string must have become lower-case before this stage.
            duplicates = [s for s in tweets_list if s == t]

        clustered_tweets.append([t, len(duplicates)])
        tweets_list = [x for x in tweets_list if x not in duplicates]
        if len(duplicates) > 1:
            print len(duplicates), 'duplicate: ', t
            for d in duplicates:
                print d
        print 'removing duplicates ... tweets remained:', len(tweets_list)
    unique_tweets = [d[0] for d in clustered_tweets]#take one element from each cluster
    return unique_tweets#ß, clustered_tweets

def truncate_and_remove_duplicates(tweets, trunc_size):
    #a trunc_size of 2 or 3 sounds reasonable!
    clustered_tweets = []#cluster duplicated/similar tweets together

    #split tweet texts
    truncs = [t.split() for t in tweets]
    #truncate the beginning and the end of tweets. If very short, then don't truncate.
    truncs = [t[trunc_size : len(t) - trunc_size] for t in truncs if len(t) > 2*trunc_size]
    #stick back the split truncated tweets
    truncs = [' '.join(t) for t in truncs]
    #this dict is used to retrieve original tweets later
    trunc_tweet_dict = dict( zip(truncs, tweets) )

    unique_truncs = remove_duplicate_tweets(truncs, False, None)
    unique_tweets = [trunc_tweet_dict[s] for s in unique_truncs]

    return unique_tweets


def remove_retweets(tweets, use_qr_to_remove_dups):
    #make all letters lower-case --> this is essential when comparing strings and also when using quick_ratio
    tweets = [t.lower() for t in tweets]
    #put a space between any non punct char and a punct char
    tweets = [re.sub(r"(?u)(\w)(\W)", r"\1 \2", t) for t in tweets]
    #put a space between any punct char and a non punct char
    tweets = [re.sub(r"(?u)(\W)([\w@#])", r"\1 \2", t) for t in tweets]
    #remove extra spaces that may exist between words, by first splitting the words and then re-joining them.
    tweets = [' '.join(t.split()) for t in tweets]
    #remove duplicates by direct comparison of the truncated strings
    tweets = truncate_and_remove_duplicates(tweets, 4)
    if use_qr_to_remove_dups:
        #remove duplicates by direct comparison of strings
        tweets = remove_duplicate_tweets(tweets, True, 0.89)

    return tweets

def write_labels_features_in_libsvm_form(labels, features, file_name):
    """
    this function creates a tab deliminator csv file of the labels and features in libsvm format:
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
    my_util.write_csv_file(file_name, True, True, final_list)


def train_and_test_with_libsvm(y_train, x_train, y_test, x_test, svm_params):
    prob = svmutil.svm_problem(y_train, x_train)
    param = svmutil.svm_parameter(svm_params)
    #Show values of parameters
    print(param)
    m = svmutil.svm_train(prob, param)
    #p_labels --> classification labels predicted by the system.
    #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
    #p_val --> classification values predicted by the system.
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, x_test, m)
    return p_label, p_acc, p_val


def calc_probs(features_dict, feature_vects_neg, feature_vects_pos):
    print 'calculating probabilities...'
    prob_thresh = 0.69
    high_prob_features_pos = []
    high_prob_features_neg = []
    c = 0
    for f, a in features_dict.iteritems():

        c += 1
        if c % 1000 == 0:
            print c

        neg_tweets_containing_f = [v for v in feature_vects_neg if
                                   a in v]#Note: a in v is exactly the same as v.has_key(a)
        pos_tweets_containing_f = [v for v in feature_vects_pos if a in v]
        p = len(pos_tweets_containing_f)
        n = len(neg_tweets_containing_f)
        smoothing_fact = 5

        if n != 0:
            prob_pos_given_f = float(p) / (p + n + smoothing_fact)
            if prob_pos_given_f > prob_thresh:
                high_prob_features_pos.append([f, prob_pos_given_f])
                #print prob_pos_given_f, f

        if p != 0:
            prob_neg_given_f = float(n) / (p + n + smoothing_fact)
            if prob_neg_given_f > prob_thresh:
                high_prob_features_neg.append([f, prob_neg_given_f])
                #print prob_neg_given_f, f

    return high_prob_features_pos, high_prob_features_neg

def get_params(svm_type, kernel_type, cost, nu, balance_sets, labels, training_sizes):
    #labels is a dict in the form of { 'pos':+1, 'neg':-1, 'oth':0}
    if svm_type == 'C_SVC':
        param = '-s 0'
        param = param + ' -c ' + str(cost)
    elif svm_type == 'nu_SVC':
        param = '-s 1'
        param = param + ' -n ' + str(nu)

    param = param + ' -t ' + str(kernel_type)

    #for 3-class classification, we balance the set so that w1*set1 == w2*set2 == w3*set3
    #we can first define w1=1, then w2=len(set1)/len(set2), w3=len(set1)/len(set3) is derived
    if balance_sets:
        first_iter = True
        for k, v in labels.iteritems():
            if first_iter:
                k0 = k
                w = 1
                weights = ' -w' + str(v) + ' ' + str(w)
                first_iter = False
            else:
                w = round(float(training_sizes[k0])/(training_sizes[k]), 2)
                weights = weights + ' -w' + str(v) + ' ' + str(w)

    param = param + weights
    print param

    return param


def find_pos_neg_tweets(keyword, tweets):

    nots = ["never", "don't", "dont", "no", "not", "ain", "ainn", "aint", "ain't", "aren't", "arent", "isn't", "isnt",
            "wasn't", "wasnt", "weren't", "werent", "haven't", "havent", "hasn't", "hasnt", "won't", "wont", "can not",
            "cannot", "couldn't", "couldnt", "shouldn't", "shouldnt", "wouldn't", "wouldnt"]

    verbs = ["be", "been", "get"]

    no_signs = nots + [x + ' ' + y for x in nots for y in verbs]

    adverbs = ['as', 'so', 'so much', 'to', 'too', 'too much', 'very much', 'that much', 'this much', 'completely',
               'totally', 'entirely', 'extremely', 'nobody', 'anybody', 'anyone', 'ever', 'normally', 'really', "even"]

    more_no_signs = [keyword + ' at all'] #, 'stop worrying about']

    no_signs = no_signs + [x + ' ' + y for x in no_signs for y in adverbs]
    no_signs = [x + ' ' + keyword for x in no_signs]
    no_signs = no_signs + more_no_signs

    #select tweets containing negative signs and put them in the negative set.
    positives = tweets[:]
    negatives = []
    for s in no_signs:
        temp = [t for t in positives if s in t]
        negatives = negatives + temp
        positives = [t for t in positives if t not in temp]

    print keyword + ':', len(positives)
    print 'not_' + keyword + ':', len(negatives)

    return positives, negatives

def read_labels_features_from_file(labels_features, tweet_texts, norm_factors, class_labels):

    # create an empty list corresponding to each class_labels
    labels = {}; feature_vects = {}; texts = {}; norms = {}
    for l_k, l_v in class_labels.iteritems():
        labels[l_k] = []
        feature_vects[l_k] = []
        texts[l_k] = []
        norms[l_k] = []

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
            vector = {int(a_v[0]): float(a_v[1]) for a_v in f}

            # iterate through all given labels ('pos', 'neg', 'oth', ...) to put the feature-vectors read from file into
            # the right category. labels is a dict in the form of {'pos' : 1, 'neg' : -1, 'oth' : 0, ...}
            for l_k, l_v in class_labels.iteritems():
                if l == l_v:
                    labels[l_k].append(l)
                    feature_vects[l_k].append(vector)
                    texts[l_k].append(text)
                    norms[l_k].append(n)
    return labels, feature_vects, texts, norms

def get_dimension_size(list_of_vectors):
    # method giving the feature space dimension for training/test set.
    dimensions = []
    # the following loop gathers all dimension numbers in a list:
    for v in list_of_vectors:
        dimensions = dimensions + v.keys()
    # now only keep unique dimensions as vectors share some similar dimensions
    dimensions = set(dimensions)
    return len(dimensions)
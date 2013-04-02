# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

from operator import itemgetter
import re
import my_util
import difflib
import stopwords

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
    #sometimes tokens may be empty --> eg: when tweet is just a url --> we exclude the url and it results an empty list.
    n_of_features = 0
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
                        normal_factor = len(tokens)

        # The following line is performed when the i-loop is finished. This is because, for a given tweet_text, the
        # representing feature-vector must include values of all ngrams extracted from the text.
        if new_normalisation_flag:
            # Divide all elements of the vector by the number of traversed features.
            # This is more correct than the division by len(tokens), as the no. of ngrams (addressed in a vector) is not
            # necessarily equal to the number of features.
            vector = {a : float(c)/n_of_features for a, c in vector.iteritems()}
            normal_factor = n_of_features

    return vector, max_index, normal_factor


def get_sparse_feature_vector_worry(tweet_list, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag):

    feature_vectors = []
    tweet_texts = []
    normal_factors = []
    for tweet in tweet_list:
        vector, max_index, normal_factor = get_ngrams_worry(tweet, features_dict, features_count_dict, max_index, m, n, remove_stpwds_for_unigrams, new_normalisation_flag)
        feature_vectors.append(vector)
        #in general the tweet_texts should be tweet_list itself. We return this in case the order of the vector changes
        # during the loop.
        tweet_texts.append(tweet)
        normal_factors.append(normal_factor)

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
    prediction_result=[]
    header = ['seed', 'reply', 'original_label', 'predicted_label', 'prediction_success']
    prediction_result.append(header)
    true_counts =[0]*len(labels)
    false_counts=[0]*len(labels)
    n_samples = [0]*len(labels)

    for i in range(len(tweet_texts)):
        text = tweet_texts[i]
        original_label = y_test[i]
        predicted_label = int(p_label[i])
        prediction = 'wrong'

        for j in range(len(labels)):
            if original_label == labels[j]:
                n_samples[j] +=1
                if predicted_label == original_label:
                    prediction = 'correct'
                    true_counts[j] +=1
                    break

            if predicted_label == labels[j]:
                false_counts[j] +=1

        prediction_result.append([text, original_label, predicted_label, prediction])

    accuracy = round(float(sum(true_counts)) / len(y_test), 2)

    # do-something if x, else do-something else.
    precisions = [ round(float(true_counts[i]) / (true_counts[i]+false_counts[i]), 2) if
                   (true_counts[i] + false_counts[i]) <>0 else 0 for i in range(len(labels)) ]

    recalls = [ round(float(true_counts[i]) / n_samples[i], 2) if n_samples[i] <> 0 else 0 for i in range(len(labels)) ]

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
    #truncate the beginning and the end of tweets
    truncs = [t[trunc_size : len(t) - trunc_size] for t in truncs if len(t) > 2*trunc_size]
    #stick back the split truncated tweets
    truncs = [' '.join(t) for t in truncs]
    #this dict is used to retrieve original tweets later
    trunc_tweet_dict = dict( zip(truncs, tweets) )

    unique_truncs = remove_duplicate_tweets(truncs, False, None)
    unique_tweets = [trunc_tweet_dict[s] for s in unique_truncs]

    return unique_tweets

def write_labels_and_features_to_csv(labels, features, file_name):
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
    my_util.write_csv_file(file_name, True, True, final_list)

# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

import my_util
import os
from operator import itemgetter
import re
import random
import liblinearutil

#list of URL shorteners.
shorteners = ["t.co", "goo.gl", "img.ly", "bit.ly", "is.gd", "tinyurl.com", "is.gd", "tr.im", "ow.ly", "cli.gs",
              "twurl.nl", "Digg.com", "u.mavrev.com", "tiny.cc", "short.to", "BudURL.com", "snipr.com", "6url.com",
              "snipurl.com", "Just.as", "snurl.com", "su.pr", "kl.am", "adjix.com", "xrl.us", "notlong.com", "short.ie",
              "sn.im", "idek.net", "AltURL.com", "xr.com", "twurl.cc", "Yep.it", "RedirX.com", "to.ly", "fon.gs",
              "x.se", "twirl.at", "ru.ly"]

features_dict = dict()#{'feature':feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
features_dict_reverse = dict()#{'feature_address' : feature} --> used for debugging to visualise features
features_count_dict = dict() #{feature_address : freq_count} --> freq_count: absolute frequ of ngram occurring in token.
max_index = 1#whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]

agreed_pairs=[]#original agreed tweets read line by line from a csv file.
disagreed_pairs=[]#original disagreed tweets read line by line from a csv file.
conv_id_indx = None#column no. containing conv_id which counts seed/reply pairs and is unique for both seed & reply.
tweet_id_indx = None#column no. containing the unique tweet id.
text_indx = None#column number where the tweet texts are.

#flag --> are identical seeds-ngrams to fall in different dimensions than the replies-ngrams or the same dimensions.
#In fact, it simply stores every seed ngram in a different location in the list than the reply ngrams.
separate_features_for_seed_reply = True
#flag --> are training data (features/ngrams) is to be created? or be read from a formerly created & saved csv file?
read_training_data_from_file = True

home_dir = os.path.expanduser('~')
source_dir = home_dir + '/Chatterbox_UCL_Advance/Agreement_Disagreement/'

#csv files containing line by line tweets in the form of seed/reply, seed/reply, ..
agreed_file_name = 'agreed_pairs'
disagreed_file_name = 'disagreed_pairs'

training_file_name = 'labels_and_features'#file saving training set (labels | feature-vectors) that could be read again.
features_file_name = 'features_and_freqs'#file saving features (ngrams) in one column and freq of occurrence in another.
seed_reply_file_name = 'seed_reply_texts'#file saving (seed | reply) texts side by side; to be read for visualisation.
test_result_name = 'test_result'#file saving the actual predicted labels and values on the test set.

def unicode_to_float(the_list, col_nr, strt_indx):
    """
    function changing from unicode to float to make sorting possible. Since tweet_id is very long, converting to it is
    not possible; hence, converting to float.
    """
    for r in the_list[strt_indx:]:
        val = r[col_nr]
        new_val = float(val)
        r[col_nr] = new_val
    return the_list

def make_seed_reply_list(the_list):
    """
    this function creates a list like: [ [seed1, reply1], [seed2, reply2], ... ]
    """
    #sort tweets based on the unique conv_id to have seed_reply pairs sorted.
    the_list = sorted(the_list[1:], key=itemgetter(conv_id_indx))
    #separate seeds from replies
    seeds = the_list[0::2]#start from index 0 and get every other line
    replies  = the_list[1::2]#start from index 1 and get every other line
    seed_reply_list = zip(seeds, replies)

    return seed_reply_list

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

def check_features(f_list):
    #print f_list
    ok = True
    for f in f_list:
        if not(check_feature(f)):
            return False
    return True

def check_feature(f):
    global train_labs, shorteners
    #print train_labs
    if f == "" or f == None:
        return None

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
    if f in train_labs:
        if random.randint(0, 10) == 5:
            return True
        else:
            return False
    if "www" in f:
        return False
    for short in shorteners:
        if short in f:
            return False
    else:
        return True

def add_to_dict(features_dict, t, the_length, vector, features_count_dict):
    global max_index
    try:
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

def ngrams(tweet, is_seed):
    """
    this provides a term-frequency vector
    """
    global max_index, features_dict, features_count_dict

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

    max = 3

    #i --> the length of the token
    #j --> starting index of the token
    for i in range(1, max):
        for j in xrange(0, len(tokens) - (i - 1)):
            if check_features(tokens[j:j + i]):
                t = " ".join(tokens[j:j + i])
                if separate_features_for_seed_reply:
                    if is_seed:
                        t = 's ' + t
                    else:
                        t = 'r ' + t
                #print t
                add_to_dict(features_dict, t, len(tokens), vector, features_count_dict)

                #these extra lines create extra tokens for the beginning and end of tweets.
                if j==0:#first token --> beginning of tweet
                    t = 'beg ' + t
                    add_to_dict(features_dict, t, len(tokens), vector, features_count_dict)
                if j==(len(tokens) - (i - 1) -1):#last token --> end of tweet
                    t = 'end ' + t
                    add_to_dict(features_dict, t, len(tokens), vector, features_count_dict)

    return vector

def get_sparse_feature_vector(seed_reply_list):
    """
    for each pair of tweets in the form of seed_reply, create a dict presenting the aggregate freq of tokens appearing
    in the seed and in the reply
    """
    feature_vectors = []
    seed_reply_texts= []
    for s_r in seed_reply_list:

        ngrams_seed_reply={}
        seed = s_r[0]
        reply= s_r[1]
        seed_text  = seed[ text_indx]
        reply_text = reply[text_indx]
        ngrams_seed  = ngrams( seed_text, True)
        ngrams_reply = ngrams(reply_text, False)

        if separate_features_for_seed_reply:
            #since we have put two separate letters at the beginning of ngrmas for seed and reply tokens,
            #there is no intersection between ngrams_seed and ngrams_reply. So, updating one of them results the entire dict
            ngrams_seed_reply = ngrams_seed.copy()
            ngrams_seed_reply.update(ngrams_reply)
        else:#aggregate the two dicts:
            for k, v in ngrams_seed.iteritems():#go through first dict key values.
                try:#if a key also exists in second dict, add the respective values into the new dict.
                    v2 = ngrams_reply[k]
                    ngrams_seed_reply[k] = v+v2
                except KeyError:#if the key does not exist in second dict, copy key-value of the 1st dict to the new dict
                    ngrams[k] = v
            for k, v in ngrams_reply.iteritems():#go through the elements of the second dict:
                try:
                    ngrams_seed[k]
                except KeyError:#if the second-dict key does not exist in the 1st dict, add it to the new dict.
                    ngrams_seed_reply[k] = v

        feature_vectors.append(ngrams_seed_reply)
        seed_reply_texts.append([seed_text, reply_text])

    return feature_vectors, seed_reply_texts

def make_features_dict_reverse():
    """
    create a reverse dictionary to ba able to call the address of a feature and get the feature itself.
    mainly used for debug purpose.
    """
    global features_dict_reverse
    features = features_dict.keys()
    values = features_dict.values()
    if len(features)==len(set(values)):
        features_dict_reverse = dict(zip(values,features))
    else:
        features_dict_reverse = None

def strip_singles(vectors):
    make_features_dict_reverse()
    global features_count_dict, features_dict_reverse
    out_vectors = []
    for vector in vectors:
        new_vector = dict()
        for k, v in vector.items():
            if features_count_dict[k] > 1:
                new_vector[k] = v
#            else:
#                print 'single_feature:', features_dict_reverse[k]
        out_vectors.append(new_vector)
    return out_vectors

def write_features_and_freqs_to_csv():
    feature_freq_list=[]
    for f, a in features_dict.iteritems():
        c = features_count_dict[a]
        if c>1:
            feature_freq_list.append([f,c])
        else:
            pass
    my_util.write_csv_file(source_dir+features_file_name, False, True, feature_freq_list)


def write_labels_and_features_to_csv(labels, features):
    """
    this function creates a tab deliminator csv file of the labels and features in the form of:
    label dimention_nr1:feature1 dimention_nr2:feature2 ...
    """
    #labels --> [+1,+1,+1...,+1] or [-1,-1,-1,...,-1]
    #features --> [dict1, dict2, dict3, ...]
    #dicts ---> {feature_address1 : feature_freq1, feature_address2 : feature_freq2, ...]
    final_list = []
    if len(labels)==len(features):
        for i in range(0, len(labels)):
            l = labels[i]
            feature_dict = features[i]
            feature_list = [str(k)+":"+str(v) for k,v in feature_dict.iteritems()]
            the_list = [str(l)]+ feature_list
            final_list.append(the_list)
    my_util.write_csv_file(source_dir+training_file_name, True, True, final_list)


def extract_training_and_test_data(y, x, all_seed_reply_texts):
    y_training=[]
    x_training=[]
    y_test=[]
    x_test=[]
    training_texts=[]
    test_texts=[]
    if len(y)==len(x):
        length = len(y)
        for i in range(0, length):
            n = random.randint(0 ,10)
            if n > 2 :#if n>2 --> 70% data for training and 30% for validation
                y_training.append(y[i])
                x_training.append(x[i])
                training_texts.append(all_seed_reply_texts[i])
            else:
                y_test.append(y[i])
                x_test.append(x[i])
                test_texts.append(all_seed_reply_texts[i])
    return y_training, x_training, y_test, x_test, training_texts, test_texts

def read_tweet_files():
    #read the csv files. The first row is the header.
    global agreed_pairs, disagreed_pairs, conv_id_indx, tweet_id_indx, text_indx
    agreed_pairs = my_util.read_csv_file(source_dir + agreed_file_name, True)
    disagreed_pairs = my_util.read_csv_file(source_dir + disagreed_file_name, True)

    header = agreed_pairs[0]

    conv_id_indx = header.index('conv_id')
    tweet_id_indx = header.index('tweet_id')
    text_indx = header.index('text')

    unicode_to_float(agreed_pairs, conv_id_indx, 1)
    unicode_to_float(agreed_pairs, tweet_id_indx, 1)
    unicode_to_float(disagreed_pairs, conv_id_indx, 1)
    unicode_to_float(disagreed_pairs, tweet_id_indx, 1)


def go_train():

    if read_training_data_from_file:
        y, x = liblinearutil.svm_read_problem(source_dir + training_file_name + '.csv')
        all_seed_reply_texts = my_util.read_csv_file(source_dir+seed_reply_file_name, True)
    else:
        seed_reply_agreed = make_seed_reply_list(agreed_pairs)
        seed_reply_disagreed = make_seed_reply_list(disagreed_pairs)

        pos_feature_vectors, pos_seed_reply_texts = get_sparse_feature_vector(seed_reply_agreed)
        neg_feature_vectors, neg_seed_reply_texts = get_sparse_feature_vector(seed_reply_disagreed)

        write_features_and_freqs_to_csv()

        pos_feature_vectors = strip_singles(pos_feature_vectors)
        neg_feature_vectors = strip_singles(neg_feature_vectors)

        y = [1] * len(pos_feature_vectors)
        y = y + [-1] * len(neg_feature_vectors)

        x = pos_feature_vectors + neg_feature_vectors
        all_seed_reply_texts = pos_seed_reply_texts + neg_seed_reply_texts

        write_labels_and_features_to_csv(y, x)
        my_util.write_csv_file(source_dir+seed_reply_file_name, False, True, all_seed_reply_texts)

    y_training, x_training, y_test, x_test, training_texts, test_texts = \
    extract_training_and_test_data(y, x, all_seed_reply_texts)

    prob = liblinearutil.problem(y_training, x_training)
    param = liblinearutil.parameter('-c 4 -B 1')
    m = liblinearutil.train(prob, param)

    test_result = []

    if x_test <>[]:
        #p_labels --> classification labels predicted by the system.
        #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
        #p_val --> classification values predicted by the system.
        p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test, m)
        print '\n test set calssification accuracy:', p_acc[0]

        header = ['seed', 'reply', 'original_label', 'predicted_label', 'predicted_value', 'prediction_success']
        test_result.append(header)
        for i in range(0, len(test_texts)):
            seed_reply_text = test_texts[i]
            seed = seed_reply_text[0]
            reply = seed_reply_text[1]
            original_label=y_test[i]
            predicted_label=int(p_label[i])
            predicted_value=p_val[i]
            if original_label*predicted_label>0:
                prediction = 'correct'
            else:
                prediction = 'wrong'
            test_result.append([seed, reply, original_label, predicted_label, predicted_value, prediction])
    my_util.write_csv_file(source_dir+test_result_name, False, True, test_result)


    #-v n --> n-fold cross validation
#    param_cv = liblinearutil.parameter('-c 1 -B 0 -v 2')
#    m_cv = liblinearutil.train(prob, param_cv)

read_tweet_files()
go_train()


########################################################################################################################
######################################### check how many of a token exists in tweets ###################################
#def count_tokens():
#    c = 0
#    for s_r in seed_reply_agreed:
#        text = s_r[0][text_indx]
#        text = text.split()
#        for t in text:
#            if t == 'a':
#                c = c + 1
#
#    print 'c= ', c
########################################################################################################################
########################################################################################################################
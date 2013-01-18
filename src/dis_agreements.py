# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

import my_util
import os
from operator import itemgetter
import re
import random

shorteners = ["t.co", "goo.gl", "img.ly", "bit.ly", "is.gd", "tinyurl.com", "is.gd", "tr.im", "ow.ly", "cli.gs",
              "twurl.nl", "Digg.com", "u.mavrev.com", "tiny.cc", "short.to", "BudURL.com", "snipr.com", "6url.com",
              "snipurl.com", "Just.as", "snurl.com", "su.pr", "kl.am", "adjix.com", "xrl.us", "notlong.com", "short.ie",
              "sn.im", "idek.net", "AltURL.com", "xr.com", "twurl.cc", "Yep.it", "RedirX.com", "to.ly", "fon.gs",
              "x.se", "twirl.at", "ru.ly"]

filtered_feature_labels = set()
features_dict = dict() #dict storing ratio of ngrams appearing in tokens.
features_count_dict = dict() #dict storing absolute frequency of ngrams occurring in tokens.
max_index = 1
train_labs = []
prob = 1
results = {}
lang = ''
min_data = False
separate_feature_for_seed_reply = True

home_dir = os.path.expanduser('~')
source_dir = home_dir + '/Chatterbox_UCL_Advance/Agreement_Disagreement/'

agreed_file_name = 'agreed_pairs'
disagreed_file_name = 'disagreed_pairs'

#read the csv files. The first row is the header.
agreed_pairs    = my_util.read_csv_file(source_dir + agreed_file_name, True)
disagreed_pairs = my_util.read_csv_file(source_dir + disagreed_file_name, True)

header = agreed_pairs[0]

conv_indx = header.index('conv_id')
tweet_indx = header.index('tweet_id')
text_indx = header.index('text')


def unicode_to_float(the_list, col_nr, strt_indx):
    #function changing from unicode to int to make sorting possible
    for r in the_list[strt_indx:]:
        val = r[col_nr]
        new_val = float(val)
        r[col_nr] = new_val
    return the_list


unicode_to_float(agreed_pairs, conv_indx, 1)
unicode_to_float(agreed_pairs, tweet_indx, 1)
unicode_to_float(disagreed_pairs, conv_indx, 1)
unicode_to_float(disagreed_pairs, tweet_indx, 1)

def make_seed_reply_list(the_list, sort_col_nr, strt_indx, ):
    #this function creates a list like: [ [seed1, reply1], [seed2, reply2], ... ]
    #sort tweets based on conv_id to have seed_reply pairs sorted.
    the_list = sorted(the_list[1:], key=itemgetter(sort_col_nr))
    #separate seeds from replies
    seeds = the_list[0::2]
    replies  = the_list[1::2]
    seed_reply_list = zip(seeds, replies)
    return seed_reply_list


seed_reply_agreed    = make_seed_reply_list(agreed_pairs,    conv_indx, 1)
seed_reply_disagreed = make_seed_reply_list(disagreed_pairs, conv_indx, 1)


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
    print f_list
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


def ngrams(tweet, is_seed):
    global max_index, features_dict, features_count_dict
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

    max = 3

    for i in range(1, max):
        for j in xrange(0, len(tokens) - (i - 1)):
            if check_features(tokens[j:j + i]):
                t = " ".join(tokens[j:j + i])
                if separate_feature_for_seed_reply:
                    if is_seed:
                        t = 's ' + t
                    else:
                        t = 'r ' + t
                print t
                try:
                    indx = features_dict[t]
                except KeyError:
                    features_dict[t] = max_index
                    indx = max_index
                    max_index = max_index + 1
                try:
                    vector[indx] = vector[indx] + (1.0 / len(tokens))
                except KeyError:
                    vector[indx] = (1.0 / len(tokens))
                try:
                    features_count_dict[indx] = features_count_dict[indx] + 1
                except KeyError:
                    features_count_dict[indx] = 1
    return vector

def get_sparse_feature_vector(seed_reply):

    ngrams_seed_reply={}
    ngrams_seed  = ngrams(seed_reply[0][text_indx], True)
    ngrams_reply = ngrams(seed_reply[1][text_indx], False)

    if separate_feature_for_seed_reply:
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

    return ngrams_seed_reply


def build_feature_vectors(seed_reply_list):
    #feature_vectors = [get_sparse_feature_vector(tweet=pt) for pt in tweets]
    feature_vectors = []
    #for each pair of tweets in the form of seed_reply, create a dict presenting the aggregate freq of tokens appearing
    #in the seed and in the reply
    for s_r in seed_reply_list:
        sparse_feature_vector = get_sparse_feature_vector(s_r)
        feature_vectors.append(sparse_feature_vector)
    return feature_vectors

pos_feature_vectors = build_feature_vectors(seed_reply_agreed)
neg_feature_vectors = build_feature_vectors(seed_reply_disagreed)


########################################################################################################################
######################################### check how many of a token exists in tweets ###################################
c = 0
for s_r in seed_reply_agreed:
    text = s_r[0][text_indx]
    text = text.split()
    for t in text:
        if t == 'a':
            c = c + 1

print 'c= ', c
########################################################################################################################
########################################################################################################################

print ""
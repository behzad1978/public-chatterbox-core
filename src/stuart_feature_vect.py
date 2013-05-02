# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'
import random, re

features_dict = dict() #dict storing ratio of ngrams appearing in tokens.
features_count_dict = dict() #dict storing absolute frequency of ngrams occurring in tokens.
max_index = 1
train_labs = []

def set_random(seed_val):
    random.seed(seed_val)

def set_train_labs(the_lab):
    global train_labs
    train_labs = the_lab

shorteners = ["t.co","goo.gl","img.ly","bit.ly","is.gd","tinyurl.com","is.gd","tr.im","ow.ly","cli.gs","twurl.nl","Digg.com","u.mavrev.com","tiny.cc","short.to","BudURL.com","snipr.com","6url.com","snipurl.com","Just.as","snurl.com","su.pr","kl.am","adjix.com","xrl.us","notlong.com","short.ie","sn.im","idek.net","AltURL.com","xr.com","twurl.cc","Yep.it","RedirX.com","to.ly","fon.gs","x.se","twirl.at","ru.ly"]

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
        if random.randint(1,10) == 5:#random.randint(0,10) == 5:
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

def ngrams(tweet):

    global max_index, features_dict, features_count_dict
    """
        this provides a term-frequency vector
    """

    vector = dict()
    #print tweet
    tweet_text = tweet.lower().split()
    no_url_tweet_text = []
    #drop the URLs
    for t in tweet_text:
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

    tweet_text = " ".join(no_url_tweet_text)
    #print tweet_text
    ##put a space between any non punct char and a punct char
    tweet_text = re.sub(r"([^'\".,;:/?\!@#£$%^&*()_\-=+`~])(['\".,;:/?\!@#£$%^&*()_\-=+`~])", r"\1 \2",tweet_text)
    #print tweet_text
    ##put a space between any punct char and a non punct char
    tweet_text = re.sub(r"(['\".,;:/?\!£$%^&*()_\-=+`~])([^'\".,;:/?\!@#£$%^&*()_\-=+`~#@])", r"\1 \2",tweet_text)
    #print tweet_text
    #stick heart symbols back together
    tweet_text = re.sub(" < 3 "," <3 ",tweet_text)
    tweet_text = re.sub(" : d "," :d ",tweet_text)
    tweet_text = re.sub(" : p "," :p ",tweet_text)


    #print tweet_text
    #stick n't back together
    tweet_text = re.sub(r"(\w)n ' t\b",r"\1 n't",tweet_text)
    #print tweet_text
    tweet_text = re.sub(r" ([:;][-]?) ([DP]) ",r"\1\2",tweet_text)
    #print tweet_text
    tweet_text = re.sub(r"(\S)\1\1+",r"\1\1\1",tweet_text)

    tokens = tweet_text.split()

    max = 3

    for i in range(1,max):
        for j in xrange(0,len(tokens)-(i-1)):
            if check_features(tokens[j:j+i]):
                t = " ".join(tokens[j:j+i])
                #print t
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

def check_features(f_list):
    ok = True
    for f in f_list:
        if not(check_feature(f)):
            return False
    return True

def get_sparse_feature_vector(tweet):
    return ngrams(tweet)

def build_feature_vectors(tweets):
    feature_vectors = [get_sparse_feature_vector(tweet = pt) for pt in tweets]
    return feature_vectors

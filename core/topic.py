"""
    Module for performing Chatterbox Topic Analysis
    Copyright Chatterbox Analytics Ltd 2012
"""
import operator
import math
import re
import stopwords as stops
import os
import difflib
import string
import logging
from datetime import timedelta, datetime
    
def ngrams(tokens,lang):
    """
        produces all the word n-grams for
        a given list of tokens.
        Requires a language parameter to use for stopwords
        
        Example: ngrams("i like cat on toast in the sunny morning".split(),'en')
    """
    stopwords = stops.stopwords[lang]
    max = len(tokens)
    ngrams = []
    left_punctuation = '!"%&\'()*+,-./:;<=>?[\\]^_`{|}~'

    for i in range(1,max):
        for j in xrange(0,len(tokens)-(i-1)):
            if __check_features(tokens[j:j+i],stopwords):
               ng_str = " ".join(tokens[j:j+i])
               ng_str = (ng_str.rstrip(string.punctuation)).lstrip(left_punctuation)                   
               ngrams.append(ng_str)
               
    ng_str = " ".join(tokens)
    ng_str = (ng_str.rstrip(string.punctuation)).lstrip(left_punctuation)                   
    ngrams.append(ng_str)
    return ngrams


def weight_term_frequencies_one(a_in,b_in):
    """
        Produce a weighted ranking of phrases in a_in, 
        weighted by the ranking of phrases in b_in
    """
    plus_value = 2
    a_out = {}
    for(term, val) in a_in.items():
        try:
            b_val = b_in[term]
            a_out[term] = float(val)/float(val+b_val+plus_value)
        except KeyError:
            a_out[term] = float(val)/(float(val + plus_value))   
    return a_out 
   
def weight_term_frequencies(a_in,b_in):
    """
        Produce a weighted ranking of phrases in a_in, 
        weighted by the ranking of phrases in b_in
        and
        Produce a weighted ranking of phrases in b_in, 
        weighted by the ranking of phrases in a_in        
    """
    plus_value = 2
    a_out = {}
    b_out = {}   
    #print "in weight term frequencies" 
    for(term, val) in a_in.items():
        try:
            b_val = b_in[term]
            a_out[term] = float(val)/float(val+b_val+plus_value)
        except KeyError:
            a_out[term] = float(val)/(float(val + plus_value)) 
    
    for(term, val) in b_in.items():
        try:
            a_val = a_in[term]
            b_out[term] = float(val)/float(val+a_val+plus_value)
        except KeyError:
            b_out[term] = float(val)/(float(val + plus_value))  
             
    return (a_out,b_out)

def __check_feature(f,stopwords):
    """
        For a given feature, checks if it is 
        a stopword or not.  Do not pass the check
        if it is.
    """
    if f == "" or f == None:
        return None
    if f == "RT":
        return False
    if f == "via":
        return False
    if len(re.findall(r"(\w)", f)) < 1:
        return False
    if f == "&amp":
        return False
    if f in stopwords:
        return False
    if len(f) < 2:
        return False
    else:
        return True

def __check_features(f_list, stopwords):
    """
        For a given list of features, checks if they are
        stopwords or not.  Do not pass the check
        if they are.
    """
    ok = True
    for f in f_list:
        if not(__check_feature(f,stopwords)):
            return False
    return True

def _analyse_topics(frequencies):
    """
        Private function that requires a frequency distribution
    """
    freq = frequencies[0]
    freq_ref = frequencies[1]
    the_dict = weight_term_frequencies_one(freq,freq_ref)
    sorted_toks = sorted(the_dict.iteritems(),
                        key=operator.itemgetter(1))
    
    sorted_toks.reverse()
    sorted_toks = sorted_toks[:400]
    final_toks = []
    for (k,v) in sorted_toks:
        best = True
        for (k1,v1) in sorted_toks:
            if k != k1:
                if (abs(v1-v)) < 0.2:
                    if k in k1:
                        best = False
                        #print "dropped", k
        if best:
            final_toks.append((k,v))
            
    very_final_toks = {}
    for (k,v) in final_toks:
        close_val = [(k2,v2) for k2,v2 in final_toks[:50] if abs(v-v2) < 0.2]
        if len(close_val) < 1:
            very_final_toks[k] = v
        else:
            similar = [(k3,v3,len(k3)) for k3,v3 in close_val if difflib.SequenceMatcher(None,k,k3).quick_ratio() > 0.89]
            if len(similar) > 1:
                a,b,c = sorted(similar,key=operator.itemgetter(2))[0]
                very_final_toks[a] = b
            else:
                very_final_toks[k] = v
    
    very_final_toks = sorted(very_final_toks.iteritems(),
                             key=operator.itemgetter(1))
    very_final_toks.reverse()
    return very_final_toks

def get_pos_ranked_topics(data, exclude, lang):
    """
        Produce a ranked list of the top 30 positive phrases
        offset by the negative tokens.
        
        Return as a dictionary
    """
    pos_tokens = []
    neg_tokens = []
    for k, v in data.items():
        if abs(v['value']) > 0.2:#exclude low sentiment messages
            final_tokens = v['ngrams']
            if v['sent'] == -1:
                neg_tokens = neg_tokens + [final_tokens]
    
            else:
                pos_tokens = pos_tokens + [final_tokens]
    pos = term_frequency(pos_tokens,lang)
    neg = term_frequency(neg_tokens,lang)
    final_pos = _analyse_topics([pos,neg])
    return {'pos':final_pos[:30]}

def get_neg_ranked_topics(data, exclude, lang):
    """
        Produce a ranked list of the top 30 negative phrases
        offset by the negative tokens.
        
        Return as a dictionary
    """
    pos_tokens = []
    neg_tokens = []
    for k, v in data.items():
        if abs(v['value']) > 0.2:#exclude low sentiment tweets
            final_tokens = v['ngrams']
    
            if v['sent'] == -1:
                neg_tokens = neg_tokens + [final_tokens]
    
            else:
                pos_tokens = pos_tokens + [final_tokens]
    
    pos = term_frequency(pos_tokens,lang)
    neg = term_frequency(neg_tokens,lang)
    final_neg = _analyse_topics([neg,pos])
    return {'neg':final_neg[:30]}


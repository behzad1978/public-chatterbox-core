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
import nltk
import stringcomp
import math_extra
from datetime import timedelta, datetime
    
def ngrams(tokens,lang, stop_word_flag):
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
            if __check_features(tokens[j:j+i],stopwords, stop_word_flag):
               ng_str = " ".join(tokens[j:j+i])
               ng_str = (ng_str.rstrip(string.punctuation)).lstrip(left_punctuation)                   
               ngrams.append(ng_str)
               
    ng_str = " ".join(tokens)
    ng_str = (ng_str.rstrip(string.punctuation)).lstrip(left_punctuation)                   
    ngrams.append(ng_str)
    return ngrams

def __check_features(f_list, stopwords, stop_word_flag):
    """
        For a given list of features, checks if they are
        stopwords or not.  Do not pass the check
        if they are. If stop_word_flag is false do not check for stopwords.
    """
    ok = True
    for f in f_list:
        if not(__check_feature(f,stopwords, stop_word_flag)):
            return False
    return True

def __check_feature(f,stopwords, stop_word_flag):
    """
        For a given feature, checks if it is
        a stopword or not.  Do not pass the check
        if it is. If stop_word_flag is false do not check for stopwords and return True.
    """
    if f == "" or f == None:
        return None
    if f == "RT":
        return False
    if f == "via":
        return False
    if len(re.findall(r"(\w)", f)) < 1:  # find all regular expressions (not brackets, paranthesis, ...) and if there are all white return false! #eg: re.findall(r"(\w)", 'it is cold') --> ['i', 't', 'i', 's', 'c', 'o', 'l', 'd']
        return False
    if f == "&amp":
        return False
    if stop_word_flag:
        if f in stopwords:
            return False
    if len(f) < 2:
        return False
    else:
        return True

def term_frequency(ngrams,lang):
    """
        Produces the term frequencies for each ngram
        in a list of ngrams

        Could we do this offline.. save cycles?
    """
    token_dictionary = {}
    for ng in ngrams:
        try:
            token_dictionary[ng] = token_dictionary[ng] + 1
        except KeyError:
            token_dictionary[ng] = 1
    return token_dictionary

def weight_term_frequencies_one(a_in, b_in, plus_value):
    """
        Produce a weighted ranking of phrases in a_in, 
        weighted by the ranking of phrases in b_in
    """
    #plus_value = 2
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


def calc_probs(a_in, b_in, c_in, plus_val):
    # prob = p/(p+n+g)
    probs = {}
    for(tok_a, val_a) in a_in.items():
        try:
            val_b = b_in[tok_a]
        except KeyError:
            val_b=0
        try:
            val_c = c_in[tok_a]
        except KeyError:
            val_c=0
        probs[tok_a] = float(val_a) / float(val_a + val_b + val_c + plus_val)
    return probs





def find_encompassing_toks(weighted_toks, weight_similarity_val):
    """
    Within those ngrams that have close weights (eg: <0.2) keep those that are most encompassing! For instance if
    'nice weather' has a probability (weight) of 0.6 and 'nice weather today' has a probability of 0.65, select the
     latter and discard the former as the latter contains the former!
    """
    encompasing_toks = []
    for (k,v) in weighted_toks:
        best = True
        for (k1, v1) in weighted_toks:
            if k != k1:
                if (abs(v1 - v)) < weight_similarity_val: # 0.2:
                    if k in k1:
                        best = False
                        #print "dropped", k
        if best:
            encompasing_toks.append((k, v))
    return encompasing_toks


def find_toks_by_clustering(weighted_toks, cluster_ids, str_comp_type, string_sim_val, str_length_flag):
    """
    Within those ngrams that are in the same cluster (i.e. have close distance) keep those that are most encompassing!
    For instance if 'nice weather' has a probability (weight) of 0.6 and 'nice weather today' has a probability of 0.65,
     select the latter and discard the former as the latter contains the former!
    """
    final_list = []

    all_id_indices = []
    unique_cluster_ids = set(cluster_ids)
    for id in unique_cluster_ids:
        same_id_indices = [i for i, x in enumerate(cluster_ids) if x == id]
        all_id_indices.append(same_id_indices)

    all_closeWeight_and_encompass = []
    for same_id_indices in all_id_indices:
        close_weights = [weighted_toks[indx] for indx in same_id_indices]
        closeWeights_and_encompass = []
        for t, w in close_weights:
            encompassing = [(t1, w1) for t1, w1 in close_weights if t in t1]
            if len(encompassing) == 1:
                closeWeights_and_encompass.append((t, w))
        all_closeWeight_and_encompass.append(closeWeights_and_encompass)

    all_closeWeight_and_encompass_and_similar=[]
    for closeWeight_and_encompass in all_closeWeight_and_encompass:
        similars = []
        selected_elements = []
        for k, v in closeWeight_and_encompass:
            if (k, v) not in selected_elements:

                if str_comp_type == 'quick_ratio':
                    similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if
                                difflib.SequenceMatcher(None, k.lower(), k1.lower()).quick_ratio() > string_sim_val] # 0.89]
                if str_comp_type == 'ratio':
                    similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if
                                difflib.SequenceMatcher(None, k, k1).ratio() > string_sim_val]
                if str_comp_type == 'Levenshtein':
                    # we suppose len(k)+len(k3)<>0, otherwise 1 must be returned.
                    similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if(float(
                        len(k) + len(k1) - nltk.edit_distance(k, k1)) / float(len(k) + len(k1))) > string_sim_val]
                if str_comp_type == 'stringComp':
                    similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if
                                stringcomp.stringcomp(k, k1) > string_sim_val]

                    #sort on token-length in ascending order
                similars = sorted(similars, key=operator.itemgetter(2))
                if str_length_flag == 'longest':
                    similars.reverse()

                all_closeWeight_and_encompass_and_similar.append(similars)

                selected_elements = selected_elements + [(k, v) for k, v, l in similars]

                a, b, c = similars[0]
                final_list.append([a, b, c])

    return final_list


#def find_toks_by_clustering(weighted_toks, cluster_ids, str_comp_type, string_sim_val, str_length_flag):
#    """
#    Within those ngrams that are in the same cluster (i.e. have close distance) keep those that are most encompassing!
#    For instance if 'nice weather' has a probability (weight) of 0.6 and 'nice weather today' has a probability of 0.65,
#     select the latter and discard the former as the latter contains the former!
#    """
#    final_list = []
#
#    all_indices=[]
#    for i in range(0, len(cluster_ids)):
#        same_id_indices = []
#        id_1 = cluster_ids[i]
#        if id_1 <> -1:
#            same_id_indices = same_id_indices + [i]
#            cluster_ids[i] = -1
#            for j in range(i, len(cluster_ids)-1):
#                id_2 = cluster_ids[j+1]
#                if id_1 == id_2:
#                    same_id_indices = same_id_indices + [j+1]
#                    cluster_ids[j+1] = -1
#            all_indices.append(same_id_indices)
#
#    all_indices_closeWeight_and_encompass = []
#    for same_id_indices in all_indices:
#        indices_closeWeight_and_encompass = same_id_indices[:]#clone the list
#        for i in range(0, len(same_id_indices)):
#            indx = same_id_indices[i]
#            k, v = weighted_toks[indx]
#            for j in range(0, len(same_id_indices)):
#                indx2 = same_id_indices[j]
#                k1, v1 = weighted_toks[indx2]
#                if k <> k1:
#                    if k in k1:#k1 is encompassing
#                        indices_closeWeight_and_encompass.remove(indx)
#                        break
#        all_indices_closeWeight_and_encompass.append(indices_closeWeight_and_encompass)
#
#    all_closeWeight_and_encompass = []
#    for indices_closeWeight_and_encompass in all_indices_closeWeight_and_encompass:
#        closeWeight_and_encompass = [weighted_toks[indx] for indx in indices_closeWeight_and_encompass]
#        #mean, stdev = math_extra.calc_mean_stdev([v for k, v in closeWeight_and_encompass])
#        all_closeWeight_and_encompass.append((closeWeight_and_encompass))#, mean))
#    #all_closeWeight_and_encompass = sorted(all_closeWeight_and_encompass, key=operator.itemgetter(1))
#    #all_closeWeight_and_encompass = [x for x, mean in all_closeWeight_and_encompass]
#
#    for closeWeight_and_encompass in all_closeWeight_and_encompass:
#        similars = []
#        while (len(closeWeight_and_encompass)) > 0:
#            k, v = closeWeight_and_encompass[0]
#
#            if str_comp_type == 'quick_ratio':
#                similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if
#                           difflib.SequenceMatcher(None, k, k1).quick_ratio() > string_sim_val] # 0.89]
#            if str_comp_type == 'ratio':
#                similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if
#                           difflib.SequenceMatcher(None, k, k1).ratio() > string_sim_val]
#            if str_comp_type == 'Levenshtein':
#                # we suppose len(k)+len(k3)<>0, otherwise 1 must be returned.
#                similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if(float(
#                            len(k) + len(k1) - nltk.edit_distance(k, k1)) / float(len(k) + len(k1))) > string_sim_val]
#            if str_comp_type == 'stringComp':
#                similars = [(k1, v1, len(k1)) for k1, v1 in closeWeight_and_encompass if stringcomp.stringcomp(k, k1) > string_sim_val]
#            #sort on token-length in ascending order
#            similars = sorted(similars, key=operator.itemgetter(2))
#            if str_length_flag == 'longest':
#                similars.reverse()
#            a, b, c = similars[0]
#            final_list.append([a, b, c])
#
#            similars = [(k, v) for (k, v, l) in similars]
#            closeWeight_and_encompass = [x for x in closeWeight_and_encompass if x not in similars]
#
#    return final_list

def find_encompassing_toks2(weighted_toks, prob_close_val): #, stdev, sigma_mult):
    """
    Within those ngrams that have close weights (eg: <0.2) keep those that are most encompassing! For instance if
    'nice weather' has a probability (weight) of 0.6 and 'nice weather today' has a probability of 0.65, select the
     latter and discard the former as the latter contains the former!
    """
    #computing diff is very expensive, so we use the stdev of the weights as a substitution for stdev of diff. We can
    #justify that on average points have a distance of stdev from the mean. Hence, on average their distance from each
    #other (i.e. mean of diff) is equal to stdev.
#    diff = [abs(v1 - v2) for k1, v1 in weighted_toks for k2, v2 in weighted_toks if k1 <> k2]
#    real_mean, real_stdev = math_extra.calc_mean_stdev(diff)

    encompasing_toks = []
    for (k, v) in weighted_toks:
        best = True
        for (k1, v1) in weighted_toks:
            if k != k1:
                if (abs(v1 - v)) < prob_close_val: #(stdev*sigma_mult):
                    if k in k1:
                        best = False
                        #print "dropped", k
        if best:
            encompasing_toks.append((k, v))
    return encompasing_toks


def find_similar_toks(toks, pickup_no_2, str_comp_type, string_sim_val, prob_close_val, str_length_flag):
    very_final_toks = {}
    for (k, v) in toks:
    #close_val = [(k2,v2) for k2,v2 in final_toks[:50] if abs(v-v2) < 0.2]
        close_val = [(k2, v2) for k2, v2 in toks[:pickup_no_2] if abs(v-v2) < prob_close_val]
        if len(close_val) < 1:
            very_final_toks[k] = v
        else:
            similar = []
            if str_comp_type == 'quick_ratio':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if difflib.SequenceMatcher(None, k, k3).quick_ratio() > string_sim_val] # 0.89]
            if str_comp_type == 'ratio':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if difflib.SequenceMatcher(None, k, k3).ratio() > string_sim_val]
            if str_comp_type == 'Levenshtein':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if (float(len(k)+len(k3)-nltk.edit_distance(k, k3))/float(len(k)+len(k3))) > string_sim_val] # we suppose len(k)+len(k3)<>0, otherwise 1 must be returned.
            if str_comp_type == 'stringComp':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if stringcomp.stringcomp(k, k3) > string_sim_val]
            if len(similar) > 1:
                if str_length_flag == 'longest':
                    sort_list = sorted(similar, key=operator.itemgetter(2))
                    sort_list.reverse()
                    a, b, c = sort_list[0]
                else:
                    a, b, c = sorted(similar, key=operator.itemgetter(2))[0] # sorts on len(k3) in ascending order. Prefers shortest strings!?!?!?!
                very_final_toks[a] = b
            else:
                very_final_toks[k] = v
    return very_final_toks


def find_similar_toks2(toks, str_comp_type, string_sim_val, stdev):
    very_final_toks = {}
    for (k, v) in toks:
    #computing diff is very expensive, so we use the stdev of the weights as a substitution for stdev of diff. We can
    #justify that on average points have a distance of stdev from the mean. Hence, on average their distance from each
    #other (i.e. mean of diff) is equal to stdev.
    #        diff_val = [(k2, v2, abs(v-v2)) for k2, v2 in toks if k<>k2]
    #        diff_val_mean, diff_val_stdev = math_extra.calc_mean_stdev([d[2] for d in diff_val])
    #        close_val = [(k2, v2) for k2, v2, d in diff_val if d < (diff_val_mean-diff_val_stdev/2)]
        close_val = [(k2, v2) for k2, v2 in toks if abs(v - v2) < (stdev/float(2))]
        if len(close_val) < 1:
            very_final_toks[k] = v
        else:
            similar = []
            if str_comp_type == 'quick_ratio':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if
                           difflib.SequenceMatcher(None, k, k3).quick_ratio() > string_sim_val]
            if str_comp_type == 'ratio':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if
                           difflib.SequenceMatcher(None, k, k3).ratio() > string_sim_val]
            if str_comp_type == 'Levenshtein':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if(float(
                    len(k) + len(k3) - nltk.edit_distance(k, k3)) / float(
                    len(k) + len(k3))) > string_sim_val] # we suppose len(k)+len(k3)<>0, otherwise 1 must be returned.
            if str_comp_type == 'stringComp':
                similar = [(k3, v3, len(k3)) for k3, v3 in close_val if stringcomp.stringcomp(k, k3) > string_sim_val]
            if len(similar) > 1:
                # sorts on len(k3) in ascending order. Prefers shortest strings!?!?!?!
                #a, b, c = sorted(similar, key=operator.itemgetter(2))[0]

                sort_list = sorted(similar, key=operator.itemgetter(2))
                sort_list.reverse()
                a, b, c =sort_list[0]

                very_final_toks[a] = b
            else:
                very_final_toks[k] = v
    return very_final_toks

#def analyse_topics(sorted_toks, pickup_no_1, pickup_no_2, weight_similarity_val, quick_ratio_val):
##def _analyse_topics(frequencies):
#    """
#        Private function that requires two frequency distributions
#    """
#    # freq = frequencies[0]
#    # freq_ref = frequencies[1]
#    # the_dict = weight_term_frequencies_one(freq,freq_ref)
#    # sorted_toks = sorted(the_dict.iteritems(),
#    #                     key=operator.itemgetter(1))
#    #sorted_toks.reverse()
#
#    #sorted_toks = sorted_toks[:400]
#    #sorted_toks = sorted_toks[:pickup_no_1]
#    final_toks = find_encompassing_toks(sorted_toks, weight_similarity_val)
#
#    very_final_toks = find_longest_similar_toks(final_toks, pickup_no_2, quick_ratio_val, weight_similarity_val)
#
#    very_final_toks = sorted(very_final_toks.iteritems(),
#                             key=operator.itemgetter(1))
#    very_final_toks.reverse()
#    return very_final_toks
#
#
#
#def get_offset_ranked_topics(test_ngrams, offset_ngrams, exclude, lang):
#    """
#        Produce a ranked list of the top 30 phrases from test_ngrams
#        offset by the offset_ngrams.
#
#        Return as a dictionary
#
#        Example:
#        p = topic.ngrams("i like cat on toast in the sunny morning".split(),'en')
#        p1 = topic.ngrams("i like dog on toast in the sunny morning dog dog dog".split(),'en')
#        p1 = p1 + topic.ngrams("i like dog on toast in the sunny morning".split(),'en')
#        topic.get_offset_ranked_topics(p1,p,[],'en')
#    """
#
#    test = term_frequency(test_ngrams,lang)
#    offset = term_frequency(offset_ngrams,lang)
#    final_test = _analyse_topics([test,offset])
#    return {'ranked_testngrams':final_test[:30]}
#
#def get_neg_ranked_topics(neg_ngrams, pos_ngrams, exclude, lang):
#    """
#        Produce a ranked list of the top 30 negative phrases
#        offset by the negative tokens.
#
#        Return as a dictionary
#    """
#    pos_tokens = []
#    neg_tokens = []
#    for png in pos_ngrams:
#        pos_tokens = pos_tokens + [png]
#    for nng in neg_ngrams:
#        neg_tokens = neg_tokens + [nng]
#    pos = term_frequency(pos_tokens,lang)
#    neg = term_frequency(neg_tokens,lang)
#    final_neg = _analyse_topics([neg,pos])
#    return {'neg':final_neg[:30]}

# t1 = "i like cat on toast in the sunny morning"
# t2 = "i like dog on toast in the sunny morning dog dog dog"
# lan='en'

# ngrams_t1=ngrams(t1.split(), lan)
# ngrams_t2=ngrams(t2.split(), lan)
# ngrams_t1_2=ngrams_t1+ngrams_t2

# freq_t1=term_frequency(ngrams_t1, lan)
# freq_t2=term_frequency(ngrams_t2, lan)
# freq_t1_2=term_frequency(ngrams_t1_2, lan)

# weight_freq = weight_term_frequencies_one(freq_t1_2, freq_t1)

# sorted_weight_freq = sorted(weight_freq.iteritems(), key=operator.itemgetter(1))
# sorted_weight_freq.reverse() #large values first

# print'ngrams_t1 = '
# print ngrams_t1
# print'ngrams_t2 ='
# print ngrams_t2
# print'ngrams_t1_2 ='
# print ngrams_t1_2

# print'freq_t1 =' 
# print freq_t1
# print'freq_t2 =' 
# print freq_t2
# print'freq_t1_2 =' 
# print freq_t1_2

# print 'weight_freq ='
# print weight_freq

# print 'sorted_weight_freq ='
# print sorted_weight_freq

# long_tokens=[]
# for (k,v) in sorted_weight_freq:

#         print '(lookup_key, lookup_val):', '(', k, ', ', v, ')'

#         best = True
#         for (k1,v1) in sorted_weight_freq:
            
#             #print 'innerLoop:', k1

#             if k != k1:
#                 if (abs(v1-v)) < 0.2:

#                     #print 'close-val pair:', '(', k1, ', ', v1, ')'

#                     if k in k1:
#                         best = False
#                         #print "dropped:", k
#         if best:
#             #print '(best_key, best_val):', '(', k, ', ', v, ')' 
#             long_tokens.append((k,v))

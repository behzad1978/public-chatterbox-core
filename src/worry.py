__author__ = 'behzadbehzadan'

import my_util
import os
import funcs
import operator
import string

source_dir = os.path.expanduser('~')
file_dir = '/Chatterbox_UCL_Advance/Worry/'
collection_name = 'worried'
source_file = 'source' + '_' + collection_name
source_file_noDup = source_file + '_noDup'

########################################################################################################################
remove_retweets = True
use_qr_to_remove_dups = False
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
        #remove extra spaces that may exist between words, by first splitting the words and then re-joining them.
        tweets = [' '.join(t.split())]
        #remove duplicates by direct comparison of strings
        tweets = funcs.remove_duplicate_tweets(tweets, False, None)
        #remove duplicates by direct comparison of the truncated strings
        tweets = funcs.truncate_and_remove_duplicates(tweets, 4)
        if use_qr_to_remove_dups:
            tweets = funcs.remove_duplicate_tweets(tweets, True, 0.89)
        my_util.write_csv_file(source_dir + file_dir + source_file_noDup, False, True, [[t] for t in tweets])

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

#{'feature':feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
features_dict_pos = dict()
features_dict_neg = dict()
#{'feature_address (dimension no) : feature} --> used for debugging to visualise features
features_dict_reverse_pos = dict()
features_dict_reverse_neg = dict()
#{feature_address (dimension no.) : freq_count} --> freq_count: absolute freq of ngram occurring in token.
features_count_dict_pos = dict()
features_count_dict_neg = dict()
#whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]
max_index_pos = 1
max_index_neg = 1

def create_feature_Vector(tweets, features_dict, features_count_dict, max_index, m, n):
    #m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
    #length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
    vectors = []
    for t in tweets:
        vector, tokens_length, max_index = funcs.get_ngrams_worry(t, features_dict, features_count_dict, max_index, m, n)
        vectors.append([vector, tokens_length])
    return vectors

print 'creating feature vectors...'
vectors_pos = create_feature_Vector(positives, features_dict_pos, features_count_dict_pos, max_index_pos, 1, 1)
vectors_neg = create_feature_Vector(negatives, features_dict_neg, features_count_dict_neg, max_index_neg, 1, 1)
print 'finished creating feature vectors!'

visualising_thresh = 50
funcs.write_features_and_freqs_to_csv(features_dict_pos, features_count_dict_pos, visualising_thresh, source_dir + file_dir + collection_name + "_count_pos")
funcs.write_features_and_freqs_to_csv(features_dict_neg, features_count_dict_neg, visualising_thresh, source_dir + file_dir + collection_name + "_count_neg")

def find_double_keywords(list):
    worried_worried = []
    for t in list:
        t = t.split()
        keyword = collection_name
        count = len([x for x in t if x==keyword])
        if count>=2:
            t = ' '.join(t)
            worried_worried.append([t])
    return worried_worried


keyword_keyword_pos = find_double_keywords(positives)
keyword_keyword_neg = find_double_keywords(negatives)
my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'pos', False, True, keyword_keyword_pos)
my_util.write_csv_file(source_dir + file_dir + collection_name + '_' + collection_name + '_' + 'neg', False, True, keyword_keyword_neg)

all_features = list((set(features_dict_neg.keys())).union(set(features_dict_pos.keys())))
print 'no of features:', len(all_features)

prob_thresh = 0.50
high_prob_features_pos=[]
high_prob_features_neg=[]
print 'calculating probabilities...'
c=0
for f in all_features:

    c+=1
    m = operator.mod(c, 1000)
    if m==0:
        print c

    neg_tweets_containing_f = []
    if f in features_dict_neg:
        a = features_dict_neg[f]
        neg_tweets_containing_f = [v for v, l in vectors_neg if a in v]#Note: a in v is exactly the same as v.has_key(a)

    pos_tweets_containing_f = []
    if f in features_dict_pos:
        a = features_dict_pos[f]
        pos_tweets_containing_f = [v for v, l in vectors_pos if a in v]

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


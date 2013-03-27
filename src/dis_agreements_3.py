# -*- coding: utf-8 -*-
__author__ = 'behzadbehzadan'

"""
In this version agreed pairs are trained versus both of the disagreed pairs and followups. So agreed pairs fall in
class A whereas disagreed pairs and other pairs fall class B.
"""

import my_util
import os
from operator import itemgetter
import re
import random
random.seed(7)
import liblinearutil
import math_extra
import svmutil

#list of URL shorteners.
shorteners = ["t.co", "goo.gl", "img.ly", "bit.ly", "is.gd", "tinyurl.com", "is.gd", "tr.im", "ow.ly", "cli.gs",
              "twurl.nl", "Digg.com", "u.mavrev.com", "tiny.cc", "short.to", "BudURL.com", "snipr.com", "6url.com",
              "snipurl.com", "Just.as", "snurl.com", "su.pr", "kl.am", "adjix.com", "xrl.us", "notlong.com", "short.ie",
              "sn.im", "idek.net", "AltURL.com", "xr.com", "twurl.cc", "Yep.it", "RedirX.com", "to.ly", "fon.gs",
              "x.se", "twirl.at", "ru.ly"]

features_dict = dict()#{'feature':feature_address} --> feature is an ngrmam, address is a number referring to the ngram.
features_dict_reverse = dict()#{'feature_address (dimension no)' : feature} --> used for debugging to visualise features
features_count_dict = dict() #{feature_address (dimension no.) : freq_count} --> freq_count: absolute frequ of ngram occurring in token.
max_index = 1#whenever a new ngram is created --> max_index++ --> the ngram is stored in features_dict[max_index]
train_labs=[]


#flag --> are identical seeds-ngrams to fall in different dimensions than the replies-ngrams or the same dimensions.
#In fact, it simply stores every seed ngram in a different location in the list than the reply ngrams.
separate_features_for_seed_reply = True
#flag --> is training data (features/ngrams) to be created? or read from a formerly created & saved csv file?
read_training_data_from_file = False
equalise_pos_neg_sets = False
strip_singles_flag = False
classifier_labels = [+1, -1, +1]#[+1, -1, -1]#[agreed, disagreed, others]

home_dir = os.path.expanduser('~')
source_dir = home_dir + '/Chatterbox_UCL_Advance/Agreement_Disagreement/convs_123/'

#csv files containing line by line tweets in the form of seed/reply, seed/reply, ..
agreed_file_name = 'agreed_pairs'
positive_answers_file_name = 'positive_answers'
supportive_pairs_file_name = 'supportives'
disagreed_file_name = 'disagreed_pairs'
negative_answers_file_name = 'negative_answers'
offensive_sarcastic_tweets = 'offensives'
followups_file_name = 'followups'

training_file_name = 'labels_and_features'#file saving training set (labels | feature-vectors) that could be read again.
features_file_name = 'features_and_freqs'#file saving features (ngrams) in one column and freq of occurrence in another.
seed_reply_file_name = 'seed_reply_texts'#file saving (seed | reply) texts side by side; to be read for visualisation.
test_result_name = 'Results3/test_set'#file saving the actual predicted labels and values on the test set.
tabel_result_name = 'Results3/tabel'

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

def code_0_1_for_seed_reply(the_list, col_nr, strt_indx):
    """
    function changing the seed reply stickers (S/R) to 0/1 so that data can be sorted on that column too..
    """
    for r in the_list[strt_indx:]:
        val = r[col_nr]
        if val == 'S':
            new_val = 0
        else:
            new_val=1
        r[col_nr] = new_val
    return the_list

def make_seed_reply_list(the_list, conv_id_indx, s_r_indx):
    """
    this function creates a list like: [ [seed1, reply1], [seed2, reply2], ... ]
    """
    #sort tweets based on the unique conv_id, then by S/R key to have seed_reply pairs sorted. An extra step for better
    #assurance in spite of the fact that the csv file may be in the correct order and does not require sorting.
    the_list = sorted(the_list[1:], key=itemgetter(conv_id_indx, s_r_indx))
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

def add_to_dict(t, the_length, vector):
    global max_index, features_dict, features_count_dict
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

    max = 4

    #i --> the length of the token
    #j --> starting index of the token
    if len(tokens)>0:#sometimes happens that tokens is empty --> eg: when tweet is just a url --> we exclude the url and it results an empty list.
        very_last_tok=''
        for i in range(1, max):
            is_the_first_tok = True#adding the very first ngram to dict
            for j in xrange(0, len(tokens) - (i - 1)):
                if check_features(tokens[j:j + i]):

                    t = " ".join(tokens[j:j + i])
                    if separate_features_for_seed_reply:
                        if is_seed:
                            t = 's ' + t
                        else:
                            t = 'r ' + t
                    #print t
                    add_to_dict(t, len(tokens), vector)

                    #these extra lines create extra tokens for the beginning and end of tweets.
                    if is_the_first_tok:#beginning of tweet -- note that j is not necessarily 0 as at j==0 the check_feature can turn out to be False!
                        very_first_tok = 'beg ' + t
                        add_to_dict(very_first_tok, len(tokens), vector)
                        is_the_first_tok = False
                    #every time check_features is correct the vary_last_tok gets updated, but will be added to the dict only when the j loop is over!
                    #this is because when j is in its final iteration, the check_features can be False and we can miss adding the very final tok!
                    very_last_tok = 'end ' + t
            add_to_dict(very_last_tok, len(tokens), vector)

    return vector

def get_sparse_feature_vector(seed_reply_list, text_indx):
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
            #there is no intersection between ngrams_seed and ngrams_reply. So, updating one results the entire dict
            ngrams_seed_reply = ngrams_seed.copy()
            ngrams_seed_reply.update(ngrams_reply)
        else:#aggregate the two dicts:
            for k, v in ngrams_seed.iteritems():#go through first dict key values.
                try:#if a key also exists in second dict, add the respective values into the new dict.
                    v2 = ngrams_reply[k]
                except KeyError:#if the key does not exist in second dict, keep the key-value of the 1st dict
                    v2=0
                ngrams_seed_reply[k] = v+v2
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

def randomly_extract_training_and_test_data(y, x, all_seed_reply_texts):
    y_training=[]
    x_training=[]
    y_test=[]
    x_test=[]
    training_texts=[]
    test_texts=[]
    if len(y)==len(x):
        length = len(y)
        for i in range(0, length):
            n = random.randint(1, 10)
            if n > 3 :#if n>3 --> 70% data for training and 30% for validation
                y_training.append(y[i])
                x_training.append(x[i])
                training_texts.append(all_seed_reply_texts[i])
            else:
                y_test.append(y[i])
                x_test.append(x[i])
                test_texts.append(all_seed_reply_texts[i])

    if equalise_pos_neg_sets:
        #duplicating pos/neg feature vectors may cause duplicate elements in the test set; remove!
        z = zip(x_test, y_test, test_texts)#zip x, y, text to remove associated elements-->[(x1,y1,t1), (x2,y2,t2), ...]
        z_noDuplicate = z[:]
        for xyt in z:
            duplicates = [xyt_1 for xyt_1 in z if xyt==xyt_1]
            dup = duplicates.pop(0)
            if len(duplicates)>0:
                z_noDuplicate = [xyt for xyt in z_noDuplicate if xyt not in duplicates]
                z_noDuplicate.append(dup)
        x_test, y_test, test_texts = zip(*z_noDuplicate)
        x_test, y_test, test_texts = list(x_test), list(y_test), list(test_texts)

        #duplicating pos/neg feature vectors may result a feature vector be present in both training and test sets when
        #randomly selected. So remove the intersection.
        for vect in x_test:
            if vect in x_training:
                indx = x_test.index(vect)
                x_test.remove(vect)
                y_test.pop(indx)
                test_texts.pop(indx)

    return y_training, x_training, y_test, x_test, training_texts, test_texts


def create_labels_and_features(seed_reply_agreed, seed_reply_disagreed, seed_reply_others, text_indx):
    global features_count_dict

    agreed_feature_vectors,    agreed_seed_reply_texts    = get_sparse_feature_vector(seed_reply_agreed,    text_indx)
    disagreed_feature_vectors, disagreed_seed_reply_texts = get_sparse_feature_vector(seed_reply_disagreed, text_indx)
    other_feature_vectors,     other_seed_reply_texts     = get_sparse_feature_vector(seed_reply_others, text_indx)

    #since we want to classify positive tweets from negative and other tweets, 50% of the training set must contain
    #positive tweets and 50% contain negative and other tweets (25% negative and 25% other tweets).
    if equalise_pos_neg_sets:
        size_diff = len(agreed_feature_vectors) - len(disagreed_feature_vectors)
        if size_diff > 0:
            length = len(disagreed_feature_vectors)
            for i in xrange(0, size_diff):
                r = random.randint(0, length - 1)
                duplicated_vect = disagreed_feature_vectors[r]
                disagreed_feature_vectors.append(duplicated_vect)
                disagreed_seed_reply_texts.append(disagreed_seed_reply_texts[r])
                #update the feature_count_dict for duplicated tokens
                for indx in duplicated_vect.keys():
                    features_count_dict[indx] += 1
        elif size_diff < 0:
            length = len(agreed_feature_vectors)
            for i in xrange(0, size_diff):
                r = random.randint(0, length - 1)
                duplicated_vect = agreed_feature_vectors[r]
                agreed_feature_vectors.append(duplicated_vect)
                agreed_seed_reply_texts.append(agreed_seed_reply_texts[r])
                #update the feature_count_dict for duplicated tokens
                for indx in duplicated_vect.keys():
                    features_count_dict[indx] += 1

    write_features_and_freqs_to_csv()

    if strip_singles_flag:
        agreed_feature_vectors = strip_singles(agreed_feature_vectors)
        disagreed_feature_vectors = strip_singles(disagreed_feature_vectors)
        other_feature_vectors=strip_singles(other_feature_vectors)


    l1 = classifier_labels[0]
    l2 = classifier_labels[1]
    l3 = classifier_labels[2]


    if (l1==+1 and l2==l3==-1):# or (l1==-1 and l2==l3==+1):

        if equalise_pos_neg_sets:
            disagreed_class_size = other_class_size = len(agreed_feature_vectors)/2
            sampled = random.sample(zip(disagreed_feature_vectors, disagreed_seed_reply_texts), disagreed_class_size)
            disagreed_feature_vectors = list((zip(*sampled))[0])
            disagreed_seed_reply_texts= list((zip(*sampled))[1])
        else:
            #keep negative set as it is, and sample the rest from other pairs
            other_class_size = len(agreed_feature_vectors) - len(disagreed_feature_vectors)

        sampled = random.sample(zip(other_feature_vectors, other_seed_reply_texts), other_class_size)
        other_feature_vectors = list((zip(*sampled))[0])
        other_seed_reply_texts = list((zip(*sampled))[1])

    elif (l2==-1 and l1==l3==+1):# or (l2==+1 and l1==l2==-1):

        agreed_class_size = other_class_size = len(disagreed_feature_vectors)/2
        sampled = random.sample(zip(agreed_feature_vectors, agreed_seed_reply_texts), agreed_class_size)
        agreed_feature_vectors = list((zip(*sampled))[0])
        agreed_seed_reply_texts= list((zip(*sampled))[1])

        sampled = random.sample(zip(other_feature_vectors, other_seed_reply_texts), other_class_size)
        other_feature_vectors = list((zip(*sampled))[0])
        other_seed_reply_texts = list((zip(*sampled))[1])

    else:
        raise Exception('classifier label list is wrong!')


    y = [l1] * len(agreed_feature_vectors)
    y = y + [l2] * len(disagreed_feature_vectors)
    y = y + [l3] * len(other_feature_vectors)

    x = agreed_feature_vectors + disagreed_feature_vectors + other_feature_vectors
    all_seed_reply_texts = agreed_seed_reply_texts + disagreed_seed_reply_texts + other_seed_reply_texts

    write_labels_and_features_to_csv(y, x)
    my_util.write_csv_file(source_dir + seed_reply_file_name, False, True, all_seed_reply_texts)

    return x, y, all_seed_reply_texts

def calc_stats_of_training_classification(y, x, m):

    p_label, p_acc, p_val = liblinearutil.predict(y, x, m)
    pos=[]
    neg=[]
    for i in range(0, len(y)):
        if y[i]==1:
            pos = pos+p_val[i]
        else:
            neg = neg+p_val[i]

    mean_pos, std_pos = math_extra.calc_mean_stdev(pos)
    mean_neg, std_neg = math_extra.calc_mean_stdev(neg)

    return mean_pos, std_pos, mean_neg, std_neg


def go_train(x, y, all_seed_reply_texts, trial_nr):

    y_training, x_training, y_test, x_test, training_texts, test_texts = \
                                                            randomly_extract_training_and_test_data(y, x, all_seed_reply_texts)

    prob = liblinearutil.problem(y_training, x_training)
    param = liblinearutil.parameter('-s 1 -c 1 -B 1')
    m = liblinearutil.train(prob, param)

    mean_pos, std_pos, mean_neg, std_neg = calc_stats_of_training_classification(y_training, x_training, m)

    test_result = []

    if x_test <>[]:
        #p_labels --> classification labels predicted by the system.
        #p_acc --> tuple including accuracy (for classification), MSE, and variance (for regression).
        #p_val --> classification values predicted by the system.
        p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test, m)

        header = ['seed', 'reply', 'original_label', 'predicted_label', 'predicted_value', 'prediction_success']
        test_result.append(header)
        nCorrect_classifications=0
        nCorrect_classification_pos=0
        nCorrect_classification_neg=0
        nIncorrec_classification_pos=0
        nIncorrec_classification_neg=0
        for i in range(0, len(test_texts)):
            seed_reply_text = test_texts[i]
            seed = seed_reply_text[0]
            reply= seed_reply_text[1]
            original_label=y_test[i]
            predicted_label=int(p_label[i])
            predicted_value=p_val[i]
            #if original_label*predicted_label>0:
            prediction = 'wrong'
            threshold_pos = 0 #mean_pos - 2 * std_pos
            threshold_neg = 0 #mean_neg + 2 * std_neg
            if original_label==1:
                if predicted_value[0] > threshold_pos:
                    prediction = 'correct'
                    nCorrect_classification_pos +=1
                    nCorrect_classifications +=1
                else:
                    nIncorrec_classification_pos +=1
            elif original_label==-1:
                if predicted_value[0] < threshold_neg:
                    prediction = 'correct'
                    nCorrect_classification_neg +=1
                    nCorrect_classifications +=1
                else:
                    nIncorrec_classification_neg +=1
            test_result.append([seed, reply, original_label, predicted_label, predicted_value, prediction])
        #print 'threshold_pos', threshold_pos, "threshold_neg:", threshold_neg

        classification_accuracy = round(float(nCorrect_classifications) / len(test_texts), 2)
        precision_pos = round(float(nCorrect_classification_pos) / (nCorrect_classification_pos+nIncorrec_classification_neg), 2)
        precision_neg = round(float(nCorrect_classification_neg) / (nCorrect_classification_neg+nIncorrec_classification_pos), 2)
        recall_pos = round(float(nCorrect_classification_pos) / (nCorrect_classification_pos+nIncorrec_classification_pos), 2)
        recall_neg = round(float(nCorrect_classification_neg) / (nCorrect_classification_neg+nIncorrec_classification_neg), 2)

        if nCorrect_classifications <> (nCorrect_classification_pos+nCorrect_classification_neg):
            print "nr of total correct classification is not equal to the number of +1 and -1 classification!!!!!"

        print 'trial no:', trial_nr, '\nTest set calssification accuracy:', classification_accuracy

    my_util.write_csv_file(source_dir+test_result_name+str(trial_nr)+'_'+str(classification_accuracy)+'%', False, True, test_result)

    return classification_accuracy, precision_pos, precision_neg, recall_pos, recall_neg


    #-v n --> n-fold cross validation
#    param_cv = liblinearutil.parameter('-c 1 -B 0 -v 2')
#    m_cv = liblinearutil.train(prob, param_cv)


#original agreed tweets read line by line from a csv file. The first row is the header.
agreed_pairs = my_util.read_csv_file(source_dir + agreed_file_name, False, True)
positive_answers_pairs = my_util.read_csv_file(source_dir + positive_answers_file_name, False, True)
supportive_pairs = my_util.read_csv_file(source_dir + supportive_pairs_file_name, False, True)
#original disagreed tweets read line by line from a csv file. The first row is the header.
disagreed_pairs = my_util.read_csv_file(source_dir + disagreed_file_name, False, True)
negative_answers_pairs = my_util.read_csv_file(source_dir + negative_answers_file_name, False, True)
offensive_pairs = my_util.read_csv_file(source_dir + offensive_sarcastic_tweets, False, True)
other_pairs = my_util.read_csv_file(source_dir + followups_file_name, False, True)

header = agreed_pairs[0]
conv_id_indx = header.index('conv_id')#onv_id counts seed/reply pairs and is unique for both seed & reply.
tweet_id_indx = header.index('tweet_id')#tweet_id is a unique no. associated to each single tweet.
s_r_indx = header.index('S/R')#S/R stickers implying whether the tweet is seed or reply.
text_indx = header.index('text')#the tweet texts.

##############################################################################################################
##############################################################################################################
agreed_pairs    = agreed_pairs    + positive_answers_pairs[1:] + supportive_pairs[1:]
disagreed_pairs = disagreed_pairs + negative_answers_pairs[1:] + offensive_pairs[1:]
##############################################################################################################
##############################################################################################################

agreed_pairs    = unicode_to_float(agreed_pairs,    conv_id_indx,  1)
agreed_pairs    = unicode_to_float(agreed_pairs,    tweet_id_indx, 1)
disagreed_pairs = unicode_to_float(disagreed_pairs, conv_id_indx,  1)
disagreed_pairs = unicode_to_float(disagreed_pairs, tweet_id_indx, 1)
other_pairs     = unicode_to_float(other_pairs,      conv_id_indx, 1)
other_pairs     = unicode_to_float(other_pairs,     tweet_id_indx, 1)

agreed_pairs    = code_0_1_for_seed_reply(agreed_pairs,    s_r_indx, 1)
disagreed_pairs = code_0_1_for_seed_reply(disagreed_pairs, s_r_indx, 1)
other_pairs     = code_0_1_for_seed_reply(other_pairs,     s_r_indx, 1)

seed_reply_agreed    = make_seed_reply_list(agreed_pairs,    conv_id_indx, s_r_indx)
seed_reply_disagreed = make_seed_reply_list(disagreed_pairs, conv_id_indx, s_r_indx)
seed_reply_other_pairs=make_seed_reply_list(other_pairs,     conv_id_indx, s_r_indx)

if read_training_data_from_file:
    y, x = liblinearutil.svm_read_problem(source_dir + training_file_name + '.csv')
    all_seed_reply_texts = my_util.read_csv_file(source_dir + seed_reply_file_name, False, True)
else:
    x, y, all_seed_reply_texts = create_labels_and_features(seed_reply_agreed, seed_reply_disagreed, seed_reply_other_pairs, text_indx)

########################################################################################################################
########################################################################################################################
print "--------------------------------------------------------------------------------------------------------------\n"
print "\ncross validation in progress..."
prob_CV = liblinearutil.problem(y, x)
param = liblinearutil.parameter('-c 1 -B 1 -v 10')
m = liblinearutil.train(prob_CV, param)
print "\ncross validation finished."
print "--------------------------------------------------------------------------------------------------------------\n"
########################################################################################################################
########################################################################################################################

results = []
header = ['iteration', 'accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg']
for i in range(0, 20):
    classification_accuracy, precision_pos, precision_neg, recall_pos, recall_neg = go_train(x, y, all_seed_reply_texts, i+1)
    result = [i+1, classification_accuracy, precision_pos, precision_neg, recall_pos, recall_neg]
    results.append(result)

results = sorted(results, key=itemgetter(1))
results.reverse()
mean_ac, stdev_ac = math_extra.calc_mean_stdev([row[1] for row in results])
mean_pp, stdev_pp = math_extra.calc_mean_stdev([row[2] for row in results])
mean_pn, stdev_pn = math_extra.calc_mean_stdev([row[3] for row in results])
mean_rp, stdev_rp = math_extra.calc_mean_stdev([row[4] for row in results])
mean_rn, stdev_rn = math_extra.calc_mean_stdev([row[5] for row in results])

print '\nmax:', results[0][1], 'iteration:', results[0][0]
print 'min:', results[len(results) - 1][1], 'iteration:', results[len(results) - 1][0]
print 'mean:', mean_ac
print 'stdev:', stdev_ac

means = ['mean', mean_ac, mean_pp, mean_pn, mean_rp, mean_rn]
stdevs = ['stdev', stdev_ac, stdev_pp, stdev_pn, stdev_rp, stdev_rn]
results.append(means)
results.append(stdevs)

results = [header]+results
my_util.write_csv_file(source_dir + tabel_result_name, False, True, results)

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
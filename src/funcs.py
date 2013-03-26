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

def ngrams(tweet, is_seed, separate_features_for_seed_reply, features_dict, features_count_dict, max_index):
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
    #sometimes tokens may be empty --> eg: when tweet is just a url --> we exclude the url and it results an empty list.
    if len(tokens) > 0:
        very_last_tok = ''
        for i in range(1, max):
            is_the_first_tok = True#adding the very first ngram to dict
            for j in xrange(0, len(tokens) - (i - 1)):
                if check_features(tokens[j:j + i], False):

                    t = " ".join(tokens[j:j + i])
                    if separate_features_for_seed_reply:

                        if is_seed:
                            t = 's ' + t
                        else:
                            t = 'r ' + t
                            #print t

                        #note: vector, features_dict, features_count_dict are passed by reference
                        #only max_index must be returned, as its value changes inside the method.
                        max_index = add_to_dict(t, len(tokens), vector, features_dict, features_count_dict, max_index)

                    #these extra lines create extra tokens for the beginning and end of tweets.
                    if is_the_first_tok:
                        #beginning of tweet -- note that j is not necessarily 0,
                        # as at j==0 the check_feature can turn out to be False!
                        very_first_tok = 'beg ' + t
                        max_index = add_to_dict(very_first_tok, len(tokens), vector, features_dict, features_count_dict, max_index)
                        is_the_first_tok = False
                    #every time check_features is correct the vary_last_tok gets updated, but will be added to the dict
                    #only when the j loop is over! this is because when j is in its final iteration, the check_features
                    #can be False and we can miss adding the very final tok!
                    very_last_tok = 'end ' + t
            max_index = add_to_dict(very_last_tok, len(tokens), vector, features_dict, features_count_dict, max_index)

    return vector, max_index

def get_sparse_feature_vector(s_r_list, text_indx, sep_s_r_features, features_dict, features_count_dict, max_index):
    """
    for each pair of tweets in the form of seed_reply, create a dict presenting the aggregate freq of tokens appearing
    in the seed and in the reply
    """
    feature_vectors = []
    seed_reply_texts = []
    for s_r in s_r_list:

        ngrams_seed_reply = {}
        seed = s_r[0]
        reply = s_r[1]
        seed_text = seed[text_indx]
        reply_text = reply[text_indx]

        ngrams_seed, max_index= ngrams(seed_text, True, sep_s_r_features, features_dict, features_count_dict, max_index)

        ngrams_reply,max_index= ngrams(reply_text,False, sep_s_r_features, features_dict,features_count_dict, max_index)

        if sep_s_r_features:
            #since we have put two separate letters at the beginning of ngrmas for seed and reply tokens,
            #there is no intersection between ngrams_seed and ngrams_reply. So, updating one results the entire dict
            ngrams_seed_reply = ngrams_seed.copy()
            ngrams_seed_reply.update(ngrams_reply)
        else:#aggregate the two dicts:
            for k, v in ngrams_seed.iteritems():#go through first dict key values.
                try:#if a key also exists in second dict, add the respective values into the new dict.
                    v2 = ngrams_reply[k]
                except KeyError:#if the key does not exist in second dict, keep the key-value of the 1st dict
                    v2 = 0
                ngrams_seed_reply[k] = v + v2
            for k, v in ngrams_reply.iteritems():#go through the elements of the second dict:
                try:
                    ngrams_seed[k]
                except KeyError:#if the second-dict key does not exist in the 1st dict, add it to the new dict.
                    ngrams_seed_reply[k] = v

        feature_vectors.append(ngrams_seed_reply)
        seed_reply_texts.append([seed_text, reply_text])

    return feature_vectors, seed_reply_texts, max_index


def create_labels_and_features(features_dict, features_count_dict, max_index,
                               seed_reply_agreed, seed_reply_disagreed, seed_reply_others,
                               text_indx,
                               separate_seed_reply_features):

    feature_vects_agr, s_r_texts_agr, max_index = get_sparse_feature_vector\
         (seed_reply_agreed, text_indx, separate_seed_reply_features, features_dict, features_count_dict, max_index)

    feature_vects_dis, s_r_texts_dis, max_index = get_sparse_feature_vector\
     (seed_reply_disagreed, text_indx, separate_seed_reply_features, features_dict, features_count_dict, max_index)

    feature_vects_others, s_r_texts_others, max_index = get_sparse_feature_vector\
        (seed_reply_others, text_indx, separate_seed_reply_features, features_dict, features_count_dict, max_index)

    return feature_vects_agr, s_r_texts_agr, feature_vects_dis, s_r_texts_dis, feature_vects_others, s_r_texts_others, max_index

def get_ngrams_worry(tweet, features_dict, features_count_dict, max_index, m, n):
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
    nr_of_features = 0
    stopword_flag = False
    if len(tokens) > 0:
        for i in range(m, n + 1):
            # if i == 1:
            #     stopword_flag = True
            for j in xrange(0, len(tokens) - (i - 1)):
                if check_features(tokens[j:j + i], stopword_flag):
                    nr_of_features += 1
                    t = " ".join(tokens[j:j + i])
                    #note: vector, features_dict, features_count_dict are passed by reference
                    #only max_index must be returned, as its value changes inside the method.
                    #max_index = add_to_dict(t, len(tokens), vector, features_dict, features_count_dict, max_index)

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

        #divide all elements of the vector by the number of traversed features.
        #this is more correct than the dividing by len(tokens).
        #this is because the number of ngrams (addressed in a vector) is not necessarily equal to the number of features.
        vector = {a : float(c)/nr_of_features for a, c in vector.iteritems()}

    return vector, max_index


def get_sparse_feature_vector_worry(tweet_list, features_dict, features_count_dict, max_index, m, n):

    feature_vectors = []
    tweet_texts = []
    for tweet in tweet_list:
        vector, max_index = get_ngrams_worry(tweet, features_dict, features_count_dict, max_index, m, n)
        feature_vectors.append(vector)
        #in general the tweet_texts should be tweet_list itself. We return this in case the order of the vector changes
        # during the loop.
        tweet_texts.append(tweet)

    return feature_vectors, tweet_texts, max_index


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
    #method creating test-set dictionary!

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

def code_0_1_for_seed_reply(the_list, col_nr, strt_indx):
    """
    function changing the seed reply stickers (S/R) to 0/1 so that data can be sorted on that column too..
    """
    for r in the_list[strt_indx:]:
        val = r[col_nr]
        if val == 'S':
            new_val = 0
        else:
            new_val = 1
        r[col_nr] = new_val
    return the_list

def make_seeds_replies(the_list, conv_id_indx, s_r_indx):
    """
    this function creates a list of tuples like: [ (seed1, reply1), (seed2, reply2), ... ]
    """
    #sort tweets based on the unique conv_id, then by S/R key to have seed_reply pairs sorted.
    the_list = sorted(the_list[1:], key=itemgetter(conv_id_indx, s_r_indx))
    seeds = the_list[0::2]#start from index 0 and get every other line
    replies = the_list[1::2]#start from index 1 and get every other line
    seeds_replies = zip(seeds, replies)
    is_data_missed = False
    message = 'no data is missed'
    c=0
    for s_r in seeds_replies:
        c +=1
        seed = s_r[0]
        reply= s_r[1]
        conv_id_seed = seed[conv_id_indx]
        conv_id_reply= reply[conv_id_indx]
        seed_code = seed[s_r_indx]
        reply_code = reply[s_r_indx]
        if (conv_id_seed <> conv_id_reply):
            is_data_missed = True
            message = 'seed and reply conv_id do not match at index ' + str(c)
            break
        else:
            ids_s = [seed[conv_id_indx] for seed in seeds if seed[conv_id_indx] == conv_id_seed]
            ids_r = [reply[conv_id_indx] for reply in replies if reply[conv_id_indx] == conv_id_reply]
            if len(ids_r)+len(ids_s) >2:
                is_data_missed = True
                message = 'there is duplicated conv_id!'
        if (seed_code <> 0):
            is_data_missed = True
            message = 'seed code is not 0 at index ' + str(c)
            break
        if (reply_code <> 1):
            is_data_missed = True
            message = 'reply code is not 1 at index ' + str(c)
            break

    return is_data_missed, message, seeds_replies

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

def calc_prediction_stats(thresh_pos, thresh_neg, y_test, s_r_texts_test, p_label, p_val):
    prediction_result=[]
    if p_val<>[]:
        header = ['seed', 'reply', 'original_label', 'predicted_label', 'predicted_value', 'prediction_success']
    else:
        header = ['seed', 'reply', 'original_label', 'predicted_label', 'prediction_success']
    prediction_result.append(header)
    true_pos = true_neg = true_zero = 0
    false_pos = false_neg = false_zero = 0
    nPos = nNeg = nZer = 0
    for i in range(0, len(s_r_texts_test)):
        s_r_text = s_r_texts_test[i]
        seed = s_r_text[0]
        reply = s_r_text[1]
        original_label = y_test[i]
        predicted_label = int(p_label[i])
        if p_val<>[]:
            predicted_value = p_val[i]
            if predicted_value[0] > thresh_pos:
                predicted_label = +1
            elif predicted_value[0] < thresh_neg:
                predicted_label = -1
            else:
                predicted_label = 0
                #if original_label*predicted_label>0:
        prediction = 'wrong'

        if original_label == +1:
            nPos +=1
            if predicted_label == +1:
                prediction = 'correct'
                true_pos += 1
            elif predicted_label == -1:
                false_neg += 1
            elif predicted_label == 0:
                false_zero += 1
        elif original_label == -1:
            nNeg +=1
            if predicted_label == -1:
                prediction = 'correct'
                true_neg += 1
            elif predicted_label == +1:
                false_pos += 1
            elif predicted_label == 0:
                false_zero += 1
        elif original_label == 0:
            nZer +=1
            if predicted_label == 0:
                prediction = 'correct'
                true_zero += 1
            elif predicted_label == +1:
                false_pos += 1
            elif predicted_label == -1:
                false_neg += 1
        if p_val<>[]:
            prediction_result.append([seed, reply, original_label, predicted_label, predicted_value, prediction])
        else:
            #in this case there is no 'predicted_value'
            prediction_result.append([seed, reply, original_label, predicted_label, prediction])

    accuracy = round(float(true_pos + true_neg + true_zero) / len(y_test), 2)
    precision_pos = precision_neg = precision_zero = recall_pos = recall_neg = recall_zero = 0
    try:
        precision_pos = round(float(true_pos) / (true_pos + false_pos), 2)
    except ZeroDivisionError:#let the value be zero in this case!
        pass
    try:
        precision_neg = round(float(true_neg) / (true_neg + false_neg), 2)
    except ZeroDivisionError:#let the value be zero in this case!
        pass
    try:
        precision_zero = round(float(true_zero) / (true_zero + false_zero), 2)
    except ZeroDivisionError:#let the value be zero in this case!
        pass
    try:
        recall_pos = round(float(true_pos) / nPos, 2)
    except ZeroDivisionError:#let the value be zero in this case!
        pass
    try:
        recall_neg = round(float(true_neg) / nNeg, 2)
    except ZeroDivisionError:#let the value be zero in this case!
        pass
    try:
        recall_zero = round(float(true_zero) / nZer, 2)
    except ZeroDivisionError:#the value would be zero in this case!
        pass

    return prediction_result, accuracy, precision_pos, precision_neg, precision_zero, recall_pos, recall_neg, recall_zero

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

import topic_funcs
import operator
import getter
import my_util
import tokens_similarity
import math_extra
import os
from operator import itemgetter
import difflib
import math

"""
This version is similar to version 10. We just change the parameter to fine tune it.

"""

lang = 'en'
tweets = []
results = []

tab_delimiter_flag = False
unicode_flag = True

# The conversion of ~ to $HOME is called tilde expansion and is a common user interface feature.
# The file system does not know anything about it. In Python, this feature is implemented by os.path.expanduser.
home_dir = os.path.expanduser('~')
saving_dir = home_dir + '/chatterbox-core/twitter_param_results/Results11_qr'

#Reaa the tweets file if the have already pickled. Otherwise, get the tweets from the data base.
try:
    tweets = my_util.read_pickled_file('the_tweets_file')
except IOError:
    import getter

    tweets = getter.get_tweets()
    my_util.pickle_to_file('the_tweets_file', tweets)

#tweets are given in a list [[t1,s1], [t2,s2], [t3,s3],...] where t is the tweet and s is the sentimental value.
#sentimental values are between 0 to 1 for positive tweets, and -1 to 0 for negative tweets.
sentiment_thresholds_pos_neg = [(0.4, -0.01)]
stop_word_flags = [True]#, False]
pickup_nums_1_and_2 = [(250, 100)]#, (500, 50), (250, 100), (250, 50)]
prob_plus_vals = [2, 3, 5, 10]
stdev_mults = [0.0625, 0.125, 0.25, 0.5]#, 0.75, 1, 1.25, 1.5, 1.75, 2]
str_comp_types = ['quick_ratio']#, 'ratio', 'Levenshtein', 'stringComp']
str_comp_vals = [0.75]
duplicate_removal_flags = [False, True]
str_length_flags = ['shortest','longest']
exclude_toks_with_freq_one_flag = False
no_of_presenting_tweets = 50
compare_results = False

def remove_duplicate_tweets(tweets_list):
    c=0
    return_list = []
    while len(tweets_list) > 0:
        t = tweets_list[0]
        #using the quick_ratio() is very time consuming!
        #duplicates = [s for s in tweets_list if difflib.SequenceMatcher(None, s, t).quick_ratio() > 0.90]
        duplicates = [s for s in tweets_list if s==t]
        return_list.append([t, len(duplicates)])
        tweets_list = [x for x in tweets_list if x not in duplicates]
        if len(duplicates)>1:
            c+=1
            print c, len(duplicates), 'duplicate: ', t
    return return_list

header = [
    'stp_wrd_flag',
    'sent_thresh_pos', 'sent_thresh_neg', 'no_of_tweets_pos', 'no_of_tweets_neg', 'duplicates_removed_pos', 'duplicates_removed_neg',
    'prob_plus_val', 'pickup_no_1', 'pickup_no_2', 'stdev_mult', 'prob_close_val_pos', 'prob_close_val_neg', 'str_length_flag',
    'avg_prob_pos', 'stdev_prob_pos', #'avg_prob_diff_pos' 'stdev_prob_diff_pos',
    'avg_prob_neg', 'stdev_prob_neg', #'avg_prob_diff_neg', 'stdev_prob_diff_neg',
    'str_comp_type', 'str_comp_val',
    'final_toks_pos', 'final_toks_neg']

results.append(header)

for sent_thresh in sentiment_thresholds_pos_neg:
    sent_thresh_pos = sent_thresh[0]
    sent_thresh_neg = sent_thresh[1]
    tweets_pos = []
    tweets_neg = []
    tweets_gray =[] #these are the tweets t that: sent_thresh_neg < t < sent_thresh_pos
    # get all tweets from pair-list [twitt,centVal] which centVal>=b
    tweets_pos = [t_v[0] for t_v in tweets if t_v[1] >= sent_thresh_pos]
    no_of_tweets_pos = len(tweets_pos)
    # get all tweets from pair-list [twitt,centVal] which centVal<=b
    tweets_neg = [t_v[0] for t_v in tweets if t_v[1] <= sent_thresh_neg]
    no_of_tweets_neg = len(tweets_neg)

    tweets_gray = [t_v[0] for t_v in tweets if (sent_thresh_neg < t_v[1] and t_v[1] < sent_thresh_pos)]
    no_of_tweets_gray = len(tweets_gray)

    for duplicate_removal_flag in duplicate_removal_flags:

        if duplicate_removal_flag:
            duplicate_count_pos = remove_duplicate_tweets(tweets_pos)
            duplicate_count_neg = remove_duplicate_tweets(tweets_neg)
            duplicate_count_gray= remove_duplicate_tweets(tweets_gray)
            unique_tweets_pos = [d[0] for d in duplicate_count_pos] #we take one element from each cluster!
            unique_tweets_neg = [d[0] for d in duplicate_count_neg] #we take one element from each cluster!
            unique_tweets_gray= [d[0] for d in duplicate_count_gray]#we take one element from each cluster!
            duplicates_removed_pos = len(tweets_pos) - len(unique_tweets_pos)
            duplicates_removed_neg = len(tweets_neg) - len(unique_tweets_neg)
            duplicates_removed_gray= len(tweets_gray)- len(unique_tweets_gray)
            tweets_pos = unique_tweets_pos
            tweets_neg = unique_tweets_neg
            tweets_gray= unique_tweets_gray
        else:
            duplicates_removed_pos = 0
            duplicates_removed_neg = 0
            duplicates_removed_gray= 0

        ngrams_pos = []
        ngrams_neg = []
        ngrams_gray= []

        for stp_wrd_flag in stop_word_flags:
            for t in tweets_pos:
                ngrams_pos = ngrams_pos + topic_funcs.ngrams(t.split(), lang, stp_wrd_flag)
            for t in tweets_neg:
                ngrams_neg = ngrams_neg + topic_funcs.ngrams(t.split(), lang, stp_wrd_flag)
            for t in tweets_gray:
                ngrams_gray = ngrams_gray + topic_funcs.ngrams(t.split(), lang, stp_wrd_flag)

            freq_pos = topic_funcs.term_frequency(ngrams_pos, lang)
            freq_neg = topic_funcs.term_frequency(ngrams_neg, lang)
            freq_gray= topic_funcs.term_frequency(ngrams_gray,lang)

# When the freq of a token is 1, this mainly causes a prob of 1 given that the respective pos(neg) tok's freq is zero.
# Hence, we may want to exclude these tokens.
            if exclude_toks_with_freq_one_flag:
                non_freq_one_toks_pos = {}
                non_freq_one_toks_neg = {}
                for t_pos, f_pos in freq_pos.items():
                    if f_pos ==1:
                        try:
                            f_neg = freq_neg[t_pos]
                            non_freq_one_toks_pos[t_pos] = f_pos
                        except KeyError:
                            pass

                for t_neg, f_neg in freq_neg.items():
                    if f_neg == 1:
                        try:
                            f_pos = freq_neg[t_neg]
                            non_freq_one_toks_neg[t_neg] = f_neg
                        except KeyError:
                            pass
                freq_pos = non_freq_one_toks_pos
                freq_neg = non_freq_one_toks_neg

            for prob_plus_val in prob_plus_vals:

                # This line is different from the previous codes in that we calculate p/(p+g+n) rather than p/(p+n)
                # where p is the number of counts that token t appears in the positive set,
                # n is the number of counts that token t appears in the negative set,
                # is the number of counts that token t appears in the gray set (btwn positive and negative thresholds)
                weight_freq_pos = topic_funcs.calc_probs(freq_pos, freq_neg, freq_gray, prob_plus_val)
                weight_freq_pos = sorted(weight_freq_pos.iteritems(), key=operator.itemgetter(1))#sort on weights
                weight_freq_pos.reverse() #large weights first

                weight_freq_neg = topic_funcs.calc_probs(freq_neg, freq_pos, freq_gray, prob_plus_val)
                weight_freq_neg = sorted(weight_freq_neg.iteritems(), key=operator.itemgetter(1))#sort on weights
                weight_freq_neg.reverse() #large weights first

                avg_prob_pos, stdev_prob_pos = math_extra.calc_mean_stdev([w[1] for w in weight_freq_pos])
                avg_prob_neg, stdev_prob_neg = math_extra.calc_mean_stdev([w[1] for w in weight_freq_neg])

#                prob_diffs_pos = [abs(w1[1] - w2[1]) for w1 in weight_freq_pos for w2 in weight_freq_pos if w1 <> w2]
#                prob_diffs_neg = [abs(w1[1] - w2[1]) for w1 in weight_freq_neg for w2 in weight_freq_neg if w1 <> w2]
#                avg_prob_diff_pos, stdev_prob_diff_pos = math_extra.calc_mean_stdev(prob_diffs_pos)
#                avg_prob_diff_neg, stdev_prob_diff_neg = math_extra.calc_mean_stdev(prob_diffs_neg)

                for pickup_no_1_2 in pickup_nums_1_and_2:
                    #we are concerned in high probable toks, so only keep those with values above average!
#                        weighted_toks_pos = [w for w in weight_freq_pos if w[1] >= average_weight_pos]
#                        weighted_toks_neg = [w for w in weight_freq_neg if w[1] >= average_weight_neg]
                    pickup_no_1 = pickup_no_1_2[0]
                    pickup_no_2 = pickup_no_1_2[1]

                    weighted_toks_pos = weight_freq_pos[:pickup_no_1]
                    weighted_toks_neg = weight_freq_neg[:pickup_no_1]

                    for stdev_mult in stdev_mults:
                        prob_close_val_pos = stdev_mult * stdev_prob_pos
                        prob_close_val_neg = stdev_mult * stdev_prob_neg

                        encompassing_toks_pos = topic_funcs.find_encompassing_toks(weighted_toks_pos, prob_close_val_pos)
                        encompassing_toks_neg = topic_funcs.find_encompassing_toks(weighted_toks_neg, prob_close_val_neg)

                        for str_length_flag in str_length_flags:

                            for str_comp_type in str_comp_types:
                                for str_comp_val in str_comp_vals:

                                    final_toks_pos = topic_funcs.find_similar_toks(encompassing_toks_pos,
                                        pickup_no_2, str_comp_type, str_comp_val, prob_close_val_pos, str_length_flag)

                                    final_toks_pos = sorted(final_toks_pos.iteritems(), key=operator.itemgetter(1))
                                    final_toks_pos.reverse()
                                    final_toks_pos = final_toks_pos[:no_of_presenting_tweets]

                                    final_toks_neg = topic_funcs.find_similar_toks(encompassing_toks_neg,
                                        pickup_no_2, str_comp_type, str_comp_type, prob_close_val_neg, str_length_flag)

                                    final_toks_neg = sorted(final_toks_neg.iteritems(),key=operator.itemgetter(1))
                                    final_toks_neg.reverse()
                                    final_toks_neg = final_toks_neg[:no_of_presenting_tweets]

                                    result = [eval(h) for h in header]
                                    print result[:len(result)-2]

                                    results.append(result)


my_util.write_csv_file(saving_dir + '/results', tab_delimiter_flag, unicode_flag, results)
my_util.pickle_to_file(saving_dir + '/results_pickled', results)

if compare_results:
    #The following loop compares every unique pair of elements in results and gives a value indicating how similar the
    #tokesn are.
    comparisons = []
    comparison_header  =  header[:header.index('final_toks_pos')] + [''] \
                        + header[:header.index('final_toks_pos')] + [''] \
                        + ['total_dissimilarity'] + [''] \
                        + ['dissim_val_pos'] + ['toks1_pos'] + ['toks2_pos'] + ['diff12_pos'] + ['diff21_pos'] \
                        + ['dissim_val_neg'] + ['toks1_neg'] + ['toks2_neg'] + ['diff12_neg'] + ['diff21_neg']
    comparisons.append(comparison_header)

    for i in range(1, len(results)): #The first row is the header so we start from he second row (i.e. from index1)
        for j in range(i + 1, len(results)):
            r1 = results[i]
            r2 = results[j]
            params1 = r1[:header.index('final_toks_pos')]
            params2 = r2[:header.index('final_toks_pos')]
            t1_pos = r1[header.index('final_toks_pos')]
            t2_pos = r2[header.index('final_toks_pos')]
            t1_neg = r1[header.index('final_toks_pos')]
            t2_neg = r2[header.index('final_toks_pos')]
            comp_val_pos, diff12_pos, diff21_pos = tokens_similarity.compare_tokens(t1_pos, t2_pos)
            comp_val_neg, diff12_neg, diff21_neg = tokens_similarity.compare_tokens(t1_neg, t2_neg)
            comparison_result = params1 + [''] + params2 + [''] \
                                + [comp_val_pos+comp_val_neg] + [''] \
                                + [comp_val_pos] + [t1_pos] + [t2_pos] + [diff12_pos] + [diff21_pos] \
                                + [comp_val_neg] + [t1_neg] + [t2_neg] + [diff12_neg] + [diff21_neg]
            comparisons.append(comparison_result)

    indx = comparison_header.index('total_dissimilarity')
    comparisons = sorted(comparisons, key=itemgetter(indx))
    comparisons.reverse()
    my_util.write_csv_file(saving_dir + '/comparisons', tab_delimiter_flag, unicode_flag, comparisons)


# The following loop goes through the results and takes the positive (negative) tokens and list them in a separate list.
all_toks_pos = []
all_toks_neg = []
for i in range(1, len(results)): #The first row is the header so we start from he second row (i.e. from index1)
    r = results[i]
    t_w_pos = r[header.index('final_toks_pos')]
    for t_w in t_w_pos:
        t = t_w[0]
        # remove the white spaces before/after the string (some tokens appears to have white spaces before/after them).
        t = t.strip()
        if not(t in all_toks_pos):
            all_toks_pos.append(t)
            #        else:
            #            print 'Pos dropped: ', t

    t_w_neg = r[header.index('final_toks_neg')]
    for t_w in t_w_neg:
        t = t_w[0]
        t = t.strip()
        if not(t in all_toks_neg):
            all_toks_neg.append(t)

#we put each token t in a list ([t]) so that it can be written in rows by csv.writer()
all_toks_pos = [[t] for t in all_toks_pos]
all_toks_neg = [[t] for t in all_toks_neg]

my_util.write_csv_file(saving_dir + '/final_toks_pos', tab_delimiter_flag, unicode_flag, all_toks_pos)
my_util.write_csv_file(saving_dir + '/final_toks_neg', tab_delimiter_flag, unicode_flag, all_toks_neg)
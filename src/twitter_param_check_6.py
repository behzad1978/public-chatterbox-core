import topic_funcs
import operator
import getter
import my_util
import tokens_similarity
import math_extra
import difflib
import math

lang = 'en'
tweets = []
results = []
final_toks_qr_pos_prev = []
final_toks_qr_neg_prev = []
final_toks_r_pos_prev = []
final_toks_r_neg_prev = []
final_toks_l_pos_prev = []
final_toks_l_neg_prev = []
final_toks_sc_pos_prev = []
final_toks_sc_neg_prev = []

tab_delimiter_flag = False
unicode_flag = True
saving_dir = '~/chatterbox-core/twitter_param_results/Results7'

#Reaa the tweets file if the have already pickled. Otherwise, get the tweets from the data base.
try:
    tweets = my_util.read_pickled_file('the_tweets_file')
except IOError:
    import getter

    tweets = getter.get_tweets()
    my_util.pickle_to_file('the_tweets_file', tweets)

#tweets are given in a list [[t1,s1], [t2,s2], [t3,s3],...] where t is the tweet and s is the sentimental value.
#sentimental values are between 0 to 1 for positive tweets, and -1 to 0 for negative tweets.
sentiment_thresholds_pos = [0.15] #[i*0.1 for i in range(0,10)]
sentiment_thresholds_neg = [-0.15] #[-1*i for i in sentiment_borders_pos] # -1*sentiment_border_pos
stop_word_flags = [True]#, False]
pickup_nums_1 = [1000] # [int(i*len(tweets)) for i in [0.75, 0.50, 0.25, 0.10, 0.05]]
pickup_nums_2 = [100] # [int(i/10) for i in pickup_nums_1 if i > 10]
prob_plus_vals_pos = [0, 2, 5, 10]
prob_plus_vals_neg = [0, 2, 5, 10]
weight_close_vals = [0.75, 0.5, 0.25]
quick_ratio_vals = []
ratio_vals = [0.75, 0.5, 0.25]
Levenshtein_vals = []#[0.25, 0.5, 0.75]
stringComp_vals = []#[0.25, 0.5, 0.75]
duplicate_removal_flags = [True, False]
longest_similar_str_flags = [True, False]

def remove_duplicate_tweets(tweets_list):
    return_list = []
    while len(tweets_list) > 0:
        t = tweets_list[0]
        #using the quick_ratio() is very time consuming!
        #duplicates = [s for s in tweets_list if difflib.SequenceMatcher(None, s, t).quick_ratio() > 0.90]
        duplicates = [s for s in tweets_list if s==t]
        return_list.append([t, len(duplicates)])
        tweets_list = [x for x in tweets_list if x not in duplicates]
#        if len(duplicates)>1:
#            print 'stop for debugging purpose'
    return return_list

results_header = [
    'sent_thresh_pos', 'sent_thresh_neg', 'stp_wrd_flag', 'duplicates_removed_pos', 'duplicates_removed_neg',
    'prob_plus_val_pos', 'prob_plus_val_neg', 'pickup_no_1', 'pickup_no_2', 'weight_close_val', 'longest_similar_str_flag',
    'average_weight_pos', 'stdev_weight_pos', 'average_weight_neg', 'stdev_weight_neg',
    'difflib_qr_val', 'difflib_r_val', 'Levenshtein_val', 'string_comp_val', 'pos_tokens', 'neg_tokens']

results.append(results_header)

def __get_result(encompassing_toks_pos, encompassing_toks_neg, params, str_comp_type, string_sim_val):
    """
    params is a list of parameters. string_sim_func is the name of the function used to compare to strings
    (like quick_ratio(), Levenshtein, ...).
    """
    result = params # We add the results to the end of parameters list to have a column for pos/neg results.

    pickup_no_2 = params[results_header.index('pickup_no_2')]
    weight_similarity_val = params[results_header.index('weight_close_val')]
    longest_similar_str_flag = params[results_header.index('longest_similar_str_flag')]

    final_toks_pos = topic_funcs.find_similar_toks(encompassing_toks_pos, pickup_no_2, str_comp_type, string_sim_val, weight_similarity_val, longest_similar_str_flag)
    final_toks_pos = sorted(final_toks_pos.iteritems(), key=operator.itemgetter(1))
    final_toks_pos.reverse()
    final_toks_pos = final_toks_pos[:50]
    result.append(final_toks_pos)

    final_toks_neg = topic_funcs.find_similar_toks(encompassing_toks_neg, pickup_no_2, str_comp_type, string_sim_val, weight_similarity_val, longest_similar_str_flag)
    final_toks_neg = sorted(final_toks_neg.iteritems(), key=operator.itemgetter(1))
    final_toks_neg.reverse()
    final_toks_neg = final_toks_neg[:50]
    result.append(final_toks_neg)

    results.append(result)

for sent_pos in sentiment_thresholds_pos:
    for sent_neg in sentiment_thresholds_neg:
        tweets_pos = []
        tweets_neg = []
        # get all tweets from pair-list [twitt,centVal] which centVal>=b
        tweets_pos = [t_v[0] for t_v in tweets if t_v[1] >= sent_pos]
        # get all tweets from pair-list [twitt,centVal] which centVal<=b
        tweets_neg = [t_v[0] for t_v in tweets if t_v[1] <= sent_neg]

        for duplicate_removal_flag in duplicate_removal_flags:

            if duplicate_removal_flag:
                duplicate_count_pos = remove_duplicate_tweets(tweets_pos)
                duplicate_count_neg = remove_duplicate_tweets(tweets_neg)
                unique_tweets_pos = [d[0] for d in duplicate_count_pos]
                unique_tweets_neg = [d[0] for d in duplicate_count_neg]
                duplicates_pos = len(tweets_pos) - len(unique_tweets_pos)
                duplicates_neg = len(tweets_neg) - len(unique_tweets_neg)
                tweets_pos = unique_tweets_pos
                tweets_neg = unique_tweets_neg
            else:
                duplicates_pos = 0
                duplicates_neg = 0

            ngrams_pos = []
            ngrams_neg = []

            for stop_word_flag in stop_word_flags:
                for t in tweets_pos:
                    ngrams_pos = ngrams_pos + topic_funcs.ngrams(t.split(), lang, stop_word_flag)
                for t in tweets_neg:
                    ngrams_neg = ngrams_neg + topic_funcs.ngrams(t.split(), lang, stop_word_flag)

                freq_pos = topic_funcs.term_frequency(ngrams_pos, lang)
                freq_neg = topic_funcs.term_frequency(ngrams_neg, lang)

                for prob_plus_val_pos in prob_plus_vals_pos:
                    for prob_plus_val_neg in prob_plus_vals_neg:

                        weight_freq_pos = topic_funcs.weight_term_frequencies_one(freq_pos, freq_neg, prob_plus_val_pos)
                        weight_freq_pos = sorted(weight_freq_pos.iteritems(), key=operator.itemgetter(1))#sort on weights
                        weight_freq_pos.reverse() #large weights first

                        weight_freq_neg = topic_funcs.weight_term_frequencies_one(freq_neg, freq_pos, prob_plus_val_neg)
                        weight_freq_neg = sorted(weight_freq_neg.iteritems(), key=operator.itemgetter(1))#sort on weights
                        weight_freq_neg.reverse() #large weights first

                        average_weight_pos, stdev_weight_pos = math_extra.calc_mean_stdev([w[1] for w in weight_freq_pos])
                        average_weight_neg, stdev_weight_neg = math_extra.calc_mean_stdev([w[1] for w in weight_freq_neg])

                        for pickup_no_1 in pickup_nums_1:
                            #we are concerned in high probable toks, so only keep those with values above average!
    #                        weighted_toks_pos = [w for w in weight_freq_pos if w[1] >= average_weight_pos]
    #                        weighted_toks_neg = [w for w in weight_freq_neg if w[1] >= average_weight_neg]
                            weighted_toks_pos = weight_freq_pos[:pickup_no_1]
                            weighted_toks_neg = weight_freq_neg[:pickup_no_1]

                            for weight_close_val in weight_close_vals:

                                encompassing_toks_pos = topic_funcs.find_encompassing_toks2(weighted_toks_pos, weight_close_val)
                                encompassing_toks_neg = topic_funcs.find_encompassing_toks2(weighted_toks_neg, weight_close_val)

                                for pickup_no_2 in pickup_nums_2:

                                    for longest_similar_str_flag in longest_similar_str_flags:

                                        for quick_ratio_val in quick_ratio_vals:
                                            params = [sent_pos, sent_neg, stop_word_flag, duplicates_pos, duplicates_neg,
                                                      prob_plus_val_pos, prob_plus_val_neg, pickup_no_1, pickup_no_2,
                                                      weight_close_val, longest_similar_str_flag,
                                                      average_weight_pos, stdev_weight_pos, average_weight_neg, stdev_weight_neg,
                                                    quick_ratio_val, 'NA', 'NA', 'NA']

                                            __get_result(encompassing_toks_pos, encompassing_toks_neg, params, 'quick_ratio', quick_ratio_val)

                                        for ratio_val in ratio_vals:
                                            params = [sent_pos, sent_neg, stop_word_flag, duplicates_pos, duplicates_neg,
                                                      prob_plus_val_pos, prob_plus_val_neg, pickup_no_1, pickup_no_2,
                                                      weight_close_val, longest_similar_str_flag,
                                                      average_weight_pos, stdev_weight_pos, average_weight_neg, stdev_weight_neg,
                                                      'NA', ratio_val, 'NA', 'NA']

                                            __get_result(encompassing_toks_pos, encompassing_toks_neg, params, 'ratio', ratio_val)

                                        for Levenshtein_val in Levenshtein_vals:
                                            params = [sent_pos, sent_neg, stop_word_flag, duplicates_pos, duplicates_neg,
                                                      prob_plus_val_pos, prob_plus_val_neg, pickup_no_1, pickup_no_2,
                                                      weight_close_val, longest_similar_str_flag,
                                                      average_weight_pos, stdev_weight_pos, average_weight_neg, stdev_weight_neg,
                                                      'NA', 'NA', Levenshtein_val, 'NA']

                                            __get_result(encompassing_toks_pos, encompassing_toks_neg, params, 'Levenshtein', Levenshtein_val)

                                        for stringComp_val in stringComp_vals:
                                            params = [sent_pos, sent_neg, stop_word_flag, duplicates_pos, duplicates_neg,
                                                     prob_plus_val_pos, prob_plus_val_neg, pickup_no_1, pickup_no_2,
                                                     weight_close_val, longest_similar_str_flag,
                                                     average_weight_pos, stdev_weight_pos, average_weight_neg, stdev_weight_neg,
                                                     'NA', 'NA', 'NA', stringComp_val]

                                            __get_result(encompassing_toks_pos, encompassing_toks_neg, params, 'stringComp', stringComp_val)

my_util.write_csv_file(saving_dir + '/results', tab_delimiter_flag, unicode_flag, results)
my_util.pickle_to_file(saving_dir + '/results_pickled', results)

#The following loop compares every unique pair of elements in results and gives a value indicating how similar the
#tokesn are.
comparisons = []
most_dissimilars_pos = []
most_dissimilars_neg = []
comparison_header  =  results_header[:results_header.index('pos_tokens')] + ['Empty'] \
                    + results_header[:results_header.index('pos_tokens')] + ['Empty'] \
                    + ['dissim_val_pos'] + ['toks1_pos'] + ['toks2_pos'] + ['diff12_pos'] + ['diff21_pos'] \
                    + ['dissim_val_neg'] + ['toks1_neg'] + ['toks2_neg'] + ['diff12_neg'] + ['diff21_neg']
comparisons.append(comparison_header)
most_dissimilars_pos.append(comparison_header)
most_dissimilars_neg.append(comparison_header)
for i in range(1, len(results)): #The first row is the header so we start from he second row (i.e. from index1)
    for j in range(i + 1, len(results)):
        r1 = results[i]
        r2 = results[j]
        params1 = r1[:results_header.index('pos_tokens')]
        params2 = r2[:results_header.index('pos_tokens')]
        t1_pos = r1[results_header.index('pos_tokens')]
        t2_pos = r2[results_header.index('pos_tokens')]
        t1_neg = r1[results_header.index('neg_tokens')]
        t2_neg = r2[results_header.index('neg_tokens')]
        comp_val_pos, diff12_pos, diff21_pos = tokens_similarity.compare_tokens(t1_pos, t2_pos)
        comp_val_neg, diff12_neg, diff21_neg = tokens_similarity.compare_tokens(t1_neg, t2_neg)
        comparison_result = params1 + [''] + params2 + [''] + [comp_val_pos] + [t1_pos] + [t2_pos] + [diff12_pos]\
                            + [diff21_pos] + [comp_val_neg] + [t1_neg] + [t2_neg] + [diff12_neg] + [diff21_neg]
        comparisons.append(comparison_result)
        if comp_val_pos > 0.55:
            most_dissimilars_pos.append(comparison_result)

        if comp_val_neg > 0.55:
            most_dissimilars_neg.append(comparison_result)

my_util.write_csv_file(saving_dir + '/comparisons', tab_delimiter_flag, unicode_flag, comparisons)
my_util.write_csv_file(saving_dir + '/most_dissimilar_pos', tab_delimiter_flag, unicode_flag, most_dissimilars_pos)
my_util.write_csv_file(saving_dir + '/most_dissimilar_neg', tab_delimiter_flag, unicode_flag, most_dissimilars_neg)

#The following loop goes through the results list and takes the 10th (11th) elements which is a list of positive
#(negative) tokens and associated weights. We only takes the tokens and list them in a separate list.
all_toks_pos = []
all_toks_neg = []
for i in range(1, len(results)): #The first row is the header so we start from he second row (i.e. from index1)
    r = results[i]
    t_w_pos = r[results_header.index('pos_tokens')]
    for t_w in t_w_pos:
        t = t_w[0]
        # remove the white spaces before/after the string (some tokens appears to have white spaces before/after them).
        t = t.strip()
        if not(t in all_toks_pos):
            all_toks_pos.append(t)
            #        else:
            #            print 'Pos dropped: ', t

    t_w_neg = r[results_header.index('neg_tokens')]
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
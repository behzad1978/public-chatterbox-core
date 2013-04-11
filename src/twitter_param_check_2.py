import topic_funcs
import operator
import getter
import pickle
import csv
import tokens_similarity
from collections import Counter


lang ='en'
tweets=[]
results=[]
final_toks_qr_pos_prev=[]
final_toks_qr_neg_prev=[]
final_toks_r_pos_prev=[]
final_toks_r_neg_prev=[]
final_toks_l_pos_prev=[]
final_toks_l_neg_prev=[]
final_toks_sc_pos_prev=[]
final_toks_sc_neg_prev=[]

f_w = open('results1.csv', 'w')
tab_delimiter = False
file_header_written = True

try:
    file_read = open('the_tweets_file', 'r')
    tweets = pickle.load(file_read)
    file_read.close()
except IOError:
    import getter
    tweets = getter.get_tweets()
    file_write = open('the_tweets_file', 'w')
    pickle.dump(tweets, file_write)
    file_write.close()

sentiment_thresholds_pos = [0.25] #[i*0.1 for i in range(0,10)]
sentiment_thresholds_neg = [-0.25] #[-1*i for i in sentiment_borders_pos] # -1*sentiment_border_pos
stop_word_flags = [True]#, False]
pickup_nums_1 = [500] #[int(i*len(tweets))for i in [0.75, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.025]]
pickup_nums_2 = [50] #[int(i/10) for i in pickup_nums_1 if i>10]
prob_plus_vals = [2, 5]#[0, 0.5, 1, 1.5, 2, 2.5, 5]
weight_similarity_vals = [0.25, 0.5, 0.75]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
quick_ratio_vals = [0.25, 0.5, 0.75]#, 0.7, 0.8, 0.85, 0.9, 0.95]
ratio_vals = [0.25, 0.5, 0.75]#[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
Levenshtein_vals = [0.25, 0.5, 0.75]#, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
stringComp_vals = [0.25, 0.5, 0.75]#, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
quick_ratio_flag = True
ratio_flag = True
Levenshtein_flag = True
stringComp_flag = True

file_header = [
    'sent_thresh_pos', 'sent_thresh_neg', 'stp_wrd_flag', 'prob_plus_val', 'pickup_no_1', 'weight_sim_val',
    'pickup_no_2', 'difflib_qr_val', 'difflib_r_val', 'Levenshtein_val', 'string_comp_val', 'pos_tokens', 'neg_tokens']
    #,'diff_thisRow_topRow_pos', 'diff_topRow_thisRow_pos', 'diff_thisRow_topRow_neg', 'diff_topRow_thisRow_neg']

def __write_to_file(result):
    if tab_delimiter:
        csv.writer(f_w,'excel-tab').writerow(result)
    else:
        csv.writer(f_w).writerow(result)

__write_to_file(file_header)

for sent_pos in sentiment_thresholds_pos:
    for sent_neg in sentiment_thresholds_neg:

        tweets_pos=[]
        tweets_neg=[]
        tweets_pos = [t_v[0] for t_v in tweets if t_v[1]>=sent_pos] # get all tweets from pair-list [twitt,centVal] which centVal>=b
        tweets_neg = [t_v[0] for t_v in tweets if t_v[1]<=sent_neg] # get all tweets from pair-list [twitt,centVal] which centVal<=b

        ngrams_pos=[]
        ngrams_neg=[]

        for stop_word_flag in stop_word_flags:

            for t in tweets_pos:
                ngrams_pos = ngrams_pos + topic_funcs.ngrams(t.split(), lang, stop_word_flag)
            for t in tweets_neg:
                ngrams_neg = ngrams_neg + topic_funcs.ngrams(t.split(), lang, stop_word_flag)

            freq_pos = topic_funcs.term_frequency(ngrams_pos, lang)
            freq_neg = topic_funcs.term_frequency(ngrams_neg, lang)

            for prob_plus_val in prob_plus_vals:

                weight_freq_pos = topic_funcs.weight_term_frequencies_one(freq_pos, freq_neg, prob_plus_val)
                weight_freq_pos = sorted(weight_freq_pos.iteritems(), key=operator.itemgetter(1))
                weight_freq_pos.reverse() #large values first

                weight_freq_neg = topic_funcs.weight_term_frequencies_one(freq_neg, freq_pos, prob_plus_val)
                weight_freq_neg = sorted(weight_freq_neg.iteritems(), key=operator.itemgetter(1))
                weight_freq_neg.reverse() #large values first

                for pickup_no_1 in pickup_nums_1:

                    weighted_toks_pos = weight_freq_pos[:pickup_no_1]
                    weighted_toks_neg = weight_freq_neg[:pickup_no_1]

                    for weight_similarity_val in weight_similarity_vals:

                        encompassing_toks_pos = topic_funcs.find_encompassing_toks(weighted_toks_pos, weight_similarity_val)
                        encompassing_toks_neg = topic_funcs.find_encompassing_toks(weighted_toks_neg, weight_similarity_val)

                        for pickup_no_2 in pickup_nums_2:

                            if quick_ratio_flag:
                                for quick_ratio_val in quick_ratio_vals:

                                    result = [sent_pos, sent_neg, stop_word_flag, prob_plus_val, pickup_no_1, weight_similarity_val, pickup_no_2, quick_ratio_val, 'NA', 'NA', 'NA']

                                    final_toks_qr_pos = topic_funcs.find_similar_toks(encompassing_toks_pos, pickup_no_2, 'quick_ratio', quick_ratio_val, weight_similarity_val)
                                    final_toks_qr_pos = sorted(final_toks_qr_pos.iteritems(), key=operator.itemgetter(1))
                                    final_toks_qr_pos.reverse()
                                    final_toks_qr_pos=final_toks_qr_pos[:20]
                                    result.append(final_toks_qr_pos)

                                    final_toks_qr_neg = topic_funcs.find_similar_toks(encompassing_toks_neg, pickup_no_2, 'quick_ratio', quick_ratio_val, weight_similarity_val)
                                    final_toks_qr_neg = sorted(final_toks_qr_neg.iteritems(), key=operator.itemgetter(1))
                                    final_toks_qr_neg.reverse()
                                    final_toks_qr_neg = final_toks_qr_neg[:20]
                                    result.append(final_toks_qr_neg)

                                    __write_to_file(result)
                                    results.append(result)

                            if ratio_flag:
                                for ratio_val in ratio_vals:

                                    result = [sent_pos, sent_neg, stop_word_flag, prob_plus_val, pickup_no_1, weight_similarity_val, pickup_no_2, 'NA', ratio_val, 'NA', 'NA']

                                    final_toks_r_pos = topic_funcs.find_similar_toks(encompassing_toks_pos, pickup_no_2, 'ratio', ratio_val, weight_similarity_val)
                                    final_toks_r_pos = sorted(final_toks_r_pos.iteritems(), key=operator.itemgetter(1))
                                    final_toks_r_pos.reverse()
                                    final_toks_r_pos = final_toks_r_pos[:20]
                                    result.append(final_toks_r_pos)

                                    final_toks_r_neg = topic_funcs.find_similar_toks(encompassing_toks_neg, pickup_no_2, 'ratio', ratio_val, weight_similarity_val)
                                    final_toks_r_neg = sorted(final_toks_r_neg.iteritems(), key=operator.itemgetter(1))
                                    final_toks_r_neg.reverse()
                                    final_toks_r_neg = final_toks_r_neg[:20]
                                    result.append(final_toks_r_neg)

                                    __write_to_file(result)
                                    results.append(result)

                            if Levenshtein_flag:
                                for Levenshtein_val in Levenshtein_vals:

                                    result = [sent_pos, sent_neg, stop_word_flag, prob_plus_val, pickup_no_1, weight_similarity_val, pickup_no_2, 'NA', 'NA', Levenshtein_val, 'NA']

                                    final_toks_l_pos = topic_funcs.find_similar_toks(encompassing_toks_pos, pickup_no_2, 'Levenshtein', Levenshtein_val, weight_similarity_val)
                                    final_toks_l_pos = sorted(final_toks_l_pos.iteritems(), key=operator.itemgetter(1))
                                    final_toks_l_pos.reverse()
                                    final_toks_l_pos = final_toks_l_pos[:20]
                                    result.append(final_toks_l_pos)

                                    final_toks_l_neg = topic_funcs.find_similar_toks(encompassing_toks_neg, pickup_no_2, 'Levenshtein', Levenshtein_val, weight_similarity_val)
                                    final_toks_l_neg = sorted(final_toks_l_neg.iteritems(), key=operator.itemgetter(1))
                                    final_toks_l_neg.reverse()
                                    final_toks_l_neg = final_toks_l_neg[:20]
                                    result.append(final_toks_l_neg)

                                    __write_to_file(result)
                                    results.append(result)

                            if stringComp_flag:
                                for stringComp_val in stringComp_vals:

                                    result = [sent_pos, sent_neg, stop_word_flag, prob_plus_val, pickup_no_1, weight_similarity_val, pickup_no_2, 'NA', 'NA', 'NA', stringComp_val]

                                    final_toks_sc_pos = topic_funcs.find_similar_toks(encompassing_toks_pos, pickup_no_2, 'stringComp', stringComp_val, weight_similarity_val)
                                    final_toks_sc_pos = sorted(final_toks_sc_pos.iteritems(), key=operator.itemgetter(1))
                                    final_toks_sc_pos.reverse()
                                    final_toks_sc_pos = final_toks_sc_pos[:20]
                                    result.append(final_toks_sc_pos)

                                    final_toks_sc_neg = topic_funcs.find_similar_toks(encompassing_toks_neg, pickup_no_2, 'stringComp', stringComp_val, weight_similarity_val)
                                    final_toks_sc_neg = sorted(final_toks_sc_neg.iteritems(), key=operator.itemgetter(1))
                                    final_toks_sc_neg.reverse()
                                    final_toks_sc_neg = final_toks_sc_neg[:20]
                                    result.append(final_toks_sc_neg)

                                    __write_to_file(result)
                                    results.append(result)

f_w.close()

comparisons=[]
for i in range(0, len(results)):
    for j in range(i+1, len(results)):
        r1 = results[i]
        r2 = results[j]
        params1 = r1[0:11]
        params2 = r2[0:11]
        t1_pos = r1[11]
        t2_pos = r2[11]
        t1_neg = r1[12]
        t2_neg = r2[12]
        comparisons_pos = tokens_similarity.compare_tokens(t1_pos, t2_pos)
        comparisons_neg = tokens_similarity.compare_tokens(t1_neg, t2_neg)
        comparison_result = [params1, params2, comparisons_pos, comparisons_neg]
        comparisons.append(comparison_result)

most_dissimilars_pos = [d for d in comparisons if d[2]>0.75]
most_dissimilars_neg = [d for d in comparisons if d[3]>0.75]
f2 = open('comparisons1.csv', 'w')
csv.writer(f2).writerows(comparisons)

f3 = open('most_dissimilar_pos1.csv', 'w')
csv.writer(f3).writerows(most_dissimilars_pos)

f4 = open('most_dissimilar_neg1.csv', 'w')
csv.writer(f3).writerows(most_dissimilars_neg)

#results_pos = [r[11] for r in results]
#comparisons_pos = [tokens_similarity.compare_tokens(l1, l2) for l1 in results_pos for l2 in results_pos if (l1<>l2)]


import topicTest
import operator
import getter
import pickle

lang ='en'
tweets=[]
results=[]

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

sentiment_borders_pos = [0.7] #[i*0.1 for i in range(0,10)]
sentiment_borders_neg = [-0.7] #[-1*i for i in sentiment_borders_pos] # -1*sentiment_border_pos
stop_word_flags = [True, False]
pickup_nums_1 = [int(i*len(tweets))for i in [0.75, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.025]]
pickup_nums_2 = [int(i/10) for i in pickup_nums_1 if i>10]
prob_plus_vals = [0, 0.5, 1, 1.5, 2, 2.5, 5]
weight_similarity_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
quick_ratio_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
ratio_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
Levenshtein_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
stringcomp_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]



for b_pos in sentiment_borders_pos:
    for b_neg in sentiment_borders_neg:

        tweets_pos=[]
        tweets_neg=[]
        tweets_pos = [t_v[0] for t_v in tweets if t_v[1]>=b_pos] # get all tweets from pair-list [twitt,centVal] which centVal>=b
        tweets_neg = [t_v[0] for t_v in tweets if t_v[1]<=b_neg] # get all tweets from pair-list [twitt,centVal] which centVal<=b

        ngrams_pos=[]
        ngrams_neg=[]

        for stop_word_flag in stop_word_flags:

            for t in tweets_pos:
                ngrams_pos = ngrams_pos + topicTest.ngrams(t.split(), lang, stop_word_flag)

            for t in tweets_neg:
                ngrams_neg = ngrams_neg + topicTest.ngrams(t.split(), lang, stop_word_flag)

            freq_pos = topicTest.term_frequency(ngrams_pos, lang)
            freq_neg = topicTest.term_frequency(ngrams_neg, lang)

            for prob_plus_val in prob_plus_vals:

                weight_freq_pos = topicTest.weight_term_frequencies_one(freq_pos, freq_neg, prob_plus_val)
                weight_freq_pos = sorted(weight_freq_pos.iteritems(), key=operator.itemgetter(1))
                weight_freq_pos.reverse() #large values first

                weight_freq_neg = topicTest.weight_term_frequencies_one(freq_neg, freq_pos, prob_plus_val)
                weight_freq_neg = sorted(weight_freq_neg.iteritems(), key=operator.itemgetter(1))
                weight_freq_neg.reverse() #large values first

                for pickup_no_1 in pickup_nums_1:

                    sorted_toks_pos = weight_freq_pos[:pickup_no_1]
                    sorted_toks_neg = weight_freq_neg[:pickup_no_1]

                    for weight_similarity_val in weight_similarity_vals:
                        encompassing_toks_pos = topicTest.find_encompassing_str(sorted_toks_pos, weight_similarity_val)
                        encompassing_toks_neg = topicTest.find_encompassing_str(sorted_toks_neg, weight_similarity_val)

                        for pickup_no_2 in pickup_nums_2:

                            for quick_ratio_val in quick_ratio_vals:
                                final_toks_qr_pos = topicTest.find_longest_similar_str(encompassing_toks_pos, pickup_no_2, 'quick_ratio', quick_ratio_val, weight_similarity_val)
                                final_toks_qr_pos = sorted(final_toks_qr_pos.iteritems(), key=operator.itemgetter(1))
                                final_toks_qr_pos.reverse()
                                #this is list of parameters and labels: ['label1', p1, 'label2', p2, 'label3', p3, 'label4']
                                result = ['sent_border_pos', b_pos, 'sent_border_neg', b_neg, 'stp_w_flag', stop_word_flag, 'prob_plus_val', prob_plus_val,
                                          'pickup_no_1', pickup_no_1, 'weight_sim_val', weight_similarity_val,
                                          'pickup_no_2', pickup_no_2, 'quick_ratio', quick_ratio_val, 'pos_tokens']
                                #another list will be added to the end: ['label1', p1, 'label2', p2, 'label3', p3, 'label4', [elements related to label4]]
                                result.append(final_toks_qr_pos)

                                final_toks_qr_neg = topicTest.find_longest_similar_str(encompassing_toks_neg, pickup_no_2, 'quick_ratio', quick_ratio_val, weight_similarity_val)
                                final_toks_qr_neg = sorted(final_toks_qr_neg.iteritems(), key=operator.itemgetter(1))
                                final_toks_qr_neg.reverse()

                                result+['neg_tokens']
                                result.append(final_toks_qr_neg)

                                results.append(result)


                            for ratio_val in ratio_vals:
                                final_toks_r_pos = topicTest.find_longest_similar_str(encompassing_toks_pos, pickup_no_2, 'ratio', ratio_val, weight_similarity_val)
                                final_toks_r_pos = sorted(final_toks_r_pos.iteritems(), key=operator.itemgetter(1))
                                final_toks_r_pos.reverse()

                                final_toks_r_neg = topicTest.find_longest_similar_str(encompassing_toks_neg, pickup_no_2, 'ratio', ratio_val, weight_similarity_val)
                                final_toks_r_neg = sorted(final_toks_r_neg.iteritems(), key=operator.itemgetter(1))
                                final_toks_r_neg.reverse()

                            for Levenshtein_val in Levenshtein_vals:
                                final_toks_l_pos = topicTest.find_longest_similar_str(encompassing_toks_pos, pickup_no_2, 'Levenshtein', Levenshtein_val, weight_similarity_val)
                                final_toks_l_pos = sorted(final_toks_l_pos.iteritems(), key=operator.itemgetter(1))
                                final_toks_l_pos.reverse()

                                final_toks_l_neg = topicTest.find_longest_similar_str(encompassing_toks_neg, pickup_no_2, 'Levenshtein', Levenshtein_val, weight_similarity_val)
                                final_toks_l_neg = sorted(final_toks_l_neg.iteritems(), key=operator.itemgetter(1))
                                final_toks_l_neg.reverse()

                            for stringcomp_val in stringcomp_vals:
                                final_toks_sc_pos = topicTest.find_longest_similar_str(encompassing_toks_pos, pickup_no_2, 'stringComp', stringcomp_val, weight_similarity_val)
                                final_toks_sc_pos = sorted(final_toks_sc_pos.iteritems(), key=operator.itemgetter(1))
                                final_toks_sc_pos.reverse()

                                final_toks_sc_neg = topicTest.find_longest_similar_str(encompassing_toks_neg, pickup_no_2, 'stringComp', stringcomp_val, weight_similarity_val)
                                final_toks_sc_neg = sorted(final_toks_sc_neg.iteritems(), key=operator.itemgetter(1))
                                final_toks_sc_neg.reverse()

                                result = [b_pos, b_neg, stop_word_flag, prob_plus_val, pickup_no_1, weight_similarity_val, pickup_no_2, ]
import topic_funcs
import operator
import pickle
import stringcomp
import nltk

lang='en'
tweets=[]

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

sentiment_borders_pos = [i*0.1 for i in range(0,10)]
sentiment_borders_neg = [-1*i for i in sentiment_borders_pos] # -1*sentiment_border_pos
stop_word_flags = [True, False] 
pickup_nums_1 = [i*len(tweets) for i in [0.75, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.025]]
pickup_nums_2 = [i/10 for i in pickup_nums_1 if i>10]
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
				ngrams_pos = ngrams_pos + topic_funcs.ngrams(t.split(), lang, stop_word_flag)
			
			for t in tweets_neg:
				ngrams_neg = ngrams_neg + topic_funcs.ngrams(t.split(), lang)

			freq_pos = topic_funcs.term_frequency(ngrams_pos, lang)

			freq_neg = topic_funcs.term_frequency(ngrams_neg, lang)

			weight_freq = topic_funcs.weight_term_frequencies_one(freq_pos, freq_neg)
			weight_freq = sorted(weight_freq.iteritems(), key=operator.itemgetter(1))
			weight_freq.reverse() #large values first

			for pickup_no_1 in pickup_nums_1:
                sorted_toks = weight_freq[:pickup_no_1]
                for weight_similarity_val in weight_similarity_vals:
                    encompassing_toks = topic_funcs.find_encompassing_toks(sorted_toks, weight_similarity_val)
                    for pickup_no_2 in pickup_nums_2:



                        for quick_ratio_val in quick_ratio_vals:
                            very_final_toks = topic_funcs.find_similar_toks(encompassing_toks, pickup_no_2, 'quick_ratio', quick_ratio_val, weight_similarity_val)
                            very_final_toks = sorted(very_final_toks.iteritems(), key=operator.itemgetter(1))
                            very_final_toks.reverse()

                        for ratio_val in ratio_vals:
                            very_final_toks = topic_funcs.find_similar_toks(encompassing_toks, pickup_no_2, 'ratio', ratio_val, weight_similarity_val)
                            very_final_toks = sorted(very_final_toks.iteritems(), key=operator.itemgetter(1))
                            very_final_toks.reverse()

                        for Levenshtein_val in Levenshtein_vals:
                            very_final_toks = topic_funcs.find_similar_toks(encompassing_toks, pickup_no_2, 'Levenshtein', Levenshtein_val, weight_similarity_val)
                            very_final_toks = sorted(very_final_toks.iteritems(), key=operator.itemgetter(1))
                            very_final_toks.reverse()

                        for stringcomp_val in stringcomp_vals:
                        very_final_toks = topic_funcs.find_similar_toks(encompassing_toks, pickup_no_2, 'stringComp', stringcomp_val, weight_similarity_val)
                        very_final_toks = sorted(very_final_toks.iteritems(), key=operator.itemgetter(1))
                        very_final_toks.reverse()







# print ngrams_pos, '\n\n', ngrams_neg, '\n\n', freq_pos, '\n\n', freq_neg, '\n\n', weight_freq


# pos = ["Obama cancels Tuesday campaign event to focus on storm Sandy http://reut.rs/Ua29D4" ,\

# 	  "@MonicaCrowley My concern with Obama is some scandals in WH Admin. PR disaster w/Benghazi Attack. Why would POTUS fly to Vegas day after?" ,\

#       "@MonicaCrowley Obama proposed 4 trillion in spending cuts first term gutless Republicans ran and hid like school girls 'cowards' Your Gone" ,\

# 	  "Wind can't stop me, rain can't stop me. I'll be calling. I'll be knocking. It's crunch time now. - Obama campaign volunteer in Florida" ,\

# 	  "I bet Romney gonna blame Obama for this hurricane."]


# neg = ["Real spending cuts under Obama? Never. Tax hikes? Duh! --> Obama: 2nd term would be 'mandate' for cuts, tax increases" ,\

# 	  "Well, now---> New Projection: Romney 52, Obama 47" ,\

# 	  "Charles Woods to Obama: Better to die the death of a hero than it is to live the life of a coward" ,\

# 	  "2 months, 3 weeks, 16 hours, 59 minutes, and 59 seconds until #Obama leaves office." ,\

# 	  "Fox News: Obama's response in advance of the aftermath of Hurricane Sandy is going to have been 'pathetic'" ]
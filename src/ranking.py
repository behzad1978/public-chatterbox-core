__author__ = 'behzadbehzadan'

import my_util
from operator import itemgetter

match_old_file_to_new = False

#This function gets old_toks_ranks which has tokens in one column and one of the letters of 'g','b' and 'n' in another
#column representing whether the tok is good, bad or neutral. The new_toks is the list of all tokens generated by a new
#parameter settings. Therefore, the new_toks may have some tokens that are not in the old_toks_ranks. The function
#copies all the old ranks into the matching toks in the new_toks to save writing good, bad, neutral again for the new
#tokens.
def match_old_ranks_to_new_ranks(old_toks_ranks, new_toks):
    new_toks_ranks=[]
    for t in new_toks:
        t=t[0] #tweets are stored in a list: [t1], [t2], [t3], ...]
        try:
            rank = old_toks_ranks[t]
        except KeyError:
            rank = None
            print t, " :could not be found in the dictionary"
        new_toks_ranks.append([t,rank])
    return new_toks_ranks

#Read the list of all unique final ngrams that are produced by topic.py
toks_pos = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/final_toks_pos', False, True)
toks_neg = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/final_toks_neg', False, True)

if match_old_file_to_new==True:
    #Read a former (old) list of ngrams which are already ranked by bad, good, neutral ('b','g','n') in front of them.
    toks_ranks_pos_old = my_util.read_csv_file(
        '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/toks_ranks_pos', False, unicode_flag=True)
    toks_ranks_neg_old = my_util.read_csv_file(
        '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/toks_ranks_neg', False, unicode_flag=True)
    #make a dictionary of the first and second column.
    toks_ranks_pos_old = dict([(x[0], x[1]) for x in toks_ranks_pos_old])
    toks_ranks_neg_old = dict([(x[0], x[1]) for x in toks_ranks_neg_old])
    #find the previously rated tokens and look if they exist in the new list too. If they do, copy the rank next to it.
    toks_ranks_pos_temp = match_old_ranks_to_new_ranks(toks_ranks_pos_old, toks_pos)
    toks_ranks_neg_temp = match_old_ranks_to_new_ranks(toks_ranks_neg_old, toks_neg)
    my_util.write_csv_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/toks_ranks_pos', False, True, toks_ranks_pos_temp)
    my_util.write_csv_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/toks_ranks_neg', False, True, toks_ranks_neg_temp)

elif match_old_file_to_new == False:
#For a set of parameter settings, results was stored as results. We pickled the results to be able to retrieve it as
#its original form. For example results[11], is a list of tokens. If we don't pickle results and as an alternative save
# it for example as text, the results[11] would be a long string rather than a list of strings (i.e. rather than a list
#of tokens)
    #Read the new list of ranked ngrams by bad, good, neutral ('b','g','n') in front of them.
    toks_ranks_pos = my_util.read_csv_file(
        '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/toks_ranks_pos', False,
        unicode_flag=True)
    toks_ranks_neg = my_util.read_csv_file(
        '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/toks_ranks_neg', False,
        unicode_flag=True)
    #make a dictionary of the first and second column.
    toks_ranks_pos = dict([(x[0], x[1]) for x in toks_ranks_pos])
    toks_ranks_neg = dict([(x[0], x[1]) for x in toks_ranks_neg])
    #Alternative way of creating dictionary from list of lists:
    #toks_ranking_pos = dict( zip([x[0] for x in toks_ranking_pos] , [y[1] for y in toks_ranking_pos]) )
    #toks_ranking_neg = dict( zip([x[0] for x in toks_ranking_neg] , [y[1] for y in toks_ranking_neg]) )

    results = my_util.read_pickled_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/results_pickled')
    rankings = []
    results_header = results[0]
    ranking_header = results_header[:results_header.index('final_toks_pos')] + \
                     [''] + ['good_toks_pos'] + ['bad_toks_pos'] + ['neutral_toks_pos'] + ['rating_pos'] + \
                     [''] + ['good_toks_neg'] + ['bad_toks_neg'] + ['neutral_toks_neg'] + ['rating_neg'] + \
                     [''] + ['ranking_total'] + [''] + ['toks_pos'] + ['toks_neg']
    rankings.append(ranking_header)
    for i in range(1,len(results)): #The first row is the header so we start from he second row (i.e. from index1)
        result = results[i]
        t_w_pos = result[results_header.index('final_toks_pos')]
        good_toks_pos = 0
        bad_toks_pos = 0
        neutral_toks_pos = 0
        t_w_r_pos=[]
        for t_w in t_w_pos:
            t = t_w[0]
            t = t.strip()
            try:
                rank = toks_ranks_pos[t]
                if rank == 'g':
                    good_toks_pos += 1
                elif rank == 'b':
                    bad_toks_pos += 1
                else:
                    neutral_toks_pos += 1
                #add the rank (g, b, n) at the end of tweet for better visualisation: (tweet, weight, rank)
                #note that we cannot change tuple, so we create new ones and add them to a new list.
                t_w_r = t_w + (rank,)
                t_w_r_pos.append(t_w_r)
            except KeyError:
                print "A ranking could not be found in the positive dictionary"
        ranking = result[:results_header.index('final_toks_pos')] + [''] + [good_toks_pos] + [-bad_toks_pos] + [neutral_toks_pos] + [good_toks_pos - bad_toks_pos]

        t_w_neg = result[results_header.index('final_toks_neg')]
        good_toks_neg = 0
        bad_toks_neg = 0
        neutral_toks_neg = 0
        t_w_r_neg=[]
        for t_w in t_w_neg:
            t = t_w[0]
            t = t.strip()
            try:
                rank = toks_ranks_neg[t]
                if rank == 'g':
                    good_toks_neg += 1
                elif rank == 'b':
                    bad_toks_neg += 1
                else:
                    neutral_toks_neg += 1
                #add the rank (g, b, n) at the end of tweet for better visualisation!
                #note that we cannot change tuple, so we create new ones and add them to a new list.
                t_w_r = t_w + (rank,)
                t_w_r_neg.append(t_w_r)
            except KeyError:
                print "A ranking could not be found in the negative dictionary"
        ranking = ranking + [''] + [good_toks_neg] + [-bad_toks_neg] + [neutral_toks_neg] + [good_toks_neg - bad_toks_neg] \
                          + [''] + [good_toks_pos+good_toks_neg-bad_toks_pos-bad_toks_neg] + [''] + [t_w_r_pos] + [t_w_r_neg]
        rankings.append(ranking)

    indx = ranking_header.index('ranking_total')
    rankings = sorted(rankings, key=itemgetter(indx))
    rankings.reverse()

    my_util.write_csv_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results12_qr/rankings', False, True, rankings)
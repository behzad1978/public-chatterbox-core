__author__ = 'behzadbehzadan'

import my_util
from operator import itemgetter


#This code updates a new dictionary of tokens with values g,b,n based on the former dictionaries.
#For the first time, we label (by putting 'b','g','n' in front of) the positive/negative tokens produced by a set of experiments stored in the files
#final_toks_pos_1' and 'final_toks_neg_1'. When we do a second set of experiments, we end up having 'final_toks_pos_2'
#and 'final_toks_neg_2. Some of the tokens in the latter two files are similar to the former two files. The program
#ranking.py finds these similar toks and copy the labels from the former files to the latter files (when match_old_ranks_to_new_ranks = True).
#when the process is finished we manually put label in front of other tokens which did not exist in the former file and
#therefore do not have any label in front of them in the latter file. After labeling these remaining tokens manually, we
#use the current codes to update the two dictionary which contains the tokens and labels of both dictionaries.


#Read the dict which needs to be updated:
toks_ranks_pos = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results11_qr/toks_ranks_pos', False, True)
toks_ranks_neg = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results11_qr/toks_ranks_neg', False, True)

# Read a former list of ngrams which are already ranked by bad, good, neutral ('b','g','n') in front of them.
toks_ranks_pos_old = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results10_qr/toks_ranks_pos', False, unicode_flag=True)
toks_ranks_neg_old = my_util.read_csv_file(
    '/Users/behzadbehzadan/chatterbox-core/twitter_param_results/Results10_qr/toks_ranks_neg', False, unicode_flag=True)

#make a dictionary of the first and second column.
toks_ranks_pos = dict([(x[0], x[1]) for x in toks_ranks_pos])
toks_ranks_neg = dict([(x[0], x[1]) for x in toks_ranks_neg])

toks_ranks_pos_old = dict([(x[0], x[1]) for x in toks_ranks_pos_old])
toks_ranks_neg_old = dict([(x[0], x[1]) for x in toks_ranks_neg_old])

toks_ranks_pos.update(toks_ranks_pos_old)
toks_ranks_neg.update(toks_ranks_neg_old)

# In order to save the updated file in a csv form we need to have a list of list of each pair:
toks_ranks_pos = zip(toks_ranks_pos.keys(), toks_ranks_pos.values())
toks_ranks_neg = zip(toks_ranks_neg.keys(), toks_ranks_neg.values())

#save the update dictionary to a folder to use in future.
my_util.write_csv_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/toks_ranks_pos', False, True, toks_ranks_pos)
my_util.write_csv_file('/Users/behzadbehzadan/chatterbox-core/twitter_param_results/toks_ranks_neg', False, True, toks_ranks_neg)

##create a dictionary having None values from the new toks to be able to update them.
#toks_ranks_pos_new = dict(zip(toks_pos), [None] * len(toks_pos))
#toks_ranks_neg_new = dict(zip(toks_neg), [None] * len(toks_neg))
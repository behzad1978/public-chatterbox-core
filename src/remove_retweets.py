__author__ = 'behzadbehzadan'

import os
import my_util
import funcs_worry

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/Worry/Sources/'
collection_name = 'relaxed'
file_name = 'source' + '_' + collection_name
saving_file_name = file_name + '_noDup'
use_qr_to_remove_dups = False

#read the source file --> [[text1], [text2], [test3], ...]
tweets = my_util.read_csv_file(home_dir + source_dir + file_name, False, True)
original_n_tweets = len(tweets)
#create list of texts --> [text1, text2, text3, ...]
tweets = [t[0] for t in tweets]
tweets = funcs_worry.remove_retweets(tweets, use_qr_to_remove_dups)
final_n_tweets = len(tweets)
my_util.write_csv_file(home_dir + source_dir + saving_file_name, False, True, [[t] for t in tweets])
print 'original_n_tweets:', original_n_tweets
print 'final_n_tweets:', final_n_tweets

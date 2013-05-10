__author__ = 'behzadbehzadan'

import os
import random
import funcs_worry
import my_util
import copy

home_dir = os.path.expanduser('~')
source_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/source/'
save_dir = '/Chatterbox_UCL_Advance/worry_brit_gas_exp/exp_hashtags/'
########################################################################################################################
test_on = 'hand_picked_data'
test_on = 'mech_turk'
remove_retweets = False
remove_stpwds_for_unigrams = False
new_normalisation_flag = False
random.seed(7)
# positive labels are associated to worried/concerned/stressed... tweets.
# negative labels are associated to NOT worried/concerned/stressed... tweets.
# other labels are associated to any other types of tweets.
class_labels = { 'pos' : +1, 'neg' : -1}#, 'oth' : 0}
#m=1: starts from unigram; m=2: starts from bigram; m=3: starts from trigram
min_ngram = 1
#length of ngram --> n=1: unigram; n=2: bigram; n=3: trigram
max_ngram = 3
###################################################### libsvm settings #################################################
# The nu_CSV does not take the C parameter (i.e. the cost function). Hence, there is no weight balancing option.
svm_type = 'C_SVC'#'nu_SVC'#
# Set the kernel. linear --> 0; polynomial --> 1; radial basis --> 2; sigmoid --> 3; precomputed kernel --> 4
kernel_type = 0
# Set the cost parameter for the C_CSV
cost = 10
# Set the nu parameter for the nu_SVC
# Note: if nu is not small enough, the following error message is shown: "specified nu is infeasible"
nu = 0.05
n_fold_cross_val = 10
# Assign different costs to balance unbalanced (different sized) training sets.
balance_sets = True
########################################################################################################################

def read_hash_tweets_source_data():
    if remove_retweets:
        data_source = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags', False, True)
        header = data_source[0]
        tweet_hashtags = [row[header.index('text')] for row in data_source]
        tweet_hashtags = [t.lower() for t in tweet_hashtags]
        tweet_hashtags = [' '.join(t.split()) for t in tweet_hashtags]
        tweet_hashtags_noDup = funcs_worry.remove_duplicate_tweets(tweet_hashtags, False, None)
        my_util.write_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True,
                               [[t] for t in tweet_hashtags_noDup])
    else:
        tweet_hashtags_noDup = my_util.read_csv_file(home_dir + source_dir + 'source_worry_hashtags_noDup', False, True)
        tweet_hashtags_noDup = [t[0] for t in tweet_hashtags_noDup]

    return tweet_hashtags_noDup

def read_amazon_mech_turk_data():

    print 'reading Amazon MTurk source files ...'
    tweets_MTurk = my_util.read_csv_file(home_dir + source_dir + 'AmazonMTurk', False, True)
    header = tweets_MTurk[0]
    worrieds_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'worry']
    not_worrieds_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'notworry']
    tell_not_worry_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'tellnotworry']
    nothing_MTurk = [t[header.index('Input.TEXT')] for t in tweets_MTurk if t[header.index('Answer.Worry')] == 'nothing']
    # select those tweets that turkers agreed when annotating --> thresshol==2 means at least two annotators agreed.
    worrieds_MTurk = funcs_worry.filter_same_turked_tweets(worrieds_MTurk, 2)
    not_worrieds_MTurk = funcs_worry.filter_same_turked_tweets(not_worrieds_MTurk, 2)
    tell_not_worry_MTurk = funcs_worry.filter_same_turked_tweets(tell_not_worry_MTurk, 2)
    nothing_MTurk = funcs_worry.filter_same_turked_tweets(nothing_MTurk, 2)
    print 'Amazon MTurk data size:', len(worrieds_MTurk) + len(not_worrieds_MTurk) + len(tell_not_worry_MTurk) + len(nothing_MTurk)

    return worrieds_MTurk, not_worrieds_MTurk, tell_not_worry_MTurk, nothing_MTurk

def read_hand_picked_data():

    print 'reading test set source files ...'
    tweets_test_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_worry', False, True)
    tweets_test_worry = [t[0] for t in tweets_test_worry]
    tweets_test_worry = [t.lower() for t in tweets_test_worry]
    tweets_test_worry = [' '.join(t.split()) for t in tweets_test_worry]
    print 'test_set_worry size:', len(tweets_test_worry)

    tweets_test_not_worry = my_util.read_csv_file(home_dir + source_dir + 'test_set_not_worry', False, True)
    tweets_test_not_worry = [t[0] for t in tweets_test_not_worry]
    tweets_test_not_worry = [t.lower() for t in tweets_test_not_worry]
    tweets_test_not_worry = [' '.join(t.split()) for t in tweets_test_not_worry]
    print 'test_set_not_worry size:', len(tweets_test_not_worry)

    tweets_test_others = my_util.read_csv_file(home_dir + source_dir + 'test_set_others', False, True)
    tweets_test_others = [t[0] for t in tweets_test_others]
    tweets_test_others = [t.lower() for t in tweets_test_others]
    tweets_test_others = [' '.join(t.split()) for t in tweets_test_others]
    print 'test_set_others size:', len(tweets_test_others)

    return tweets_test_worry, tweets_test_not_worry, tweets_test_others

def find_tweets_with_keyword(tweets, keyword):
    tweets_with_the_keyword = []
    for tweet in tweets:
        if keyword in tweet:
            tweets_with_the_keyword.append(tweet)

    return tweets_with_the_keyword

def find_tweets_with_exact_keyword(tweets, keyword):
    tweets_with_the_keyword = []
    for tweet in tweets:
        if keyword in tweet.split():
            tweets_with_the_keyword.append(tweet)

    return tweets_with_the_keyword

def remove_intersections(tweets_with_keywords_pos, tweets_with_keywords_neg):

    intersection_file = []
    comparison_nr = 0
    tweets_with_keywords = copy.deepcopy(tweets_with_keywords_pos)
    tweets_with_keywords.update(tweets_with_keywords_neg)

    traversed_keywords = []
    for keyword, tweets in tweets_with_keywords.items():

        traversed_keywords.append(keyword)
        no_intersect = tweets[:]
        for other_keyword, other_tweets in tweets_with_keywords.items():

            if other_keyword not in traversed_keywords:

                comparison_nr+=1
                print keyword + ' and ' + other_keyword + ' ...'
                other_tweets = tweets_with_keywords[other_keyword]
                no_intersect = funcs_worry.remove_intersection_from_the_list(no_intersect, other_tweets)
                # # this part is to visualise what is the original intersection between two lists
                # real_no_intersect = funcs_worry.remove_intersection_from_the_list(tweets, other_tweets)
                # intersection_file.append([keyword, other_keyword, len(tweets)-len(real_no_intersect)])

        if keyword in tweets_with_keywords_pos:
            tweets_with_keywords_pos[keyword] = no_intersect
        if keyword in tweets_with_keywords_neg:
            tweets_with_keywords_neg[keyword] = no_intersect
    print 'number of comparisons to remove intersections:', comparison_nr
    my_util.write_csv_file(home_dir + source_dir + 'hash_tweets_intersect_size', False, True, intersection_file)

keywords_pos = ['#worried', '#anxious']#, 'worry', 'help', 'eek', 'anxious']
keywords_neg = ['#easy', '#relaxed', '#calm']

###################################################### read source data ################################################
hash_tweets = read_hash_tweets_source_data()
worrieds_mech_turk, not_worrieds_mech_turk, tell_not_worry_mech_turk, nothing_MTurk = read_amazon_mech_turk_data()
worried_hand_picked, not_worried_hand_picked, nothing_hand_picked = read_hand_picked_data()
########################################################################################################################

tweets_with_hash_keywords_pos = {}
tweets_with_hash_keywords_neg = {}

for keyword in keywords_pos:
    if '#' in keyword:
        tweets_with_keyword = find_tweets_with_keyword(hash_tweets, keyword)
    # if '#' not in keyword:
    #     tweets_with_keyword = find_tweets_with_exact_keyword(hash_tweets, keyword)
    tweets_with_hash_keywords_pos[keyword] = tweets_with_keyword
    print 'number of tweets containing '+ keyword + ' :', len(tweets_with_keyword)

for keyword in keywords_neg:
    if '#' in keyword:
        tweets_with_keyword = find_tweets_with_keyword(hash_tweets, keyword)
    # if '#' not in keyword:
    #     tweets_with_keyword = find_tweets_with_exact_keyword(hash_tweets, keyword)
    tweets_with_hash_keywords_neg[keyword] = tweets_with_keyword
    print 'number of tweets containing ' + keyword + ' :', len(tweets_with_keyword)

remove_intersections(tweets_with_hash_keywords_pos, tweets_with_hash_keywords_neg)

all_hash_tweets_pos = []
for keyword, tweets in tweets_with_hash_keywords_pos.items():
    all_hash_tweets_pos.extend(tweets)

hash_tweets_train_labs_pos = tweets_with_hash_keywords_pos.keys()

all_hash_tweets_neg = []
for keyword, tweets in tweets_with_hash_keywords_pos.items():
    all_hash_tweets_pos.extend(tweets)

hash_tweets_train_labs_neg = tweets_with_hash_keywords_neg.keys()

hand_picked_pos = worried_hand_picked
hand_picked_neg = not_worried_hand_picked, nothing_hand_picked

mech_turk_pos = worrieds_mech_turk + tell_not_worry_mech_turk
mech_turk_neg = not_worrieds_mech_turk + nothing_MTurk

print 'creating feature vectors...'

features_dict = {}
features_count_dict = {}
#the very first index is always 1.
if new_normalisation_flag:
    max_index = 0
else:
    max_index = 1

# since there is no intersection between tweets containing keywords we can send the aggregated tweets int o the function below:
hash_tweets_feature_vects_pos, hash_tweets_texts_pos, max_index, hash_tweets_norm_factors_pos = \
    funcs_worry.get_sparse_feature_vector_worry(
    all_hash_tweets_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
    remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_pos, random)

hash_tweets_feature_vects_neg, hash_tweets_texts_neg, max_index, hash_tweets_norm_factors_neg = \
    funcs_worry.get_sparse_feature_vector_worry(
    all_hash_tweets_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
    remove_stpwds_for_unigrams, new_normalisation_flag, hash_tweets_train_labs_neg, random)


funcs_worry.write_features_count_dict_to_csv(home_dir + save_dir + 'features_count_dict_training')
funcs_worry.write_features_and_freqs_to_csv(features_dict, features_count_dict, -1,
                                            home_dir + save_dir + 'features_freq_training')

if test_on == 'hand_picked_data':

    hand_picked_feature_vects_pos, hand_picked_texts_pos, max_index, hand_picked_norm_factors_pos = \
        funcs_worry.get_sparse_feature_vector_worry(
            hand_picked_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

    hand_picked_feature_vects_neg, hand_picked_texts_neg, max_index, hand_picked_norm_factors_neg = \
        funcs_worry.get_sparse_feature_vector_worry(
            hand_picked_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)


            
if test_on == 'mech_turk':

    mech_turk_feature_vects_pos, mech_turk_texts_pos, max_index, mech_turk_norm_factors_pos = \
        funcs_worry.get_sparse_feature_vector_worry(
            mech_turk_pos, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

    mech_turk_feature_vects_neg, mech_turk_texts_neg, max_index, mech_turk_norm_factors_neg = \
        funcs_worry.get_sparse_feature_vector_worry(
            mech_turk_neg, features_dict, features_count_dict, max_index, min_ngram, max_ngram,
            remove_stpwds_for_unigrams, new_normalisation_flag, [], random)

funcs_worry.write_features_dict_to_csv(features_dict, home_dir + save_dir + 'features_dict')
__author__ = 'behzadbehzadan'

"""
This file gets two lists of tokens l1 and l2:
l1=[t1, t2, t3, ..., tn], l2=[s1, s2, s3, ..., sm]
and provide a measurement implying how similar/dissimilar the two lists are.
The algorithm first measures the percentage of the identical/not-identical tokens shared in the two lists,
and then within the identical/non-identical tokens measures how similar/dissimilar are those tokens based on
word by word comparison.
"""

import difflib

def compare_tokens(l1, l2):

    """
    The topic.py file creates a list of tokens t, and w is the list of associated weight (probabilities), i.e.:
    t=[(t1,w1), (t2,w2), (t3,w3), ..., (tn,wn)]. First we separate the tokens in one list [t1, t2, t3, ..., tn].
    """

    toks1 = [t for t,w in l1]
    toks2 = [t for t,w in l2]

    """
    The topic.py does not provide duplicate tokens, so creating a set of tokens have no effect, but facilitates working
    with elements.
    """
    toks1 = set(toks1)
    toks2 = set(toks2)


    diff12 = toks1-toks2
    diff21 = toks2-toks1
    symetric_diff = diff12 | diff21 # | this is a set union operand. Note there is no duplicate elements.

    """
    calcualte what percentage of tokens are different in the two sets! The higher the value, the more dissimilar the tow
    sets of tokens.
    """
    dissimilarity_percent = float(len(diff12)+len(diff21)) / (len(toks1)+len(toks2))

    """
    For each token in the symetric difference we split the token to sepearte words. We furthermore put the splitted words
    into a set to get rid of repeated words. For instance t1='i love, love, love it' and t2='i love it' would then be 100%
    similar!. quick_ratio() will look at the ratio of the identical words in the splited lists and provides a value. The
    higher the value, the more similar the tow sets of tokens,; hence, we subtract the result from 1 to get the dissimilarity.
    """
    dissimilarity_q_ratio = [(1-(difflib.SequenceMatcher(None,t1.split(), t2.split()).quick_ratio()))
                            for t1 in diff12 for t2 in diff21]

    if len(dissimilarity_q_ratio)==0: # this happens when one of the diff12 or diff21 is empty, which happens when one of the toks1 or toks 2 is a subset of the other.
        dissimilarity_total=dissimilarity_percent
    else:
        dissimilarity_q_ratio_avg = float(sum(dissimilarity_q_ratio))/len(dissimilarity_q_ratio)
        dissimilarity_total = dissimilarity_percent*dissimilarity_q_ratio_avg

    return dissimilarity_total, list(diff12), list(diff21)


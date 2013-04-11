__author__ = 'behzadbehzadan'
import math

def calc_mean_stdev(X):
    #this function calculates the mean and stdev of a list of numbers stored in X.
    mean = sum([x for x in X]) / float(len(X))
    stdev = math.sqrt(sum([((x-mean)**2) / float(len(X)) for x in X]))
    return mean, stdev
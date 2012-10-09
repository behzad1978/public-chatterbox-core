# -*- coding: utf-8 -*-

"""
    Module for performing Chatterbox Sentiment Analysis
    Copyright Chatterbox Analytics Ltd 2012
    
    Requires liblinear python bindings
    
    TODO: Still requires django
"""
import liblinearutil as linu
import re
import codecs
import os
import unicodecsv
import functools

from django.utils.encoding import smart_unicode

class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)

def get_sparse_feature_vector(texts,features_dict,exclude):
    """
        This provides a bigram term-frequency vector
    """
    vector = dict()
    total_toks = 0
    for text in texts:
        working_text = text.lower()
        working_text = re.sub(r"([^'\".,;:/?\!@#£$%^&*()_\-=+`~])(['\".,;:/?\!@#£$%^&*()_\-=+`~])", r"\1 \2",working_text)
        working_text = re.sub(r"(['\".,;:/?\!£$%^&*()_\-=+`~])([^'\".,;:/?\!@#£$%^&*()_\-=+`~#@])", r"\1 \2",working_text)
        working_text = re.sub(" < 3 ","<3",working_text)
        working_text = re.sub(r"(\w)n ' t\b","\tn't",working_text)
        working_text = re.sub(r" ([:;][-]?) ([DP]) ",r"\1\2",working_text)
        working_text = re.sub(r"(\S)\1\1+",r"\1\1\1",working_text)
        tokens = re.split(r"\s+",working_text)
        
        tokens = [t.strip() for t in tokens if t]

        total_toks = total_toks + len(tokens)
        
        max = 3
        for i in range(1,max):
            for j in xrange(0,len(tokens)-(i-1)):
                t = " ".join(tokens[j:j+i])
                try:
                    indx = features_dict[t]
                except KeyError:
                    pass
                else:
                    try:
                        vector[indx] = vector[indx] + (1.0 / total_toks)
                    except KeyError:
                        vector[indx] = (1.0 / total_toks) 
    return vector

def check_feature(feature, exclude):
    """
        Determines if a feature should be used.
        Returns True or False
    """
    for e in exclude:
        if e in feature:
            return False
    return True

@memoized
def load_features(lang=None):
    """
        Loads the labels sorted in labels.txt from training.
        Returns a dictionary mapping lexical tokens to integer indices
        Features are stored in the resources directory
    """
    if lang:
        print "Using", lang, "labels.."
        f = unicodecsv.reader(open(os.path.join(
                        os.path.abspath(os.path.dirname(__file__)),
                        'resources',
                        lang +"_labelsu.txt"),'rb'), encoding="utf-8", delimiter="\t")
    
    features_dict = dict()
    for line in f:
        line_spl = line
        if len(line_spl) > 1:
            val = int(line_spl[1])
            features_dict[smart_unicode(line_spl[0])] = val
    return features_dict

@memoized
def load_model(lang):
    """
        Loads the model from file and returns it.
        Models are stored in the resources directory
    """
    print "Using", lang, "linear model.."
    if lang:
        return linu.load_model(os.path.join(
                                os.path.abspath(os.path.dirname(__file__)),
                                'resources',
                                lang + '_sentiment.model'))
    else:
        raise Exception

def classify_text(text, lang, exclude=[]):
    """ 
        makes the text into a feature vector, then classifies it. 
        exclude should be a list of strings to exclude from the vectors
        
        exclude excludes any phrase that contains or is equal to any of the
        string in the exclude list using a case insensitive comparison
        
        TODO: What if an unsupported language is requested?  The files won't be on disk
    """
    
    model = load_model(lang)
    features = load_features(lang)
    
    texts = [text.lower()]
    for e in exclude:#this for loop is not right
        new_texts = []
        for t in texts:
            new_texts = new_texts + t.split(e)
        texts = new_texts
    feature_vector = get_sparse_feature_vector(texts, features, exclude)
    p_label, p_acc, p_val = linu.predict([0], [feature_vector], model)
    p_val = p_val[0][0]/(1+abs(p_val[0][0]))
    return {'label':p_label[0],'value':p_val}
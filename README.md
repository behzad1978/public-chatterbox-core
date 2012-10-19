# Chatterbox #

## Chatterbox Core ##

Exposes the following http API:

`GET /sentiment`  
Get sentiment for a piece of text  
params:  
`text`: the text to get the sentiment for  
`lang`: the language of the text  
OR  
`POST /sentiment`  
same as above but with an additional POST payload  
POST data:  
`exclude`: list of words to exclude from sentiment classification

`GET /topics/ngrams`  
Get ngrams for a piece of text  
params:  
`text`: the text to get the ngrams for  
`lang`: the language of the text  

`POST /topics/ranked`  
Needs description  
params:  
`lang`: the language of the text  
POST data:  
`test_ngrams`: ngrams to weight  
`offset_ngrams`: ngrams to weight against  

## NOTE: liblinear.so.1 needs to be compiled for the right platform! ##

## sentiment.py ##
	>>> import core.sentiment
	>>> core.sentiment.classify_text(text="i hate python, it sucks",lang="en")
	Using en linear model..
	Using en labels..
	Accuracy = 0% (0/1) (classification)
	{'value': -0.7631638282398892, 'label': -1.0}
	>>> core.sentiment.classify_text(text="i totally love the world",lang="en")
	Accuracy = 0% (0/1) (classification)
	{'value': 0.6226175130859044, 'label': 1.0}
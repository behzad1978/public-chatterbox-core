Chatterbox
==========

Chatterbox main codebase


sentiment.py
============
>>> import core.sentiment
>>> core.sentiment.classify_text(text="i hate python, it sucks",lang="en")
Using en linear model..
Using en labels..
Accuracy = 0% (0/1) (classification)
{'value': -0.7631638282398892, 'label': -1.0}
>>> core.sentiment.classify_text(text="i totally love the world",lang="en")
Accuracy = 0% (0/1) (classification)
{'value': 0.6226175130859044, 'label': 1.0}
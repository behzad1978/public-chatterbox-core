__author__ = 'behzadbehzadan'

# 'worried' vs itself

collection_names = ['worried']
# class_labels:
pos = 1
neg = -1

for collection_name in collection_names:
    if collection_name == 'worried':
        labels['pos'] = {collection_name: eval('pos')}
        labels['neg'] = {collection_name: eval('neg')}
    elif collection_name == 'concerned':
        labels['pos'] = {collection_name: None}
        labels['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        labels['pos'] = {collection_name: None}
        labels['neg'] = {collection_name: None}

for collection_name in collection_names:
    if collection_name == 'worried':
        traiining_flag['pos'] = {collection_name: True}
        traiining_flag['neg'] = {collection_name: True}
    elif collection_name == 'concerned':
        traiining_flag['pos'] = {collection_name: True}
        traiining_flag['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        traiining_flag['pos'] = {collection_name: False}
        traiining_flag['neg'] = {collection_name: False}

for collection_name in collection_names:
    if collection_name == 'worried':
        test_flag['pos'] = {collection_name: True}
        test_flag['neg'] = {collection_name: True}
    elif collection_name == 'concerned':
        test_flag['pos'] = {collection_name: True}
        test_flag['neg'] = {collection_name: None}
    elif collection_name == 'relaxed':
        test_flag['pos'] = {collection_name: False}
        test_flag['neg'] = {collection_name: False}

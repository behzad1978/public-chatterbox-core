from flask import Flask, request, url_for

import sentiment
import json

app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/sentiment')
def sentiment():
    if 'text' in request.args and 'lang' in request.args:
        text = request.args['text']
        lang = request.args['lang']
    return json.dumps(sentiment.classify_text(text=text, lang=lang))

@app.route('/topics/ngrams')
def ngrams():
    if 'text' in request.args and 'lang' in request.args:
            text = request.args['text']
            lang = request.args['lang']
    return json.dumps(sentiment.classify_text(text=text, lang='en'))

if __name__ == '__main__':
    app.run()
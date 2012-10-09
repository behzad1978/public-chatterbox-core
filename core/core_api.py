from flask import Flask, request, url_for, jsonify

from sentiment import classify_text
from topic import ngrams
import json

app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/sentiment')
def api_sentiment():
    if 'text' in request.args and 'lang' in request.args:
        text = request.args['text']
        lang = request.args['lang']
        res = classify_text(text=text, lang=lang)
        return jsonify(**res)
    #todo: return error

@app.route('/topics/ngrams')
def api_ngrams():
    if 'text' in request.args and 'lang' in request.args:
        text = request.args['text']
        lang = request.args['lang']
        res = ngrams(text.split(), lang)
        return json.dumps(res)
    #todo: return error

if __name__ == '__main__':
    app.run(debug=True)
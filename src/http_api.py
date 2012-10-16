from flask import Flask, request, url_for, jsonify
from werkzeug.exceptions import HTTPException, default_exceptions

from sentiment import classify_text
from topic import ngrams, get_offset_ranked_topics

# CONFIG

DEBUG = True

def make_json_app(import_name, **kwargs):
    """
    Creates a JSON-oriented Flask app.

    All error responses that you don't specifically
    manage yourself will have application/json content
    type, and will contain JSON like this (just an example):

    { "message": "405: Method Not Allowed" }
    """

    def make_json_error(ex):
        response = jsonify(message=str(ex))
        response.status_code = (ex.code
                                if isinstance(ex, HTTPException)
                                else 500)
        return response

    app = Flask(import_name, **kwargs)
    app.config.from_object(import_name)

    for code in default_exceptions.iterkeys():
        app.error_handler_spec[None][code] = make_json_error

    return app


app = make_json_app(__name__)
# from_object looks for all uppercase variables defined in this file

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/sentiment/', methods=['GET','POST'])
@app.route('/sentiment', methods=['GET','POST'])
def api_sentiment():
    text = request.args['text']
    lang = request.args['lang']
    exclude = []
    if request.method == 'POST':
        exclude = request.form['exclude']
    res = classify_text(text=text, lang=lang, exclude=exclude)
    return jsonify(**res)


@app.route('/topics/ngrams/')
def api_ngrams():
    text = request.args['text']
    lang = request.args['lang']
    res = ngrams(text.split(), lang)
    return jsonify(ngrams = res)

@app.route('/topics/ranked/', methods=['POST'])
@app.route('/topics/ranked', methods=['POST'])
def api_ranked_topics():
    lang = request.args['lang']
    test_ngrams = request.form.getlist('test_ngrams')
    offset_ngrams = request.form.getlist('offset_ngrams')
    res = get_offset_ranked_topics(test_ngrams, offset_ngrams, [], lang)
    return jsonify(**res)


if __name__ == '__main__':
    app.run()
from flask import Flask, request, url_for, jsonify
from werkzeug.exceptions import HTTPException, default_exceptions

from sentiment import classify_text
from topic import ngrams
import json

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

    for code in default_exceptions.iterkeys():
        app.error_handler_spec[None][code] = make_json_error

    return app


app = make_json_app(__name__)

@app.route('/')
def api_root():
    return 'Welcome'


@app.route('/sentiment')
def api_sentiment():
    text = request.args['text']
    lang = request.args['lang']
    res = classify_text(text=text, lang=lang)
    return jsonify(**res)


@app.route('/topics/ngrams')
def api_ngrams():
    text = request.args['text']
    lang = request.args['lang']
    res = ngrams(text.split(), lang)
    return json.dumps(res)


if __name__ == '__main__':
    app.run(debug=True)
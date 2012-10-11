import unittest
import http_api
import json
import topic
# test responses

bad_request_error_response = {
    'message': "400: Bad Request"
}
sentiment_ok = {
  "value": -0.6640092855131534,
  "label": -1.0
}

ngrams_ok = {
  "ngrams": [
    "apples",
    "I hate apples"
  ]
}

rank_ok = {
    'pos': [
            ('dog', 0.7142857142857143), 
            ('i like dog on toast in the sunny morning dog dog dog', 0.3333333333333333)
    ]
}

class HttpApiTestCase(unittest.TestCase):
    def setUp(self):
        http_api.app.config['TESTING'] = True
        self.app = http_api.app.test_client()
        pass

    def tearDown(self):
        pass

    # sentiment tests

    def test_sentiment_bad_request(self):
        resp = self.app.get('/sentiment')
        assert json.loads(resp.data) == bad_request_error_response

        resp = self.app.get('/sentiment?text=abc')
        assert json.loads(resp.data) == bad_request_error_response

        resp = self.app.get('/sentiment?lang=abc')
        assert json.loads(resp.data) == bad_request_error_response

    def test_sentiment(self):
        resp = self.app.get('/sentiment?text=I%20hate%20apples&lang=en')
        assert json.loads(resp.data) == sentiment_ok

    # ngrams tests

    def test_ngrams_bad_request(self):
        resp = self.app.get('/topics/ngrams')
        assert json.loads(resp.data) == bad_request_error_response

        resp = self.app.get('/topics/ngrams?text=abc')
        assert json.loads(resp.data) == bad_request_error_response

        resp = self.app.get('/topics/ngrams?lang=abc')
        assert json.loads(resp.data) == bad_request_error_response

    def test_ngrams(self):
        resp = self.app.get('/topics/ngrams?text=I%20hate%20apples&lang=en')
        assert json.loads(resp.data) == ngrams_ok

    # rank tests

    def test_rank_bad_request(self):
        resp = self.app.post('/topics/ranked')
        assert json.loads(resp.data) == bad_request_error_response

    def test_rank(self):
        pos = [topic.ngrams("i like cat on toast in the sunny morning".split(),'en')]
        neg = [topic.ngrams("i like dog on toast in the sunny morning dog dog dog".split(),'en')] + [topic.ngrams("i like dog on toast in the sunny morning".split(),'en')]
        res = self.app.post('/topics/ranked?lang=en', data = dict(
            pos_ngrams = pos,
            neg_ngrams = neg
        ))
        assert json.loads(res.data) == rank_ok

if __name__ == '__main__':
    unittest.main()
import unittest
import http_api
import json
import topic
# test responses

# bad request response we're expecting
bad_request_error_response = {
    'message': "400: Bad Request"
}

# success response we're expecting for sentiment test case
sentiment_ok = {
    "value": -0.6640092855131534,
    "label": -1.0
}

# success response we're expecting for ngrams test case
ngrams_ok = {
    "ngrams": [
        "apples",
        "I hate apples"
    ]
}

# success response we're expecting for rank test case
rank_ok = {
    'ranked_testngrams': [
        ['i like cat on toast in the sunny morning', 0.3333333333333333]
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
        response = json.loads(resp.data)
        # assert that keys are in there
        assert 'value' in response and 'label' in response
        value = response['value']
        label = response['label']
        # assert key vals are floats
        assert isinstance(value, float) and isinstance(label, float)
        # assert that label is 1 or -1
        assert abs(label) == 1
        # assert that label and value signs are consistent with each other
        assert (label < 0 and value < 0) or (label > 0 and value > 0)
        # assert that value is [-1, 1]
        assert value >= -1 and value <= 1

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
        a = topic.ngrams("i like cat on toast in the sunny morning".split(), 'en')
        b = topic.ngrams("i like dog on toast in the sunny morning dog dog dog".split(), 'en') + topic.ngrams(
            "i like dog on toast in the sunny morning".split(), 'en')
        data = dict(
            test_ngrams=a,
            offset_ngrams=b
        )
        res = self.app.post('/topics/ranked?lang=en', data=data)
        assert json.loads(res.data) == rank_ok

if __name__ == '__main__':
    unittest.main()
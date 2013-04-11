def get_tweets(db_name, collection_name, mongo_server_ip = '192.168.1.10'):

    """
        Returns a list of strings representing the tweet text.

    """

    import pymongo

    mongo_connection = pymongo.Connection(mongo_server_ip)

    mongo_db = mongo_connection[db_name]

    mongo_collection = mongo_db[collection_name]

    mongo_collection.find({}, {"text" : 1} )

    tweet_list = [[t['text'],t['sentiment_value']] for t in tweets]
    #tweets_text = [t['text'] for t in tweets if cld.detect(t['text'].encode('utf-8'))[1] == 'en']

    return tweet_list

get_tweets("behzad", "worry")
# import spacy
# import twint
import tweepy
import json
import config
import datetime
import re
import logging
from sys import argv
import re
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import urllib
import requests

logging.basicConfig(filename='errors.log', filemode='a+', format='%(asctime)s: %(message)s', level=logging.ERROR)

auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# nlp = spacy.load("en_core_web_sm")

tweets = {}
users = {}

mask = np.array(Image.open(requests.get('data:image/png;base64,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', stream=True).raw))


def process_tweet(tweet):
    """ Takes in a string, returns a list words in the string that aren't stopwords
    Parameters:
        tweet (string):  string of text to be tokenized
    Returns:
        stopwords_removed (list): list of all words in tweet, not including stopwords
    """
    stopwords_list=stopwords.words('english') +list(string.punctuation)
    stopwords_list += ["'",'"','...','``','…','’','‘','“',"''",'""','”','”','co',"'s'",'\'s','n\'t','\'m','\'re','amp','https']
    tokens = nltk.word_tokenize(tweet)
    stopwords_removed = [token.lower() for token in tokens if token not in stopwords_list]
    return stopwords_removed


def tokenized(series):
    """ Takes in a series containing strings or lists of strings, and creates a single list of all the words
    Parameters:
        series (series): series of text in the form of strings or lists of string

    Returns:
        tokens (list): list of every word in the series, not including stopwords
    """

    corpus = ' '.join([tweet.lower() if type(tweet)==str else ' '.join([tag.lower() for tag in tweet]) for tweet in series])
    tokens = process_tweet(corpus)
    return tokens


def wordfrequency(series, top):
    """ Returns the frequency of words in a list of strings.
    Parameters:
        series (iterable): List of strings to be combined and analyzed
        top (int): The number of top words to return.
    Returns:
        list (tuples): List of word and value pairs for the top words in the series.
    """
    frequencies = FreqDist(tokenized(series))
    return frequencies.most_common(top)




def create_wordcloud(series, tag, *top):
    """ Take in a list of lists and create a WordCloud visualization for those terms.
    Parameters:
            series (iterable): A list of lists containing strings.
    Returns:
        None: The ouput is a visualization of the strings in series in terms of the
            frequency of their occurrence.
    """

    vocab = tokenized(series)
    if not top[0]:
        top[0]=200
    cloud=WordCloud(max_words=top[0]).generate(' '.join([word for word in vocab]))
    plt.imshow(cloud,interpolation='bilinear')
    plt.title(f'Most Common words for {tag}')
    plt.plot(figsize = (48,24))
    plt.axis('off')
    plt.show();

def strip_tweets(tweet):
    '''Process tweet text to remove retweets, mentions,links and hashtags.'''
    retweet = r'RT:? ?@\w+:?'
    tweet = re.sub(retweet, '', tweet)
    mention = r'@\w+'
    tweet = re.sub(mention, '', tweet)
    links = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
    tweet = re.sub(links, '', tweet)
    tweet_links = r'https:\/\/t\.co\/\w+|http:\/\/t\.co\/\w+'
    tweet = re.sub(tweet_links, '', tweet)
    tweet_link = r'http\S+'
    tweet = re.sub(tweet_link, '', tweet)
    hashtag = r'#\w+'
    hashtags = re.findall(hashtag, tweet)
    tweet = re.sub(hashtag, '', tweet)
    return tweet, hashtags

def cluster_flocks(dicts):
    hashtags = Counter()
    tweets = []
    for dic in dicts.values():
        text, tag = strip_tweets(dic['text'])
        hashtags.update(tag)
        tweets.append([text, tag])
    return tweets, hashtags


def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def listen(terms, amount):
    for term in terms:
        for tweet in tweepy.Cursor(api.search, q=term, count=amount, tweet_mode ='extended').items(amount):
            if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
                try:
                    tweet_ = dict()
                    tweet_['created_at'] = tweet.created_at
                    tweet_['text'] = tweet.full_text
                    if tweet.lang:
                        tweet_['lang'] = tweet.lang
                    if tweet.in_reply_to_status_id:
                        tweet_['in_reply_to_status_id'] = tweet.in_reply_to_status_id
                    if tweet.in_reply_to_user_id:
                        tweet_['in_reply_to_user_id'] = tweet.in_reply_to_user_id
                    if tweet.retweet_count:
                        tweet_['retweet_count'] = tweet.retweet_count
                    else:
                        tweet_['retweet_count'] = 0
                    if tweet.favorite_count:
                        tweet_['favorite_count'] = tweet.favorite_count
                    else:
                        tweet_['favorite_count'] = 0
                    tweet_['user_id'] = tweet.user.id
                    tweet_['coordinates'] = tweet.coordinates
                    hash_tag = re.search(r'\#\w*',tweet.full_text)
                    if hash_tag:
                        if isinstance(hash_tag,list):
                            # for tag in hash_tag: hash_tags.add(tag)
                            tweet_['hashtags']= hash_tag
                        else:
                            # hash_tags.add(hash_tag)
                            tweet_['hashtags']= [hash_tag]
                    tweets[int(tweet.id)] = tweet_
                except Exception as e:
                    print(e)
                    print(tweet)
                    logging.error(f'Error: {e}\nFailed term: {term}Failed tweet: {tweet}')
                    continue

                try:
                    user_ = dict()
                    user_['screen_name'] = tweet.user.screen_name
                    user_['followers_count'] = tweet.user.followers_count
                    user_['verified'] = tweet.user.verified
                    user_['created_at'] = tweet.user.created_at
                    if tweet.user.lang:
                        user_['lang'] = tweet.user.lang
                    users[int(tweet.user.id)] = user_
                except Exception as e:
                    print(e)
                    print(tweet)
                    logging.error(f'Error: {e}\nFailed term: {term} Failed user: {user}\n')
                    continue
        print(f'Done with {term}. Currently {len(tweets)} collected from {len(users)} users.')

    print(f'{len(tweets)} tweets and {len(users)} users.')
    with open('tweets.json', 'a+') as t:
        json.dump(tweets, t, default=myconverter)
    with open('users.json', 'a+') as u:
        json.dump(users, u, default=myconverter)

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        json.dump(status.text)

if __name__ == "__main__":
    topics = argv[1:]
    print(topics)
    listen([str(topic) for topic in topics], 1000)
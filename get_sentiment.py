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
from nltk.stem import WordNetLemmatizer
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
nltk.download('wordnet')

tweets = {}
users = {}

mask = np.array(Image.open(requests.get('https://www.laserlogik.com/wp-content/uploads/2018/03/map-RED-Iowa-1.png', stream=True).raw))
lemmatizer = WordNetLemmatizer()


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
    stopwords_removed = [lemmatizer.lemmatize(token).lower() for token in tokens if token not in stopwords_list]
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
    cloud=WordCloud(background_color='white', max_words=top[0], mask=mask).generate(' '.join([word for word in vocab]))
    plt.imshow(cloud,interpolation='bilinear')
    plt.title(f'Most Common words for {tag}')
    plt.plot(figsize = (48,24))
    plt.axis('off')
    plt.tight_layout(pad=0)
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
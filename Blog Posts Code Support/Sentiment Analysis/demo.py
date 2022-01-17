# Default packages
import numpy as np
import pandas as pd
import os
import re
import time

# Text treatment
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Twitter
import tweepy


# Wordcloud
from wordcloud import WordCloud,  ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Visualisation
import plotly.io as pio
pio.renderers.default='notebook'
import plotly.express as px




"""
Twitter Authentication
"""
# Configuration to be able to use Tweeter API
consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'ACCESS_TOKEN_SECRET'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

variables_we_need = ['created_at', 'id', 'full_text', 'entities', 'coordinates', 'retweet_count', 'favorite_count', 'lang']


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_analysis = pipeline("sentiment-analysis", model = model, tokenizer=tokenizer)
possible_sentiments = ['negative', 'neutral', 'positive']


def get_all_tweets(count=100, q='', lang='', since='', until='', tweet_mode='extended'):
    results = []

    tweets = tweepy.Cursor(api.search, q=q, lang=lang, since=since, until=until, tweet_mode=tweet_mode).items(count)

    for tweet in tweets:
        d = {}
        for variable in variables_we_need:
            d[variable] = tweet._json[variable]
        results.append(d)

    print(f"Number of extracted tweets for keyword {q} => {str(len(results))}")
    return results

def clean_text(all_tweets):

    for tweet_ in all_tweets:
        text = tweet_['full_text']
        all_mentions = [j['screen_name'] for j in i['entities']['user_mentions']]

        # If retweet, delete RT and name of the account
        text = re.sub('(RT\s.*):', '', text)

        # Find all links and delete them
        all_links = re.findall('(https:.*?)\s', text + ' ')

        for i in all_links:
            text = text.replace(i, '')

        for i in all_mentions:
            text = text.replace('@' + i, '')

        # Tokenize
        tokens = word_tokenize(text.replace('-', ' '))

        # Convert to lower case
        tokens = [w.lower() for w in tokens]

        # Remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # Remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # Filter out stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        phrase = " ".join(words)

        tweet_['clean_text'] = phrase
        tweet_['all_link'] = all_links

    return all_tweets





def get_sentiments(input_dict, variable_text):
    for item_ in input_dict:
        sentiment = sentiment_analysis(item_[variable_text])
        for item in sentiment:
            for shade in possible_sentiments:
                if item['label'] == shade:
                    item_[shade] = item['score']
                else:
                    item_[shade] = 0

    return input_dict



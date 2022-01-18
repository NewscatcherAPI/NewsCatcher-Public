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

# News
from newscatcherapi import NewsCatcherApiClient


# Wordcloud
from wordcloud import WordCloud,  ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import animation


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


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_analysis = pipeline("sentiment-analysis", model = model, tokenizer=tokenizer)
possible_sentiments = ['negative', 'neutral', 'positive']

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


newscatcherapi = NewsCatcherApiClient(x_api_key='YOUR-X-API-KEY')

def get_news(company):
    query = f'{company} AND compnay OR {company} Inc'
    articles = []
    for i in range(1, 11):
        articles.extend(newscatcherapi.get_search(q=query,
                                            lang='en',
                                            from_='2021-10-25',
                                            to_='2021-10-31',
                                            page_size=100,page=i)['articles'])
    time.sleep(1)
                   
    return articles


N = 75

def animate_plot(companies, twitter, news):

    initial_height = [0 for i in companies]

    for source in [twitter, news]:
        for polarity in source:
            for i in range(len(polarity)):
                polarity[i] = np.linspace(0, polarity[i], num = N)
    
    x = np.arange(0, 2*len(companies) , 2) + 1 # the label locations
    width = 1.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    plt.ylim((0,1))
    plt.xlim((0,6))

    rects1 = ax.bar(x - (5*width)/12, initial_height, width/6, label='Twitter Negative', color = '#F8343F')
    rects2 = ax.bar(x - width/12, initial_height, width/6, label='Twitter Neutral', color = '#00E0FF')
    rects3 = ax.bar(x + width/4, initial_height, width/6, label='Twitter Positve', color = '#13B873')

    rects4 = ax.bar(x - width/4, initial_height, width/6, label='News Negative', color = '#F8343F', edgecolor = '#000000',linewidth =1, hatch= "x")
    rects5 = ax.bar(x + width/12, initial_height, width/6, label='News Neutral', color = '#00E0FF', edgecolor = '#000000',linewidth =1, hatch= "x")
    rects6 = ax.bar(x + (5*width)/12, initial_height, width/6, label='News Positve', color = '#13B873', edgecolor = '#000000',linewidth =1, hatch= "x")

    ax.set_title('Sentiment Analysis')
    ax.set_xticks(x, labels = companies)
    ax.legend()

    neg_rects = [r for r in rects1]
    neu_rects = [r for r in rects2]
    pos_rects = [r for r in rects3]
    T_rects = [neg_rects, neu_rects, pos_rects]

    neg_rects = [r for r in rects4]
    neu_rects = [r for r in rects5]
    pos_rects = [r for r in rects6]
    N_rects = [neg_rects, neu_rects, pos_rects]

    def animate(_):
        for tr, nr, t_sent, n_sent in zip(T_rects, N_rects, twitter, news):
            for j in range(3):
                tr[j].set_height(t_sent[j][_])
                nr[j].set_height(n_sent[j][_])

        return T_rects[0] + T_rects[1] + T_rects[2] + N_rects[0] + N_rects[1] + N_rects[2]

    anim = animation.FuncAnimation(fig, animate, frames=N, interval=5, repeat_delay = 1500, blit=True)

    plt.tight_layout()
    plt.show()
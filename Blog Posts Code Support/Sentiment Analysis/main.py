# Install package

# documentation https://docs.tweepy.org/en/latest/api.html
# Authentication

import os
import re
import tweepy
from tweepy import OAuthHandler
import numpy as np
import pandas as pd

# text treatement
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Wordcloud
from wordcloud import WordCloud,  ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Graphs
import plotly.io as pio
pio.renderers.default='browser'
import plotly.express as px

# transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForTokenClassification

consumer_key = os.environ['CONSUMER_KEY']
consumer_secret = os.environ['CONSUMER_SECRET']
access_token = os.environ['ACCESS_TOKEN']
access_token_secret = os.environ['ACCESS_SECRET']


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


variables_we_need = ['created_at', 'id', 'full_text', 'entities', 'user', 'coordinates', 'retweet_count', 'favorite_count', 'lang']


# Search method
"""
q – the search query string of 500 characters maximum, including operators. Queries may additionally be limited by complexity.
geocode – Returns tweets by users located within a given radius of the given latitude/longitude. The location is preferentially taking from the Geotagging API, but will fall back to their Twitter profile. The parameter value is specified by 'latitide,longitude,radius', where radius units must be specified as either 'mi' (miles) or 'km' (kilometers). Note that you cannot use the near operator via the API to geocode arbitrary locations; however you can use this geocode parameter to search near geocodes directly. A maximum of 1,000 distinct 'sub-regions' will be considered when using the radius modifier.
lang – Restricts tweets to the given language, given by an ISO 639-1 code. Language detection is best-effort.
locale – Specify the language of the query you are sending (only ja is currently effective). This is intended for language-specific consumers and the default should work in the majority of cases.
result_type –
Specifies what type of search results you would prefer to receive. The current default is 'mixed.' Valid values include:
    mixed : include both popular and real time results in the response
    recent : return only the most recent results in the response
    popular : return only the most popular results in the response
count – The number of results to try and retrieve per page.
until – Returns tweets created before the given date. Date should be formatted as YYYY-MM-DD. Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.
since_id – Returns only statuses with an ID greater than (that is, more recent than) the specified ID. There are limits to the number of Tweets which can be accessed through the API. If the limit of Tweets has occurred since the since_id, the since_id will be forced to the oldest ID available.
max_id – Returns only statuses with an ID less than (that is, older than) or equal to the specified ID.
include_entities – The entities node will not be included when set to false. Defaults to true.
"""

keyword = '#ios15'
count = 100

# Normal search
tweets = api.search(q=keyword, count=count)
# Extended with more info and full text
tweets = api.search(q=keyword, count=count, tweet_mode='extended')
# Lang
tweets = api.search(q=keyword, count=count, lang='ru', tweet_mode='extended')
# Advanced search search
tweets = api.search(q="gym"+'-filter:retweets', count=count, lang='ru', tweet_mode='extended')
# Time period
tweets = api.search(q=keyword, count=count, since='2021-09-20', tweet_mode='extended')



# Get one user
user = api.get_user("MikezGarcia")

print("User details:")
print(user.name)
print(user.description)
print(user.location)

print("Last 20 Followers:")
for follower in user.followers():
    print(follower.name)

# For trends
trends_result = api.trends_place(1)
for trend in trends_result[0]["trends"]:
    print(trend["name"])



# Part 0 Get trends

"""We begin with looking for current trends. \'id\' parameter stands fro WOEID geographical id of every city or village
 on the Earth. For tests, I chose New York City. But you can choose whatever you want, just find your location's ID on https://www.findmecity.com/.
 After finding trends, it can be interesting to visualize them using word cloud with Hashtag shape. """

trends_result = api.trends_place(id=2459115)[0]['trends']
trends = {}
for i in trends_result:
    trends[i['name']] = i['tweet_volume']
trends_names = ' '.join(list(trends.keys()))

# Image of corona
hashtag = np.array(Image.open("hashtag.jpg"))
hashtag[hashtag == 0] = 255

wordcloud = WordCloud(background_color="white",max_words=200, mask=hashtag, contour_width=3, contour_color='firebrick', collocations=False).generate(trends_names)
plt.figure(figsize=[20,10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Part 1 Get tweets

""" Trends can give us some ideas, but it is up to us to make a search. Using several available parameters of tweepy package, 
we make a search on Bitcoin. For the last week. I put 1000 tweets to extract, but you can try to get more. """

def get_all_tweets(count=100, q='', lang='', since='', tweet_mode='extended'):
    results = []

    tweets = tweepy.Cursor(api.search, q=q, lang=lang, since=since, tweet_mode=tweet_mode).items(count)

    for tweet in tweets:
        d = {}
        for variable in variables_we_need:
            d[variable] = tweet._json[variable]
        results.append(d)

    return results


results = get_all_tweets(count=1000, q='bitcoin', tweet_mode='extended', since='2021-09-13', lang='en')





# Part 2 Clean and divide into keywords
"""This is the most important part, because the cleaning phase will define whether our NLP algo will work or not. Here are the list of cleanings I apply:
- If it is a retweet, it begins with \"RT @account_name: \". I delete this part. 
    All account mentioned in the tweeter can be found in \"entities\" variable of each tweet
- Extract and then delete all links to other tweets 
- Delete all account mentions
- Tokenize words
- Remove punctuation
- Remove characters that are not alphabetic
- Remove stopwords

You do not have to apply all cleaning phases. depens on your use case"""

def clean_text(text, all_mentions):
    # If retweet, delete RT and name of the account
    text = re.sub('(RT\s.*):', '', text)
    # Find all links and delete them
    all_links = re.findall('(https:.*?)\s', text + ' ')

    for i in all_links:
        text = text.replace(i, '')

    for i in all_mentions:
        text = text.replace('@' + i, '')

    # Tokens
    tokens = word_tokenize(text.replace('-', ' '))
    # convert to lower case
    tokens = [w.lower() for w in tokens ]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    phrase = " ".join(words)

    return phrase, all_links


for i in results:
    i['clean_text'], i['all_link'] = clean_text(i['full_text'], [j['screen_name'] for j in i['entities']['user_mentions']])


"""
Let's do some visualization. For each tweet I extract:
- Hashtag mentioned in the tweet
- number_mentioned : How many times we have seen this hashtag
- retweet_count : Accumulate number of retweets for a tweet where this hashtag was mentioned.
- favorite_count : Accumulate number of how many time a tweet was put to \'favorite\'.

After that I try to visualize with plotly package
"""
all_hashtags = {}

for i in results:
    if i['entities']['hashtags']:
        for j in i['entities']['hashtags']:
            if j['text'].lower() in all_hashtags:
                all_hashtags[j['text'].lower()]['number_mentioned'] += 1
                all_hashtags[j['text'].lower()]['retweet_count'] += i['retweet_count']
                all_hashtags[j['text'].lower()]['favorite_count'] += i['favorite_count']
            else:
                all_hashtags[j['text'].lower()] = {'number_mentioned': 1, 'retweet_count': i['retweet_count'], 'favorite_count': i['favorite_count']}


all_hashtags_df = pd.DataFrame.from_dict(all_hashtags, orient='index')
all_hashtags_df = all_hashtags_df.reset_index()
all_hashtags_df.columns = ['hashtag', 'number_mentioned', 'retweet_count', 'favorite_count']
all_hashtags_df = all_hashtags_df.sort_values('number_mentioned', ascending=False)
all_hashtags_df = all_hashtags_df[all_hashtags_df['number_mentioned'] > 5]

# Number of hashtags and how many time there were mentioned in tweets
fig = px.histogram(all_hashtags_df, x='hashtag', y='number_mentioned', template='plotly_white', title='Hashtags by number of mentions')
fig.update_xaxes(categoryorder='total descending', title='Found Hashtags').update_yaxes(title='Number of mentions')
fig.show()

# Scatter plot, hashtags + retweeted + mentioned
fig = px.scatter(all_hashtags_df, x='number_mentioned', y='retweet_count', color='hashtag', title='Hashtags by number of mentions and retweets', size='number_mentioned', size_max=60, log_y=True)
fig.update_xaxes(categoryorder='total descending', title='Number of mentions').update_yaxes(title='Number of retweets')
fig.show()




# Part 3 Polarity
"""
In this part, I import finbert package to identify the polarity of tweets. 
"""


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

nlp = pipeline("sentiment-analysis", model = model, tokenizer=tokenizer)


possible_sentiments = ['negative', 'neutral', 'positive']

for tweet in results:
    sentiment = nlp(tweet['full_text'])
    for item in sentiment:
        for shade in possible_sentiments:
            if item['label'] == shade:
                tweet[shade] = item['score']
            else:
                tweet[shade] = 0



"""
Once again, we try to visualize, avg sentiment score for each hashtag. IMPORTANT: when I calculate the average, I take all 0s as well.
Negative and positive tweets are rare, that it why if we won't take 0s into account. The value can be very high, but it comes
from 4-5 tweets. And taking 0s into account will represent the real average. 
"""
all_hashtags2 = {}

for i in results:
    if i['entities']['hashtags']:
        for j in i['entities']['hashtags']:
            if j['text'].lower() in all_hashtags2:
                all_hashtags2[j['text'].lower()]['number_mentioned'] += 1
                all_hashtags2[j['text'].lower()]['retweet_count'] += i['retweet_count']
                all_hashtags2[j['text'].lower()]['favorite_count'] += i['favorite_count']
                all_hashtags2[j['text'].lower()]['negative'].append(i['negative'])
                all_hashtags2[j['text'].lower()]['neutral'].append(i['neutral'])
                all_hashtags2[j['text'].lower()]['positive'].append(i['positive'])
            else:
                all_hashtags2[j['text'].lower()] = {'number_mentioned': 1, 'retweet_count': i['retweet_count'],
                                                   'favorite_count': i['favorite_count'], 'negative': [i['negative']],
                                                   'neutral': [i['neutral']], 'positive': [i['positive']]}
for j in all_hashtags2.keys():
    all_hashtags2[j]['negative_avg'] = np.mean([i for i in all_hashtags2[j]['negative']]) #if i != 0])
    all_hashtags2[j]['neutral_avg'] = np.mean([i for i in all_hashtags2[j]['neutral']])  # if i != 0])
    all_hashtags2[j]['positive_avg'] = np.mean([i for i in all_hashtags2[j]['positive']])  # if i != 0])



all_hashtags_df2 = pd.DataFrame.from_dict(all_hashtags2, orient='index')
all_hashtags_df2 = all_hashtags_df2.reset_index()
all_hashtags_df2 = all_hashtags_df2.drop(['negative', 'neutral', 'positive'], axis=1)
all_hashtags_df2.columns = ['hashtag', 'number_mentioned', 'retweet_count', 'favorite_count', 'negative_avg', 'neutral_avg', 'positive_avg']
all_hashtags_df2 = all_hashtags_df2.sort_values('number_mentioned', ascending=False)
all_hashtags_df2 = all_hashtags_df2[all_hashtags_df2['number_mentioned'] > 5]


# Sentiment score by hashtag
fig = px.histogram(all_hashtags_df2, x='hashtag', title='Sentiment Score by Hashtag', y= ['negative_avg', 'neutral_avg','positive_avg'], barmode='group')
fig.update_xaxes( title='Hashtags').update_yaxes(title='Sentiment score')
fig.show()




# Part 4 Bert base
# Bert NER

"""
This is the shitty part, because the model does not identify a thing. So I am not able even to visualize something. 
"""
tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_ner  = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp_ner = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner)

for tweet in results:
    ner_results = nlp_ner(tweet['clean_text'])
    tweet['entities_nlp'] = []
    for ent in ner_results:
        tweet['entities_nlp'].append({ent['word']: ent['entity']})



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


# Connections
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


"""
Part I: Get trends on Twitter based on geographical location 
"""

trends_result = api.trends_place(id=2459115)[0]['trends']
trends = {}
for i in trends_result:
    trends[i['name']] = i['tweet_volume']
    print(f"{i['name']} => {trends[i['name']]}")
trends_names = ' '.join(list(trends.keys()))



# Visualize treds
hashtag = np.array(Image.open("hashtag.jpg"))
hashtag[hashtag == 0] = 255
wordcloud = WordCloud(background_color="white",max_words=200, mask=hashtag, contour_width=3, contour_color='firebrick', collocations=False).generate(trends_names)
plt.figure(figsize=[20,10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



"""
Part II: Get Tweets using a Keyword
"""
tweets = tweepy.Cursor(api.search, q='Apple', tweet_mode='extended').items(1)
for tweet in tweets:
    one_tweet = tweet

variables_we_need = ['created_at', 'id', 'full_text', 'entities', 'coordinates', 'retweet_count', 'favorite_count', 'lang']

def get_all_tweets(count=100, q='', lang='', since='', until='', tweet_mode='extended'):
    results = []

    tweets = tweepy.Cursor(api.search, q=q, lang=lang, since=since, until=until, tweet_mode=tweet_mode).items(count)

    for tweet in tweets:
        d = {}
        for variable in variables_we_need:
            d[variable] = tweet._json[variable]
        results.append(d)

    return results


results_apple = get_all_tweets(count=1000, q='apple', tweet_mode='extended', lang='en', since='2021-10-25', until='2021-10-31')
results_facebook = get_all_tweets(count=1000, q='facebook', tweet_mode='extended', lang='en', since='2021-10-25', until='2021-10-31')
results_amazon = get_all_tweets(count=1000, q='amazon', tweet_mode='extended', lang='en', since='2021-10-25', until='2021-10-31')

"""
Part 3: Clean and tokenize content of tweets
"""


def clean_text(text, all_mentions):
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

    return phrase, all_links


for i in results_apple:
    i['clean_text'], i['all_link'] = clean_text(i['full_text'],
                                                [j['screen_name'] for j in i['entities']['user_mentions']])
for i in results_facebook:
    i['clean_text'], i['all_link'] = clean_text(i['full_text'],
                                                [j['screen_name'] for j in i['entities']['user_mentions']])
for i in results_amazon:
    i['clean_text'], i['all_link'] = clean_text(i['full_text'],
                                                [j['screen_name'] for j in i['entities']['user_mentions']])


# Visualize every hashtag mentioned in a tweet
def get_hashtags(input_tweets):

    all_hashtags = {}

    for i in input_tweets:
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

    return all_hashtags_df

hashtags_apple = get_hashtags(results_apple)
hashtags_facebook = get_hashtags(results_facebook)
hashtags_amazon = get_hashtags(results_amazon)

# Display Apple hashtags
fig = px.histogram(hashtags_apple, x='hashtag', y='number_mentioned', template='plotly_white', title='Hashtags by number of mentions | Apple')
fig.update_xaxes(categoryorder='total descending', title='Found Hashtags').update_yaxes(title='Number of mentions')
fig.show()


# Display Facebook hashtags
fig = px.histogram(hashtags_facebook, x='hashtag', y='number_mentioned', template='plotly_white', title='Hashtags by number of mentions | Facebook')
fig.update_xaxes(categoryorder='total descending', title='Found Hashtags').update_yaxes(title='Number of mentions')
fig.show()

# Display Amazon hashtags
fig = px.histogram(hashtags_amazon, x='hashtag', y='number_mentioned', template='plotly_white', title='Hashtags by number of mentions | Amazon')
fig.update_xaxes(categoryorder='total descending', title='Found Hashtags').update_yaxes(title='Number of mentions')
fig.show()



# Number of mentions of a hashtag / number of times tweets containing the hashtag were retweeted
# Apple
fig = px.scatter(hashtags_apple, x='number_mentioned', y='retweet_count', color='hashtag', title='Hashtags by number of mentions and retweets', size='number_mentioned', size_max=60, log_y=True)
fig.update_xaxes(categoryorder='total descending', title='Number of mentions').update_yaxes(title='Number of retweets')
fig.show()

# Facebook
fig = px.scatter(hashtags_facebook, x='number_mentioned', y='retweet_count', color='hashtag', title='Hashtags by number of mentions and retweets', size='number_mentioned', size_max=60, log_y=True)
fig.update_xaxes(categoryorder='total descending', title='Number of mentions').update_yaxes(title='Number of retweets')
fig.show()

#Amazon
fig = px.scatter(hashtags_amazon, x='number_mentioned', y='retweet_count', color='hashtag', title='Hashtags by number of mentions and retweets', size='number_mentioned', size_max=60, log_y=True)
fig.update_xaxes(categoryorder='total descending', title='Number of mentions').update_yaxes(title='Number of retweets')
fig.show()



"""
Part 4: Sentiment Analysis  
"""

# Start Using Sentiment Analysis financial oriented package
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


results_apple = get_sentiments(results_apple, 'clean_text')
results_facebook = get_sentiments(results_facebook, 'clean_text')
results_amazon = get_sentiments(results_amazon, 'clean_text')



#Visualization
apple_tweets_pd = pd.DataFrame(results_apple).loc[:, ['negative', 'neutral', 'positive']]
facebook_tweets_pd = pd.DataFrame(results_facebook).loc[:, ['negative', 'neutral', 'positive']]
amazon_tweets_pd = pd.DataFrame(results_amazon).loc[:, ['negative', 'neutral', 'positive']]

total_score_tweets = pd.concat([apple_tweets_pd.mean(), facebook_tweets_pd.mean(), amazon_tweets_pd.mean()], axis=1)
total_score_tweets = total_score_tweets.transpose()
total_score_tweets = total_score_tweets.reset_index()
total_score_tweets.columns = ['Company', 'negative', 'neutral', 'positive']
total_score_tweets['Company'] = ['Apple', 'Facebook', 'Amazon']

fig = px.histogram(total_score_tweets,
                   x='Company',
                   title='Sentiment Score by Company | Tweets',
                   y= ['negative', 'neutral','positive'],
                   barmode='group',
                  color_discrete_sequence=["red", "blue", "green"])
fig.update_xaxes( title='Companies').update_yaxes(title='Sentiment score')
fig.show()


# NewsCatcher News API
from newscatcherapi import NewsCatcherApiClient
import time

newscatcherapi = NewsCatcherApiClient(x_api_key='YOUR-X-API-KEY')

apple_articles = []
facebook_articles = []
amazon_articles = []

for i in range(1, 11):
    apple_articles.extend(newscatcherapi.get_search(q='(Apple AND company) OR "Apple Inc"',
                                                    lang='en',
                                                    from_='2021-10-25',
                                                    to_='2021-10-31',
                                                    page_size=100,
                                                    page=i)['articles'])
    time.sleep(1)

    facebook_articles.extend(newscatcherapi.get_search(q='(Facebook AND company) OR "Facebook Inc"',
                                                       lang='en',
                                                       from_='2021-10-25',
                                                       to_='2021-10-31',
                                                       page_size=100,
                                                       page=i)['articles'])

    time.sleep(1)

    amazon_articles.extend(newscatcherapi.get_search(q='(Amazon AND company) OR "Amazon Inc"',
                                                     lang='en',
                                                     from_='2021-10-25',
                                                     to_='2021-10-31',
                                                     page_size=100,
                                                     page=i)['articles'])

    time.sleep(1)


apple_articles_pd = pd.DataFrame(get_sentiments(apple_articles, 'title')).loc[:, ['negative', 'neutral', 'positive']]
facebook_articles_pd = pd.DataFrame(get_sentiments(facebook_articles, 'title')).loc[:, ['negative', 'neutral', 'positive']]
amazon_articles_pd = pd.DataFrame(get_sentiments(amazon_articles, 'title')).loc[:, ['negative', 'neutral', 'positive']]



#Visualize
total_score_articles = pd.concat([apple_articles_pd.mean(), facebook_articles_pd.mean(), amazon_articles_pd.mean()], axis=1)
total_score_articles = total_score_articles.transpose()
total_score_articles = total_score_articles.reset_index()
total_score_articles.columns = ['Company', 'negative', 'neutral', 'positive']
total_score_articles['Company'] = ['Apple', 'Facebook', 'Amazon']

fig = px.histogram(total_score_articles,
                   x='Company',
                   title='Sentiment Score by Company | News Articles',
                   y= ['negative', 'neutral','positive'],
                   barmode='group',
                  color_discrete_sequence=["red", "blue", "green"])
fig.update_xaxes( title='Companies').update_yaxes(title='Sentiment score')
fig.show()
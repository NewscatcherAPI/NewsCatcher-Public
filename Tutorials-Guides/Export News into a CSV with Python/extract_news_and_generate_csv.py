# Step 1. Prepare Environment

# Import packages
# Default packages
import time
import csv
import os
import json

# Preinstalled packages
import requests
import pandas as pd



# Define desired work folder, where you want to save your .csv files
# Windows Example
os.chdir('C:\\Users\\user_name\\PycharmProjects\\extract_news_data')
# Linux Example
os.chdir('/mnt/c/Users/user_name/PycharmProjects/extract_news_data')

# URL of our News API
base_url = 'https://api.newscatcherapi.com/v2/search'
# Your API key
X_API_KEY = 'PUT_YOUR_API_KEY'





# Step 2. Make an API call

# Put your API key to headers in order to be authorized to perform a call
headers = {'x-api-key': X_API_KEY}

# Define your desired parameters
params = {
    'q': 'Bitcoin AND Ethereum AND Dogecoin',
    'lang': 'en',
    'to_rank': 10000,
    'page_size': 100,
    'page': 1
    }

# Make a simple call with both headers and params
response = requests.get(base_url, headers=headers, params=params)

# Encode received results
results = json.loads(response.text.encode())
if response.status_code == 200:
    print('Done')
else:
    print(results)
    print('ERROR: API call failed.')


# Import data into pandas
pandas_table = pd.DataFrame(results['articles'])





# Step 3. Extract All Found News Articles

# Variable to store all found news articles
all_news_articles = []

# Ensure that we start from page 1
params['page'] = 1

# Infinite loop which ends when all articles are extracted
while True:

    # Wait for 1 second between each call
    time.sleep(1)

    # GET Call from previous section enriched with some logs
    response = requests.get(base_url, headers=headers, params=params)
    results = json.loads(response.text.encode())
    if response.status_code == 200:
        print(f'Done for page number => {params["page"]}')


        # Adding your parameters to each result to be able to explore afterwards
        for i in results['articles']:
            i['used_params'] = str(params)


        # Storing all found articles
        all_news_articles.extend(results['articles'])

        # Ensuring to cover all pages by incrementing "page" value at each iteration
        params['page'] += 1
        if params['page'] > results['total_pages']:
            print("All articles have been extracted")
            break
        else:
            print(f'Proceed extracting page number => {params["page"]}')
    else:
        print(results)
        print(f'ERROR: API call failed for page number => {params["page"]}')
        break

print(f'Number of extracted articles => {str(len(all_news_articles))}')



# Extract articles with multiple parameters
# Define your desired parameters
params = [
    {
        'q': 'Bitcoin',
        'lang': 'en',
        'to_rank': 10000,
        'topic': "business",
        'page_size': 100,
        'page': 1
    },
    {
        'q': 'Ethereum',
        'lang': 'en',
        'to_rank': 10000,
        'topic': "business",
        'page_size': 100,
        'page': 1
    },
    {
        'q': 'Dogecoin',
        'lang': 'en',
        'to_rank': 10000,
        'topic': "business",
        'page_size': 100,
        'page': 1
    },
]

# Variable to store all found news articles, mp stands for "multiple queries"
all_news_articles_mp = []

# Infinite loop which ends when all articles are extracted
for separated_param in params:
    print(f'Query in use => {str(separated_param)}')
    while True:
        # Wait for 1 second between each call
        time.sleep(1)

        # GET Call from previous section enriched with some logs
        response = requests.get(base_url, headers=headers, params=separated_param)
        results = json.loads(response.text.encode())
        if response.status_code == 200:
            print(f'Done for page number => {separated_param["page"]}')


            # Adding your parameters to each result to be able to explore afterwards
            for i in results['articles']:
                i['used_params'] = str(separated_param)


            # Storing all found articles
            all_news_articles_mp.extend(results['articles'])

            # Ensuring to cover all pages by incrementing "page" value at each iteration
            separated_param['page'] += 1
            if separated_param['page'] > results['total_pages']:
                print("All articles have been extracted")
                break
            else:
                print(f'Proceed extracting page number => {separated_param["page"]}')
        else:
            print(results)
            print(f'ERROR: API call failed for page number => {separated_param["page"]}')
            break

print(f'Number of extracted articles => {str(len(all_news_articles_mp))}')


# Define variables
unique_ids = []
all_news_articles = []

# Iterate on each article and check whether we saw this _id before
for article in all_news_articles_mp:
    if article['_id'] not in unique_ids:
        unique_ids.append(article['_id'])
        all_news_articles.append(article)






# Step 4.1. Generate CSV file from dict
field_names = list(all_news_articles[0].keys())
with open('extracted_news_articles.csv', 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=";")
    writer.writeheader()
    writer.writerows(all_news_articles)




# Step 4.2. Generate CSV from Pandas table
# Create Pandas table
pandas_table = pd.DataFrame(all_news_articles)

# Generate CSV
pandas_table.to_csv('extracted_news_articles.csv', encoding='utf-8', sep=';')

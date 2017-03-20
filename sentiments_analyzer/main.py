import tweepy
from textblob import TextBlob
import numpy as np
import csv

consumer_key = 'CONSUMER_KEY_HERE'
consumer_secret = 'CONSUMER_SECRET_HERE'

access_token = '4120394458-ACCESS_TOKEN_HERE'
access_token_secret = 'ACCESS_TOKEN_SECRET_HERE'

word_to_search = 'Trump'
csv_file_path = 'results.csv'

# Authenticating with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Searching the keywords
public_tweets = api.search(word_to_search, count=50, lang='en')

tweet_sentiment_list = []
total_polarity = 0
total_subjectivity = 0

# Analyzing every tweet
for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    # Storing values in a list for later processing
    tweet_sentiment_list.append([tweet.text.encode('utf-8'), analysis.polarity, analysis.subjectivity])

    # Calculating the total for polarity and subjectivity
    if analysis.polarity != 0:
        total_polarity += analysis.polarity
        total_subjectivity += analysis.subjectivity

# Writting the csv file
csv_file = open(csv_file_path, "wb")
csvwritter = csv.writer(csv_file)
csvwritter.writerow(['Tweet', 'Polarity', 'Subjectivity'])

for row in tweet_sentiment_list:
    csvwritter.writerow(row)

csv_file.close()

# Creating an array to extract data
tweets_array = np.array(tweet_sentiment_list)
# Need to create a polarity array of type float to order it properly
polarity_array = np.array(tweets_array[:, 1], dtype=np.float32)

# Getting the max polarity index
highest_polarity_index = np.argmax(polarity_array)
# Getting the min polarity index
lowest_polarity_index = np.argmin(polarity_array)

# Getting averages of polarity and subjectivity
avg_polarity = total_polarity / len(tweets_array)
avg_subjectivity = total_subjectivity / len(tweets_array)

# Printing Results
print ('>>>> Results for workd "{}" <<<< \n\n'.format(word_to_search))
print tweets_array

# The polarity score is a float within the range [-1.0, 1.0]
print ('\nOverall Sentiment polarity is: {}'.format(avg_polarity))
# The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
print ('Overall Subjectivity is: {}'.format(avg_subjectivity))

print ('\nMost positive tweet:')
print (tweets_array[highest_polarity_index][0])
print ('Polarity Index: {}'.format(tweets_array[highest_polarity_index][1]))
print ('Subjectivity Index: {}'.format(tweets_array[highest_polarity_index][2]))

print ('\nMost negative tweet:')
print (tweets_array[lowest_polarity_index][0])
print ('With Polarity: {}'.format(tweets_array[lowest_polarity_index][1]))
print ('Subjectivity Index: {}'.format(tweets_array[lowest_polarity_index][2]))

print ('\nNumbers explanation:')
print ('Sentiment Polarity: -1 very negative, 1 very positive')
print ('Subjectivity: 0 very objective, 1 very subjective')

from textblob import TextBlob

wiki = TextBlob("I am super happy")

print ('Tags: {}'.format(wiki.tags))
print ('Words: {}'.format(wiki.words))

#Sentiment polarity is -1 < sentiment < 1
print ('Sentiment polarity: {!r}'.format(wiki.sentiment.polarity))
print ('Sentiment subjectivity: {!r}'.format(wiki.sentiment.subjectivity))

from textblob import TextBlob
import nltk


sampleText = TextBlob("I am super happy")

#For this we need to download the tokeninzer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print ('Tags: {}'.format(sampleText.tags))
print ('Words: {}'.format(sampleText.words))

#Sentiment polarity is -1 < sentiment < 1
print ('Sentiment polarity: {!r}'.format(sampleText.sentiment.polarity))
print ('Sentiment subjectivity: {!r}'.format(sampleText.sentiment.subjectivity))

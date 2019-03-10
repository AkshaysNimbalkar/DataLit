# pip install pandas  # Data preprocessing
# pip install tweepy  # Access Twitter api
# pip install vaderSentiment # sentiment analysis : by looking in words that built in lexicon
# pip install nltk

# step2 - import dependancies
import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# step 3 - My Twitter API Authentication Variables
consumer_key = 'Kugyq3pimt5qJSYxKUcgAVz2U'
consumer_secret = 'rPbynzelqGbt1v96v1JNJxYvY8oPBY5KRXFholde7HO5HN94Y0'
access_token = '321565644-P0tRK4WKujvewQm5A6qDAPp7jXcszXkpJaeaKVsO'
access_token_secret = 'xOqg8fs43ggl2d5KxnIOoDT2bnV3W5OsyC4RpSjAdgbul'

# step 4 - Authenticate with twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# optional
# step 5 - Define cleaning function: #function to clean tweets by removing spl. characters and links using regx
import re


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\s+)", " ", tweet).split())



# step 6 - FInd related tweets
api = tweepy.API(auth)
tweets = api.search('Terrorist Attack in India', count=200)

data = pd.DataFrame(data=[clean_tweet(tweet.text) for tweet in tweets], columns=['tweets'])
print(data.head(10))
type(data)
# Meta data from single tweet
print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].text)
print(tweets[0].geo)

####################
data['tidy_tweet'] = data['tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if w.isalpha()]))

tokenized_tweet = data['tidy_tweet'].apply(lambda x : x.lower().split())
type(tokenized_tweet)
tokenized_tweet.head()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [porter.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

data['tidy_tweet'] = tokenized_tweet

# A wordcloud is a visualization wherein the most frequent words appear in large size
# and the less frequent words appear in smaller sizes.

all_words = ' '.join([text for text in data['tidy_tweet']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#del(data['Tweets'])

# gather lexicon data
import nltk
nltk.download('vader_lexicon')



# step 7 - Go through the tweets to analyse their sentiment
sid = SentimentIntensityAnalyzer()

listy = []

for index, row in data.iterrows():
    sentiment = sid.polarity_scores(row["tidy_tweet"])
    #print(ss['compound'])
    listy.append([sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']])


data[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(listy)

data['Negative'] = data['compound'] < -0.1
data['Positive'] = data['compound'] > 0.1

data.iloc[:]['compound'] > 0.1

positive_words = ' '.join([text for text in data['tidy_tweet'][data.iloc[:]['compound'] > 0.1]])
negative_words = ' '.join([text for text in data['tidy_tweet'][data.iloc[:]['compound'] < 0.1]])

type(negative_words)

def word_count(str):
    counts = {}
    words = str.split()
    for word in words:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
    return counts

negative_dict = word_count(negative_words)
del(negative_dict['http'])


d = {'Word': list(negative_dict.keys()),'Word_Count': list(negative_dict.values())}
df = pd.DataFrame(data=d)

import seaborn as sns
sns.set_style("whitegrid")
sns.barplot(data=df.head(19), x= "Word_Count", y = "Word")
plt.ylabel("Words", fontsize=15)
plt.xlabel("Frequency", fontsize=15)
plt.title("Harsh words vs Count")


wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()




############ Bag-of-Words Features

from sklearn.feature_extraction.text import CountVectorizer
bow_vector = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vector.fit_transform(data['tidy_tweet'])

########### TF-IDF Features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vector.fit_transform(data['tidy_tweet'])


wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()









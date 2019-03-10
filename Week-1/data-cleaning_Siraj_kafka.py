# Manual Cleaning ##########
file = open("metamorphosis_clean.txt", 'r')
# print(file.read())
text = file.read()
file.close()

# split into words by white space
words = text.split()
print(words[:100])

# split based on only words
import re
#words = re.split(r'\W+', text)
#print(words[:100])

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation) #T his static method returns a translation table usable for str.translate()
stripped = [w.translate(table) for w in words]
print(stripped[:100]) # Finally, we got "wasnt" from "wasn't".

lower_stripped = [w.lower() for w in stripped]
print(lower_stripped[:100]) # It si same like NLTK output


# automatic using NLTK

########################################################### 1. Install NLTK
# pip install -U nltk
# python -m nltk.downloader all

########################################################### 2. Tokenization
# Split into Words

# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

#  1. split into words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]
#print(tokens[:100]) # same as string tokenization method

#   3. Filter Out Punctuation
# Remove all the words that are not alphabetic
# alternate for regx
words = []
for word in tokens:
    if word.isalpha():
        words.append(word)
print(words[:100]) # It provides a high-level api, it can be easily handled by a function called isalpha().

# alternative way of writting above loop :
# words = [word for word in tokens if word.isalpha()]

#  2.   Sentence Tokenization
from nltk import sent_tokenize
sentences = sent_tokenize(text)
print(sentences[:100])

############################################# Step - 3  Stopwprds/Stemming

# 1. Filter out Stopwords and Pipelines
# Word like “the” or “and” can be removed by comparing text to a list of stopwords.

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)

words = [w for w in words if not w in stop_words]
print(words[:100])

# 2. Stemming
# Stemming is a process where words are reduced to a root by removing inflection through
# dropping unnecessary characters,usually a suffix

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

stemmed_words = [porter.stem(w) for w in words]
print(stemmed_words[:100])



def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
mytrippler = myfunc(3)

print(mydoubler(11))
print(mytrippler(11))

x = lambda a : a + 5
print(x(5))

month_nums = [str(x+1).zfill(2) for x in range(12)]
month_nums

a = [str(x+1).zfill(3) for x in range(12)]
a
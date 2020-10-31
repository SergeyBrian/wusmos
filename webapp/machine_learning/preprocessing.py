from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

import re

import pandas
import numpy as np

lemmatizer = WordNetLemmatizer()

text = train.iloc[0]["text"][:len(train.iloc[0]["text"]) // 2]


def initialize():
    nltk.download('wordnet')
    nltk.download('punkt')


def cleaning(raw_text):
    # print('Preprocessing...')
    html_text = BeautifulSoup(raw_text, "html.parser").get_text()
    letters = re.sub("[^a-zA-Z]", " ", html_text)
    letters = letters.lower()
    tokens = nltk.word_tokenize(letters)
    stops = set(nltk.corpus.stopwords.words("english"))
    words = [w for w in tokens if w not in stops]
    words = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]

    return " ".join(words)


# Returns cleaned and vectorized dataframe
def preprocess(data_frame):
    print('Cleaning content... [1/2]')
    data_frame['clean'] = data_frame['text'].apply(cleaning)

    print('Updating tables... [2/2]')
    data_frame['freq_word'] = data_frame['clean'].apply(lambda x: len(str(x).split()))
    data_frame['unique_freq_word'] = data_frame['clean'].apply(lambda x: len(set(str(x).split())))

    return data_frame



from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

import re

import pandas

from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()


nltk.download('wordnet')
nltk.download('punkt')


def cleaning(raw_text):
    # print('Preprocessing...')
    letters = re.sub("[^a-zA-Z]", " ", raw_text)
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


def import_train_data(public_data_frame_path, split=False):
    public_data_frame = pandas.read_csv(public_data_frame_path)
    if split:
        train = public_data_frame.truncate(after=50001)
        test = public_data_frame.truncate(before=50000)
        return train, test
    return public_data_frame


def create_vectorizer():
    return CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                           max_features=18000, min_df=2, ngram_range=(1, 3))


def create_bag_of_words(data_frame, vectorizer):
    pipe = Pipeline([('vect', create_vectorizer())])
    bag_of_words = pipe.fit_transform(data_frame['clean'])

    bw_dict = vectorizer.get_feature_names()
    res = pandas.DataFrame(np.sum(bag_of_words, axis=0), columns=bw_dict)
    return res


def create_model(train, train_bw):
    kfold = StratifiedKFold()
    svc = LinearSVC()

    params = {
        'loss': ['hinge'],
        'class_weight': [{1: 1}],
        'C': [0.01]
    }

    model = GridSearchCV(svc, param_grid=[params], verbose=1, cv=kfold, n_jobs=-1, scoring='roc_auc')
    model.fit(train_bw, train['target'])
    return model


# test should be result of create_bag_of_words function
# model should be result of create_model function
def predict(test, model):
    return model.predict(test)


public = 'twitter_data_public.csv'
private = 'twitter_data_private_nolabels.csv'

# initialize()
ltrain, ltest = import_train_data(public, True)
private = import_train_data(private)

print('what?')


ltrain, ltest, private = preprocess(ltrain), preprocess(ltest), preprocess(private)
print(ltrain)
print(ltest)
print(private)
